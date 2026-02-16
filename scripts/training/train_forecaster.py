"""
Training script for the Latent Space Forecaster.

This script trains a neural network to predict future VAE latent states
from historical latents and context variables. This enables autoregressive
projection of emissions for 2030 targets.

The forecaster predicts z_t from [z_{t-1}, z_{t-2}, context_t, context_{t-1}],
learning the temporal evolution of the latent space while preserving the
correlation structure encoded by the VAE.

Prerequisites:
    - Trained VAE model (data/pytorch_models/vae_model.pth)

Usage:
    python -m scripts.training.train_forecaster

Outputs:
    - data/pytorch_models/forecaster_model.pth: Best model weights

Reference:
    Section 4.2.3 "Latent-Space Forecasting Model" in the paper.
"""

import multiprocessing as mp

# Set multiprocessing start method before other imports
mp.set_start_method("spawn", force=True)

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.data.output_configs import output_configs
from scripts.elements.datasets import DatasetForecasting, DatasetUnified
from scripts.elements.models import (
    Decoder,
    Encoder,
    FullLatentForecastingModel,
    LatentForecaster,
    VAEModel,
    uncertainty_aware_mae_loss,
)
from scripts.utils import (
    check_nan_gradients,
    count_parameters,
    init_weights,
    load_config,
)


def main():
    """
    Main training loop for the latent forecaster.

    Training procedure:
    1. Load pre-trained VAE and freeze its weights
    2. Build latent forecaster network
    3. Create combined model (VAE encoder + forecaster)
    4. Train using uncertainty-aware MAE loss
    5. Save best model based on smoothed validation accuracy
    """
    # ==========================================================================
    # Configuration
    # ==========================================================================

    vae_config = load_config("config/models/vae_config.yaml")
    forecast_config = load_config("config/models/latent_forecaster_config.yaml")

    # Training hyperparameters
    batch_size = 128
    num_epochs = 5001
    scaling_type = "normalization"

    # VAE architecture (must match trained model)
    vae_latent_dim = vae_config.vae_latent_dim
    vae_num_blocks = vae_config.vae_num_blocks
    vae_dim_blocks = vae_config.vae_dim_blocks
    vae_activation = vae_config.vae_activation
    vae_normalization = vae_config.vae_normalization
    vae_dropout = vae_config.vae_dropouts
    vae_input_dropout = vae_config.vae_input_dropouts

    # Forecaster architecture (from config)
    forecast_width = forecast_config.forecast_width_block
    forecast_dim = forecast_config.forecast_dim_block
    forecast_num_blocks = forecast_config.forecast_num_blocks
    forecast_activation = forecast_config.forecast_activation
    forecast_normalization = forecast_config.forecast_normalization
    forecast_dropout = forecast_config.forecast_dropouts

    # Optimizer settings (from config)
    learning_rate = forecast_config.forecast_lr
    weight_decay = forecast_config.forecast_wd
    optimizer_name = forecast_config.forecast_optimizer

    # Paths
    vae_model_path = Path("data/pytorch_models/vae_model.pth")
    forecaster_model_path = Path("data/pytorch_models/forecaster_model.pth")
    variable_file = Path("config/data/variable_selection.txt")

    # Ensure output directories exist
    forecaster_model_path.parent.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Data Loading
    # ==========================================================================

    # Load variable selection
    with open(variable_file) as f:
        nested_variables = [line.strip() for line in f if line.strip()]

    # Countries to include (EU27)
    eu27_countries = [
        "AT",
        "BE",
        "BG",
        "HR",
        "CY",
        "CZ",
        "DK",
        "EE",
        "EL",
        "FI",
        "FR",
        "DE",
        "HU",
        "IE",
        "IT",
        "LV",
        "LT",
        "LU",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SK",
        "SI",
        "ES",
        "SE",
    ]

    print("Creating dataset...")
    base_dataset = DatasetUnified(
        path_csvs="data/full_timeseries/",
        output_configs=output_configs,
        select_years=np.arange(2010, 2023 + 1),
        select_geo=eu27_countries,
        nested_variables=nested_variables,
        with_cuda=True,
        scaling_type=scaling_type,
    )

    # Wrap in forecasting dataset (provides 3 time steps: t, t-1, t-2)
    full_dataset = DatasetForecasting(base_dataset)

    # Train/validation split
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [0.85, 0.15]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    print("Data setup done")
    print(f"  Dataset: {full_dataset.base_length} unique samples (5x inflated)")
    print(f"  Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # ==========================================================================
    # Model Setup
    # ==========================================================================

    input_dim = len(base_dataset.input_variable_names)
    context_dim = len(base_dataset.context_variable_names)

    # Build VAE (architecture must match trained model)
    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=vae_latent_dim,
        num_blocks=vae_num_blocks,
        dim_blocks=vae_dim_blocks,
        activation=vae_activation,
        normalization=vae_normalization,
        dropout=vae_dropout,
        input_dropout=vae_input_dropout,
    ).cuda()

    decoder = Decoder(
        input_dim=input_dim,
        latent_dim=vae_latent_dim,
        num_blocks=vae_num_blocks,
        dim_blocks=vae_dim_blocks,
        activation=vae_activation,
        normalization=vae_normalization,
        dropout=vae_dropout,
    ).cuda()

    vae_model = VAEModel(encoder, decoder).cuda()

    # Load pre-trained VAE weights
    print(f"Loading pre-trained VAE from {vae_model_path}")
    vae_model.load_state_dict(torch.load(vae_model_path))

    # Build latent forecaster
    # Input: [z_{t-1}, z_{t-2}, context_t, context_{t-1}]
    forecaster_input_dim = 2 * (vae_latent_dim + context_dim)

    forecaster = LatentForecaster(
        input_dim=forecaster_input_dim,
        latent_dim=vae_latent_dim,
        num_blocks=forecast_num_blocks,
        dim_block=forecast_dim,
        width_block=forecast_width,
        activation=forecast_activation,
        normalization=forecast_normalization,
        dropout=forecast_dropout,
    ).cuda()
    forecaster.apply(init_weights)

    # Combine into full model
    full_model = FullLatentForecastingModel(vae=vae_model, forecaster=forecaster)

    print(f"Latent forecaster created with {count_parameters(forecaster):,} parameters")
    print(f"  Input dimension: {forecaster_input_dim}")
    print(f"  Output latent dimension: {vae_latent_dim}")

    # ==========================================================================
    # Optimizer Setup
    # ==========================================================================

    # Freeze VAE weights via requires_grad
    for p in full_model.encoder.parameters():
        p.requires_grad = False
    for p in full_model.vae.parameters():
        p.requires_grad = False

    print("VAE weights frozen for forecaster training")

    optimizer_cls = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "radam": torch.optim.RAdam,
    }.get(optimizer_name.lower())

    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    optimizer = optimizer_cls(
        [{"params": full_model.forecaster.parameters(), "lr": learning_rate}],
        weight_decay=weight_decay,
        eps=1e-6,
    )

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Optimizer: {optimizer_name}, LR: {learning_rate}")

    best_val_loss = None
    best_val_loss_smooth = None
    best_val_acc = None
    best_val_acc_smooth = None

    val_loss_history = deque(maxlen=50)
    val_acc_history = deque(maxlen=50)

    for epoch in range(num_epochs):
        # ----------------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------------
        full_model.train()
        train_loss = 0.0

        for batch in train_loader:
            (
                input_current,  # x_t (target)
                context_current,  # c_t
                input_prev,  # x_{t-1}
                context_prev,  # c_{t-1}
                input_past,  # x_{t-2}
                context_past,  # c_{t-2} (not used in current formulation)
            ) = batch

            optimizer.zero_grad()

            # Forward pass: predict z_t from [z_{t-1}, z_{t-2}, c_t, c_{t-1}]
            forecasted_latent = full_model(
                input_prev, input_past, context_current, context_prev
            )

            # Get target latent distribution parameters
            mean_target, log_var_target = full_model.encoder(input_current)

            # Loss: uncertainty-aware MAE
            loss = uncertainty_aware_mae_loss(
                mean_target,
                forecasted_latent,
                log_var_target,
                mode="regular",
            )

            # Backward pass — NaN check before clip
            loss.backward()
            check_nan_gradients(full_model)
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ----------------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------------
        full_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.inference_mode():
            for batch in val_loader:
                (
                    input_current,
                    context_current,
                    input_prev,
                    context_prev,
                    input_past,
                    context_past,
                ) = batch

                # Forward pass
                forecasted_latent = full_model(
                    input_prev, input_past, context_current, context_prev
                )

                # Get target latent
                mean_target, log_var_target = full_model.encoder(input_current)

                # Loss
                loss = uncertainty_aware_mae_loss(
                    mean_target,
                    forecasted_latent,
                    log_var_target,
                    mode="regular",
                )
                val_loss += loss.item()

                # Accuracy: MSE between forecasted and target latent means
                val_accuracy += (mean_target - forecasted_latent).pow(2).mean()

        n_val = len(val_loader)
        val_loss /= n_val
        val_accuracy = val_accuracy / n_val

        # Smoothed metrics
        val_loss_history.append(val_loss)
        val_loss_smooth = sum(val_loss_history) / len(val_loss_history)

        val_acc_history.append(val_accuracy)
        val_acc_smooth = sum(val_acc_history) / len(val_acc_history)

        # ----------------------------------------------------------------------
        # Model Selection — save on best smoothed validation accuracy
        # ----------------------------------------------------------------------
        if not best_val_loss or best_val_loss > val_loss:
            best_val_loss = val_loss

        if not best_val_loss_smooth or best_val_loss_smooth > val_loss_smooth:
            best_val_loss_smooth = val_loss_smooth

        if not best_val_acc or best_val_acc > val_accuracy:
            best_val_acc = val_accuracy

        if not best_val_acc_smooth or best_val_acc_smooth > val_acc_smooth:
            best_val_acc_smooth = val_acc_smooth
            torch.save(full_model.state_dict(), forecaster_model_path)

        # ----------------------------------------------------------------------
        # Logging — every 250 epochs
        # ----------------------------------------------------------------------
        if epoch % 250 == 0:
            print(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "pred_val_loss": val_loss,
                    "pred_val_accuracy": val_accuracy,
                    "pred_val_loss_smooth": val_loss_smooth,
                    "pred_val_acc_smooth": val_acc_smooth,
                    "pred_best_val_loss": best_val_loss,
                    "pred_best_val_loss_smooth": best_val_loss_smooth,
                    "pred_best_val_acc": best_val_acc,
                    "pred_best_val_acc_smooth": best_val_acc_smooth,
                }
            )

    # ==========================================================================
    # Done
    # ==========================================================================

    print(f"\nBest model saved to {forecaster_model_path}")
    print(f"  Best validation accuracy (smooth): {best_val_acc_smooth:.4f}")
    print("\nLatent forecaster training complete!")


if __name__ == "__main__":
    main()
