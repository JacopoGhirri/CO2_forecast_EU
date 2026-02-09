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

import random
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
    freeze_module,
    init_weights,
    load_config,
    load_dataset,
    save_dataset,
)

# Reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
    num_epochs = 5000
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
    dataset_path = Path("data/pytorch_datasets/unified_dataset.pkl")
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

    # Load or create base dataset
    if dataset_path.exists():
        print(f"Loading cached dataset from {dataset_path}")
        base_dataset = load_dataset(dataset_path)
    else:
        print("Creating new dataset...")
        base_dataset = DatasetUnified(
            path_csvs="data/full_timeseries/",
            output_configs=output_configs,
            select_years=np.arange(2010, 2024),
            select_geo=eu27_countries,
            nested_variables=nested_variables,
            with_cuda=True,
            scaling_type=scaling_type,
        )
        save_dataset(base_dataset, dataset_path)
        print(f"Dataset saved to {dataset_path}")

    # Wrap in forecasting dataset (provides 3 time steps: t, t-1, t-2)
    full_dataset = DatasetForecasting(base_dataset)

    # Train/validation split
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    print(f"Dataset loaded: {full_dataset.base_length} unique samples (5x inflated)")
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

    # Freeze VAE weights - only train the forecaster
    freeze_module(full_model.encoder)
    freeze_module(full_model.vae)

    print("VAE weights frozen for forecaster training")

    optimizer_cls = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "radam": torch.optim.RAdam,
    }.get(optimizer_name.lower())

    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    optimizer = optimizer_cls(
        full_model.forecaster.parameters(),
        lr=learning_rate,
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
    best_weights = None

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
            # The model takes (x_{t-1}, x_{t-2}, c_t, c_{t-1}) and internally
            # encodes x to z before forecasting
            forecasted_latent = full_model(
                input_prev, input_past, context_current, context_prev
            )

            # Get target latent distribution parameters
            mean_target, log_var_target = full_model.encoder(input_current)

            # Loss: uncertainty-aware MAE
            # The target is the mean of the latent distribution
            # Uncertainty is provided by the latent variance (from encoder)
            loss = uncertainty_aware_mae_loss(
                mean_target,
                forecasted_latent,
                log_var_target,
                mode="regular",
            )

            # Backward pass
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
                val_accuracy += (mean_target - forecasted_latent).pow(2).mean().item()

        n_val = len(val_loader)
        val_loss /= n_val
        val_accuracy /= n_val

        # Smoothed metrics
        val_loss_history.append(val_loss)
        val_loss_smooth = sum(val_loss_history) / len(val_loss_history)

        val_acc_history.append(val_accuracy)
        val_acc_smooth = sum(val_acc_history) / len(val_acc_history)

        # ----------------------------------------------------------------------
        # Model Selection
        # ----------------------------------------------------------------------
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss

        if best_val_loss_smooth is None or val_loss_smooth < best_val_loss_smooth:
            best_val_loss_smooth = val_loss_smooth

        if best_val_acc is None or val_accuracy < best_val_acc:
            best_val_acc = val_accuracy

        # Select best model based on smoothed accuracy
        if best_val_acc_smooth is None or val_acc_smooth < best_val_acc_smooth:
            best_val_acc_smooth = val_acc_smooth
            best_weights = full_model.state_dict()

        # ----------------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------------
        if epoch % 250 == 0 or epoch == num_epochs - 1:
            print(f"\n--- Epoch {epoch} ---")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val loss: {val_loss:.4f} (smooth: {val_loss_smooth:.4f})")
            print(
                f"  Val MSE (latent): {val_accuracy:.4f} (smooth: {val_acc_smooth:.4f})"
            )
            print(
                f"  Best: loss={best_val_loss:.4f}, smooth_loss={best_val_loss_smooth:.4f}"
            )
            print(
                f"  Best: acc={best_val_acc:.4f}, smooth_acc={best_val_acc_smooth:.4f}"
            )

    # ==========================================================================
    # Save Best Model
    # ==========================================================================

    if best_weights is not None:
        torch.save(best_weights, forecaster_model_path)
        print(f"\nBest model saved to {forecaster_model_path}")
        print(f"  Best validation accuracy (smooth): {best_val_acc_smooth:.4f}")
    else:
        print("\nWarning: No best weights saved")

    print("\nLatent forecaster training complete!")


if __name__ == "__main__":
    main()
