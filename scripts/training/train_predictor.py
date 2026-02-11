"""
Training script for the CO2 Emission Predictor.

This script trains a neural network to predict sectoral emission changes
(deltas) from VAE latent representations and context variables. The model
outputs both predictions and learned uncertainty estimates.

Prerequisites:
    - Trained VAE model (data/pytorch_models/vae_model.pth)

Usage:
    python -m scripts.training.train_predictor

Outputs:
    - data/pytorch_models/predictor_model.pth: Best model weights

Training details:
    - The VAE encoder weights are frozen during predictor training
    - The model predicts emission DELTAS: Δy_t = y_t - y_{t-1}
    - Uncertainty-aware MSE loss balances accuracy and calibration

Reference:
    Section 4.2.2 "Emissions Prediction Model" in the paper.
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
from scripts.elements.datasets import DatasetPrediction
from scripts.elements.models import (
    Decoder,
    Encoder,
    EmissionPredictor,
    FullPredictionModel,
    VAEModel,
    vae_loss,
    uncertainty_aware_mse_loss,
)
from scripts.utils import (
    check_nan_gradients,
    init_weights,
    load_config,
    load_dataset,
    save_dataset,
    count_parameters,
    freeze_module,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():
    """
    Main training loop for the emission predictor.

    Training procedure:
    1. Load pre-trained VAE and freeze its weights
    2. Build emission predictor network
    3. Create combined model (VAE encoder + predictor)
    4. Train using uncertainty-aware MSE loss on emission deltas
    5. Save best model based on smoothed validation loss
    """
    # ==========================================================================
    # Configuration
    # ==========================================================================

    vae_config = load_config("config/models/vae_config.yaml")
    pred_config = load_config("config/models/co2_predictor_config.yaml")

    # Training hyperparameters
    batch_size = 128
    num_epochs = 5000
    scaling_type = "normalization"

    # VAE architecture (must match trained model)
    vae_latent_dim = vae_config.vae_latent_dim

    # Predictor architecture (from config)
    pred_width = pred_config.pred_width_block
    pred_dim = pred_config.pred_dim_block
    pred_num_blocks = pred_config.pred_num_blocks
    pred_activation = pred_config.pred_activation
    pred_normalization = pred_config.pred_normalization
    pred_dropout = pred_config.pred_dropouts

    # Optimizer settings (from config)
    learning_rate = pred_config.pred_lr
    weight_decay = pred_config.pred_wd
    optimizer_name = pred_config.pred_optimizer

    # Loss mode (from config)
    loss_mode = pred_config.mode_loss

    # Paths
    dataset_path = Path("data/pytorch_datasets/predictor_dataset.pkl")
    vae_model_path = Path("data/pytorch_models/vae_model.pth")
    predictor_model_path = Path("data/pytorch_models/predictor_model.pth")
    variable_file = Path("config/data/variable_selection.txt")

    # Ensure output directories exist
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    predictor_model_path.parent.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Data Loading
    # ==========================================================================

    # Load variable selection
    with open(variable_file, "r") as f:
        nested_variables = [line.strip() for line in f if line.strip()]

    # Countries to include (EU27)
    eu27_countries = [
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "EL", "FI", "FR", "DE",
        "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
        "SI", "ES", "SE",
    ]

    # Load or create dataset
    if dataset_path.exists():
        print(f"Loading cached dataset from {dataset_path}")
        full_dataset = load_dataset(dataset_path)
    else:
        print("Creating new dataset...")
        full_dataset = DatasetPrediction(
            path_csvs="data/full_timeseries/",
            output_configs=output_configs,
            select_years=np.arange(2010, 2024),
            select_geo=eu27_countries,
            nested_variables=nested_variables,
            with_cuda=True,
            scaling_type=scaling_type,
        )
        save_dataset(full_dataset, dataset_path)
        print(f"Dataset saved to {dataset_path}")

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

    print(f"Dataset loaded: {len(full_dataset)} samples")
    print(f"  Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # ==========================================================================
    # Model Setup
    # ==========================================================================

    input_dim = len(full_dataset.input_variable_names)
    context_dim = len(full_dataset.context_variable_names)

    # Build VAE (architecture must match trained model)
    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=vae_latent_dim,
        num_blocks=vae_config.vae_num_blocks,
        dim_blocks=vae_config.vae_dim_blocks,
        activation=vae_config.vae_activation,
        normalization=vae_config.vae_normalization,
        dropout=vae_config.vae_dropouts,
        input_dropout=vae_config.vae_input_dropouts,
    ).cuda()

    decoder = Decoder(
        input_dim=input_dim,
        latent_dim=vae_latent_dim,
        num_blocks=vae_config.vae_num_blocks,
        dim_blocks=vae_config.vae_dim_blocks,
        activation=vae_config.vae_activation,
        normalization=vae_config.vae_normalization,
        dropout=vae_config.vae_dropouts,
    ).cuda()

    vae_model = VAEModel(encoder, decoder).cuda()

    # Load pre-trained VAE weights
    print(f"Loading pre-trained VAE from {vae_model_path}")
    vae_model.load_state_dict(torch.load(vae_model_path))

    # Build emission predictor
    # Input: [z_t, context_t, z_{t-1}, context_{t-1}]
    predictor_input_dim = 2 * (vae_latent_dim + context_dim)

    predictor = EmissionPredictor(
        input_dim=predictor_input_dim,
        output_configs=output_configs,
        num_blocks=pred_num_blocks,
        dim_block=pred_dim,
        width_block=pred_width,
        activation=pred_activation,
        normalization=pred_normalization,
        dropout=pred_dropout,
        uncertainty=True,
    ).cuda()
    predictor.apply(init_weights)

    # Combine into full model
    full_model = FullPredictionModel(vae=vae_model, predictor=predictor)

    print(f"Emission predictor created with {count_parameters(predictor):,} parameters")
    print(f"  Input dimension: {predictor_input_dim}")
    print(f"  Output sectors: {predictor.output_size}")

    # ==========================================================================
    # Optimizer Setup
    # ==========================================================================

    # Freeze VAE weights - only train the predictor
    freeze_module(full_model.encoder)
    freeze_module(full_model.decoder)

    print("VAE weights frozen for predictor training")

    optimizer_cls = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "radam": torch.optim.RAdam,
    }.get(optimizer_name.lower())

    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Different learning rates: lower for VAE (if unfrozen), higher for predictor
    optimizer = optimizer_cls(
        [
            {"params": full_model.encoder.parameters(), "lr": learning_rate * 1e-3},
            {"params": full_model.decoder.parameters(), "lr": learning_rate * 1e-3},
            {"params": full_model.predictor.parameters(), "lr": learning_rate},
        ],
        weight_decay=weight_decay,
        eps=1e-6,
    )

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Optimizer: {optimizer_name}, LR: {learning_rate}")
    print(f"  Loss mode: {loss_mode}")

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
                input_current,
                context_current,
                emissions,
                input_prev,
                context_prev,
                emissions_prev,
            ) = batch

            optimizer.zero_grad()

            # Forward pass through combined model
            # NOTE: emissions_prev is NOT passed to the model - it's only used
            # to compute the ground truth delta in the training loop
            (
                emission_delta_pred,
                emission_uncertainty,
                recon_current,
                recon_prev,
                mean_current,
                mean_prev,
                log_var_current,
                log_var_prev,
            ) = full_model(input_current, input_prev, context_current, context_prev)

            # Compute ground truth emission delta
            emission_delta_true = emissions - emissions_prev

            # Uncertainty-aware loss
            loss = uncertainty_aware_mse_loss(
                emission_delta_true,
                emission_delta_pred,
                emission_uncertainty,
                mode=loss_mode,
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
        val_uncertainty = 0.0
        val_accuracy = 0.0
        val_recon = 0.0

        with torch.inference_mode():
            for batch in val_loader:
                (
                    input_current,
                    context_current,
                    emissions,
                    input_prev,
                    context_prev,
                    emissions_prev,
                ) = batch

                (
                    emission_delta_pred,
                    emission_uncertainty,
                    recon_current,
                    recon_prev,
                    mean_current,
                    mean_prev,
                    log_var_current,
                    log_var_prev,
                ) = full_model(input_current, input_prev, context_current, context_prev)

                emission_delta_true = emissions - emissions_prev

                # Loss
                loss = uncertainty_aware_mse_loss(
                    emission_delta_true,
                    emission_delta_pred,
                    emission_uncertainty,
                    mode=loss_mode,
                )
                val_loss += loss.item()

                # Metrics
                val_uncertainty += emission_uncertainty.mean().item()

                # Accuracy: MSE of predicted vs actual emissions (not deltas)
                emissions_pred = emission_delta_pred + emissions_prev
                val_accuracy += (emissions_pred - emissions).pow(2).mean().item()

                # Reconstruction loss (monitor VAE health)
                recon_loss, _ = vae_loss(
                    input_current, recon_current, mean_current, log_var_current
                )
                val_recon += recon_loss.item()

        n_val = len(val_loader)
        val_loss /= n_val
        val_uncertainty /= n_val
        val_accuracy /= n_val
        val_recon /= n_val

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
            best_weights = full_model.state_dict()

        if best_val_acc is None or val_accuracy < best_val_acc:
            best_val_acc = val_accuracy

        if best_val_acc_smooth is None or val_acc_smooth < best_val_acc_smooth:
            best_val_acc_smooth = val_acc_smooth

        # ----------------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------------
        if epoch % 250 == 0 or epoch == num_epochs - 1:
            print(f"\n--- Epoch {epoch} ---")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val loss: {val_loss:.4f} (smooth: {val_loss_smooth:.4f})")
            print(f"  Val uncertainty: {val_uncertainty:.4f}")
            print(f"  Val MSE (emissions): {val_accuracy:.4f} (smooth: {val_acc_smooth:.4f})")
            print(f"  Val VAE recon: {val_recon:.4f}")
            print(f"  Best smooth loss: {best_val_loss_smooth:.4f}")
            print(f"  Best smooth acc: {best_val_acc_smooth:.4f}")

    # ==========================================================================
    # Save Best Model
    # ==========================================================================

    if best_weights is not None:
        torch.save(best_weights, predictor_model_path)
        print(f"\nBest model saved to {predictor_model_path}")
        print(f"  Best validation loss (smooth): {best_val_loss_smooth:.4f}")
        print(f"  Best validation MSE (smooth): {best_val_acc_smooth:.4f}")
    else:
        print("\nWarning: No best weights saved")

    print("\nEmission predictor training complete!")


if __name__ == "__main__":
    main()