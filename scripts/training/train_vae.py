"""
Training script for the Variational Autoencoder (VAE).

This script trains a VAE to learn compressed latent representations of
high-dimensional socioeconomic indicators. The VAE captures both the
central structure of the data and the variability around it.

The trained VAE is used downstream by:
1. Emission Predictor: Uses latent representations to predict emissions
2. Latent Forecaster: Projects future latent states for 2030 projections

Usage:
    python -m scripts.training.train_vae

Outputs:
    - data/pytorch_models/vae_model.pth: Best model weights
    - data/pytorch_datasets/unified_dataset.pkl: Processed dataset (cached)

Reference:
    Section 4.2.1 "Variational Autoencoder" in the paper.
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
from scripts.elements.datasets import DatasetUnified
from scripts.elements.models import Decoder, Encoder, VAEModel, vae_loss
from scripts.utils import (
    check_nan_gradients,
    count_parameters,
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
    Main training loop for the VAE.

    Training procedure:
    1. Load or create dataset with preprocessing and scaling
    2. Split into train/validation sets (85/15)
    3. Build encoder and decoder networks
    4. Train using weighted ELBO loss (reconstruction + KL divergence)
    5. Save best model based on smoothed validation reconstruction loss
    """
    # ==========================================================================
    # Configuration
    # ==========================================================================

    config = load_config("config/models/vae_config.yaml")

    # Training hyperparameters
    batch_size = 128
    num_epochs = 5000
    scaling_type = "normalization"

    # Model architecture (from config)
    latent_dim = config.vae_latent_dim
    num_blocks = config.vae_num_blocks
    dim_blocks = config.vae_dim_blocks
    activation = config.vae_activation
    normalization = config.vae_normalization
    dropout = config.vae_dropouts
    input_dropout = config.vae_input_dropouts

    # Optimizer settings (from config)
    learning_rate = config.vae_lr
    weight_decay = config.vae_weight_decay
    optimizer_name = config.vae_optimizer

    # Loss weights (from config)
    weight_recon = config.vae_wr  # Reconstruction loss weight
    weight_kl = config.vae_wd  # KL divergence weight

    # Paths
    dataset_path = Path("data/pytorch_datasets/unified_dataset.pkl")
    model_path = Path("data/pytorch_models/vae_model.pth")
    variable_file = Path("config/data/variable_selection.txt")

    # Ensure output directories exist
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Load or create dataset
    if dataset_path.exists():
        print(f"Loading cached dataset from {dataset_path}")
        full_dataset = load_dataset(dataset_path)
    else:
        print("Creating new dataset...")
        full_dataset = DatasetUnified(
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
    print(f"  Input features: {len(full_dataset.input_variable_names)}")
    print(f"  Context features: {len(full_dataset.context_variable_names)}")

    # ==========================================================================
    # Model Setup
    # ==========================================================================

    input_dim = len(full_dataset.input_variable_names)

    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_blocks=num_blocks,
        dim_blocks=dim_blocks,
        activation=activation,
        normalization=normalization,
        dropout=dropout,
        input_dropout=input_dropout,
    ).cuda()

    decoder = Decoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_blocks=num_blocks,
        dim_blocks=dim_blocks,
        activation=activation,
        normalization=normalization,
        dropout=dropout,
    ).cuda()

    model = VAEModel(encoder, decoder).cuda()
    model.apply(init_weights)

    print(f"VAE model created with {count_parameters(model):,} parameters")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Architecture: {num_blocks} blocks x {dim_blocks} layers")

    # ==========================================================================
    # Optimizer Setup
    # ==========================================================================

    optimizer_cls = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "radam": torch.optim.RAdam,
    }.get(optimizer_name.lower())

    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    optimizer = optimizer_cls(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-6,
    )

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Optimizer: {optimizer_name}, LR: {learning_rate}")
    print(f"  Loss weights: reconstruction={weight_recon}, KL={weight_kl}")

    best_val_loss = float("inf")
    best_recon_loss = float("inf")
    best_weights = None

    # Smoothed validation loss for model selection
    recon_loss_history = deque(maxlen=10)

    for epoch in range(num_epochs):
        # ----------------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------------
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_mean_latent = 0.0

        for batch in train_loader:
            input_current, _, _, _, _ = batch

            optimizer.zero_grad()

            # Forward pass
            reconstruction, mean, log_var = model(input_current)

            # Compute loss
            recon_loss, kl_loss = vae_loss(input_current, reconstruction, mean, log_var)
            loss = weight_recon * recon_loss + weight_kl * kl_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            check_nan_gradients(model)
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            train_mean_latent += mean.mean().item()

        # Average metrics
        n_batches = len(train_loader)
        train_loss /= n_batches
        train_recon /= n_batches
        train_kl /= n_batches
        train_mean_latent /= n_batches

        # ----------------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0

        with torch.inference_mode():
            for batch in val_loader:
                input_current, _, _, _, _ = batch

                reconstruction, mean, log_var = model(input_current)
                recon_loss, kl_loss = vae_loss(
                    input_current, reconstruction, mean, log_var
                )
                loss = weight_recon * recon_loss + weight_kl * kl_loss

                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()

        n_val_batches = len(val_loader)
        val_loss /= n_val_batches
        val_recon /= n_val_batches
        val_kl /= n_val_batches

        # Smoothed reconstruction loss for model selection
        recon_loss_history.append(val_recon)
        val_recon_smooth = np.mean(recon_loss_history)

        # ----------------------------------------------------------------------
        # Model Selection
        # ----------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_recon_smooth < best_recon_loss:
            best_recon_loss = val_recon_smooth
            best_weights = model.state_dict()

        # ----------------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------------
        if epoch % 250 == 0 or epoch == num_epochs - 1:
            print(f"\n--- Epoch {epoch} ---")
            print(
                f"  Train: loss={train_loss:.4f}, recon={train_recon:.4f}, "
                f"KL={train_kl:.4f}, mean_latent={train_mean_latent:.4f}"
            )
            print(
                f"  Val:   loss={val_loss:.4f}, recon={val_recon:.4f}, KL={val_kl:.4f}"
            )
            print(f"  Val (smoothed): recon={val_recon_smooth:.4f}")
            print(f"  Best: loss={best_val_loss:.4f}, recon={best_recon_loss:.4f}")

    # ==========================================================================
    # Save Best Model
    # ==========================================================================

    if best_weights is not None:
        torch.save(best_weights, model_path)
        print(f"\nBest model saved to {model_path}")
        print(f"  Best validation reconstruction loss: {best_recon_loss:.4f}")
    else:
        print("\nWarning: No best weights saved (training may have failed)")

    print("\nVAE training complete!")


if __name__ == "__main__":
    main()
