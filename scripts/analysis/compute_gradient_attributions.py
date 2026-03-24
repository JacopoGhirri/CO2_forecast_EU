"""
Gradient-based explainability for the emission prediction pipeline.

This script computes average gradient activations (sensitivities) across the
full training dataset for two components of the pipeline:

1. **Predictor attributions**: How each predictor input (latent variables and
   context variables) influences each output (sector emission deltas and
   learned uncertainties). Gradients are computed as ∂output_j / ∂input_i
   for every (output, input) pair and averaged across all observations.

2. **Encoder attributions**: How each raw input feature influences each
   latent mean dimension. Gradients are computed as ∂mean_j / ∂x_i and
   averaged across all observations.

Because the predictor receives two timesteps of the same variable
([z_t, context_t, z_{t-1}, context_{t-1}]), the gradients for the same
logical variable at t and t-1 are averaged together, yielding one
attribution per unique variable name.

Both signed and absolute gradients are stored:
    - Signed gradients preserve directionality but may cancel across samples.
    - Absolute gradients measure sensitivity regardless of direction.

Prerequisites:
    - Trained VAE model       (data/pytorch_models/vae_model.pth)
    - Trained predictor model (data/pytorch_models/predictor_model.pth)
    - Cached unified dataset  (data/pytorch_datasets/unified_dataset.pkl)

Usage:
    python -m scripts.analysis.compute_gradient_attributions

Outputs:
    - data/explainability/predictor_gradient_attributions.csv
        Rows: output neurons (emission_delta_<sector>, uncertainty_<sector>)
        Columns: unique input variable names (latent_0..N, context vars),
                 each appearing twice with suffixes _signed and _absolute.

    - data/explainability/encoder_gradient_attributions.csv
        Rows: latent mean dimensions (latent_mean_0..N)
        Columns: input feature names, each appearing twice with suffixes
                 _signed and _absolute.

Reference:
    Gradient-based sensitivity analysis complements the prediction pipeline
    described in Sections 4.2.1–4.2.2 of the paper by quantifying which
    inputs drive each output.
"""

from __future__ import annotations

import multiprocessing as mp

# Set multiprocessing start method before other imports
mp.set_start_method("spawn", force=True)

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config.data.output_configs import output_configs
from scripts.elements.models import (
    Decoder,
    EmissionPredictor,
    Encoder,
    FullPredictionModel,
    VAEModel,
    reparameterize,
)
from scripts.utils import load_config, load_dataset

# =============================================================================
# Configuration
# =============================================================================

# Paths — models and data
VAE_CONFIG_PATH = Path("config/models/vae_config.yaml")
PREDICTOR_CONFIG_PATH = Path("config/models/co2_predictor_config.yaml")

VAE_MODEL_PATH = Path("data/pytorch_models/vae_model.pth")
PREDICTOR_MODEL_PATH = Path("data/pytorch_models/predictor_model.pth")
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")

# Paths — outputs
OUTPUT_DIR = Path("data/explainability")
PREDICTOR_OUTPUT_PATH = OUTPUT_DIR / "predictor_gradient_attributions.csv"
ENCODER_OUTPUT_PATH = OUTPUT_DIR / "encoder_gradient_attributions.csv"

# Emission sectors (must match output_configs)
EMISSION_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model Loading
# =============================================================================


def load_models(
    dataset,
    vae_config_path: Path,
    predictor_config_path: Path,
    vae_model_path: Path,
    predictor_model_path: Path,
) -> tuple[VAEModel, EmissionPredictor, int]:
    """
    Loads trained VAE and predictor models for gradient computation.

    Both models are placed in eval mode with dropout fully disabled to
    ensure deterministic, reproducible gradient computations.

    Args:
        dataset: Dataset instance (provides input/context dimensions).
        vae_config_path: Path to VAE config YAML.
        predictor_config_path: Path to predictor config YAML.
        vae_model_path: Path to trained VAE weights.
        predictor_model_path: Path to trained predictor weights.

    Returns:
        Tuple of (vae_model, predictor, latent_dim).
    """
    vae_config = load_config(vae_config_path)
    predictor_config = load_config(predictor_config_path)

    input_dim = len(dataset.input_variable_names)
    context_dim = len(dataset.context_variable_names)
    latent_dim = vae_config.vae_latent_dim

    # ----- Build and load VAE ------------------------------------------------
    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_blocks=vae_config.vae_num_blocks,
        dim_blocks=vae_config.vae_dim_blocks,
        activation=vae_config.vae_activation,
        normalization=vae_config.vae_normalization,
        dropout=vae_config.vae_dropouts,
        input_dropout=vae_config.vae_input_dropouts,
    )
    decoder = Decoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_blocks=vae_config.vae_num_blocks,
        dim_blocks=vae_config.vae_dim_blocks,
        activation=vae_config.vae_activation,
        normalization=vae_config.vae_normalization,
        dropout=vae_config.vae_dropouts,
    )
    vae_model = VAEModel(encoder, decoder).to(DEVICE)
    vae_model.load_state_dict(torch.load(vae_model_path, map_location="cpu"))
    vae_model.eval()

    # ----- Build and load predictor ------------------------------------------
    predictor_input_dim = 2 * (latent_dim + context_dim)
    predictor = EmissionPredictor(
        input_dim=predictor_input_dim,
        output_configs=output_configs,
        num_blocks=predictor_config.pred_num_blocks,
        dim_block=predictor_config.pred_dim_block,
        width_block=predictor_config.pred_width_block,
        activation=predictor_config.pred_activation,
        normalization=predictor_config.pred_normalization,
        dropout=predictor_config.pred_dropouts,
        uncertainty=True,
    ).to(DEVICE)

    # The predictor checkpoint stores the full prediction model state dict
    # (VAE + predictor), so we load it via FullPredictionModel to place
    # weights into both sub-models correctly.
    full_model = FullPredictionModel(vae=vae_model, predictor=predictor)
    full_model.load_state_dict(torch.load(predictor_model_path, map_location="cpu"))
    full_model.eval()

    return vae_model, predictor, latent_dim


# =============================================================================
# Predictor Gradient Attributions
# =============================================================================


def compute_predictor_attributions(
    dataset,
    vae_model: VAEModel,
    predictor: EmissionPredictor,
    latent_dim: int,
) -> pd.DataFrame:
    """
    Computes average gradient of each predictor output w.r.t. its inputs.

    For every observation the predictor receives a concatenated vector
    ``[z_t, context_t, z_{t-1}, context_{t-1}]``.  We compute the full
    Jacobian of the predictor outputs (emission deltas + uncertainties)
    w.r.t. this input vector, then fold the two timesteps together by
    averaging gradients that correspond to the same logical variable
    (e.g. ``latent_0`` at t and at t-1).

    Both the signed mean and the mean of absolute values are accumulated.

    Args:
        dataset: Dataset instance exposing ``input_df``, ``context_df``,
            ``keys``, ``index_map``, ``input_variable_names``, and
            ``context_variable_names``.
        vae_model: Trained VAE (encoder used to produce latents).
        predictor: Trained EmissionPredictor.
        latent_dim: Dimensionality of the latent space.

    Returns:
        DataFrame with one row per output neuron and two columns per
        unique input variable (``<name>_signed``, ``<name>_absolute``).
    """
    encoder = vae_model.encoder
    n_samples = len(dataset)
    context_dim = len(dataset.context_variable_names)

    # Logical variable names after timestep folding
    latent_names = [f"latent_{i}" for i in range(latent_dim)]
    context_names = list(dataset.context_variable_names)
    unique_input_names = latent_names + context_names

    # Output neuron names: emission deltas then uncertainties
    output_names = [f"emission_delta_{s}" for s in EMISSION_SECTORS] + [
        f"uncertainty_{s}" for s in EMISSION_SECTORS
    ]
    n_outputs = len(output_names)
    n_unique_inputs = len(unique_input_names)

    # Accumulators — shape (n_outputs, n_unique_inputs)
    grad_sum_signed = torch.zeros(n_outputs, n_unique_inputs, device=DEVICE)
    grad_sum_abs = torch.zeros(n_outputs, n_unique_inputs, device=DEVICE)

    print(f"Computing predictor attributions over {n_samples} samples...")

    for idx in range(n_samples):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Predictor: sample {idx + 1}/{n_samples}")

        # --- Retrieve current and previous year data -------------------------
        geo = dataset.keys.iloc[idx, 0]
        year = dataset.keys.iloc[idx, 1]

        input_current = dataset.input_df[idx].unsqueeze(0).to(DEVICE)
        context_current = dataset.context_df[idx].unsqueeze(0).to(DEVICE)

        prev_idx = dataset.index_map.get((geo, year - 1))
        if prev_idx is not None:
            input_prev = dataset.input_df[prev_idx].unsqueeze(0).to(DEVICE)
            context_prev = dataset.context_df[prev_idx].unsqueeze(0).to(DEVICE)
        else:
            # Fallback: use current year if previous is unavailable
            input_prev = input_current.clone()
            context_prev = context_current.clone()

        # --- Encode to latent space (detached — no grad through encoder) -----
        with torch.no_grad():
            mean_t, log_var_t = encoder(input_current)
            mean_t1, log_var_t1 = encoder(input_prev)
            z_t = reparameterize(mean_t, torch.exp(0.5 * log_var_t))
            z_t1 = reparameterize(mean_t1, torch.exp(0.5 * log_var_t1))

        # --- Build predictor input with gradient tracking --------------------
        # Layout: [z_t (L), context_t (C), z_{t-1} (L), context_{t-1} (C)]
        pred_input = (
            torch.cat((z_t, context_current, z_t1, context_prev), dim=1)
            .detach()
            .requires_grad_(True)
        )

        # --- Forward through predictor ---------------------------------------
        emission_delta, uncertainty = predictor(pred_input)

        # Stack into a single output vector for Jacobian computation:
        # [delta_sector_0 .. delta_sector_5, unc_sector_0 .. unc_sector_5]
        outputs = torch.cat((emission_delta, uncertainty), dim=1)  # (1, 12)

        # --- Compute Jacobian one output at a time ---------------------------
        for j in range(n_outputs):
            pred_input.grad = None
            outputs[0, j].backward(retain_graph=(j < n_outputs - 1))

            raw_grad = pred_input.grad[0]  # (2L + 2C,)

            # Split into the four logical groups
            offset = 0
            grad_z_t = raw_grad[offset : offset + latent_dim]
            offset += latent_dim
            grad_c_t = raw_grad[offset : offset + context_dim]
            offset += context_dim
            grad_z_t1 = raw_grad[offset : offset + latent_dim]
            offset += latent_dim
            grad_c_t1 = raw_grad[offset : offset + context_dim]

            # Average the two timesteps for each logical variable
            grad_latent = (grad_z_t + grad_z_t1) / 2.0
            grad_context = (grad_c_t + grad_c_t1) / 2.0
            grad_folded = torch.cat((grad_latent, grad_context))  # (L + C,)

            grad_sum_signed[j] += grad_folded
            grad_sum_abs[j] += grad_folded.abs()

    # --- Average over samples ------------------------------------------------
    grad_mean_signed = (grad_sum_signed / n_samples).cpu().numpy()
    grad_mean_abs = (grad_sum_abs / n_samples).cpu().numpy()

    # --- Assemble DataFrame --------------------------------------------------
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(unique_input_names):
        columns[f"{name}_signed"] = grad_mean_signed[:, i]
        columns[f"{name}_absolute"] = grad_mean_abs[:, i]

    df = pd.DataFrame(columns, index=output_names)
    df.index.name = "output"

    return df


# =============================================================================
# Encoder Gradient Attributions
# =============================================================================


def compute_encoder_attributions(
    dataset,
    vae_model: VAEModel,
    latent_dim: int,
) -> pd.DataFrame:
    """
    Computes average gradient of each latent mean w.r.t. raw input features.

    For every observation the encoder maps ``x → (mean, log_var)``.  We
    compute ∂mean_j / ∂x_i for every (latent dimension, input feature)
    pair and accumulate both the signed mean and the mean of absolute
    values across the full dataset.

    Args:
        dataset: Dataset instance exposing ``input_df`` and
            ``input_variable_names``.
        vae_model: Trained VAE (encoder is used).
        latent_dim: Dimensionality of the latent space.

    Returns:
        DataFrame with one row per latent mean dimension and two columns
        per input feature (``<name>_signed``, ``<name>_absolute``).
    """
    encoder = vae_model.encoder
    n_samples = dataset.input_df.shape[0]
    input_dim = dataset.input_df.shape[1]

    feature_names = list(dataset.input_variable_names)
    latent_names = [f"latent_mean_{j}" for j in range(latent_dim)]

    # Accumulators — shape (latent_dim, input_dim)
    grad_sum_signed = torch.zeros(latent_dim, input_dim, device=DEVICE)
    grad_sum_abs = torch.zeros(latent_dim, input_dim, device=DEVICE)

    print(f"Computing encoder attributions over {n_samples} samples...")

    for idx in range(n_samples):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Encoder: sample {idx + 1}/{n_samples}")

        # Prepare input with gradient tracking
        x = dataset.input_df[idx].unsqueeze(0).to(DEVICE).detach().requires_grad_(True)

        # Forward through encoder — only mean is needed
        mean, _ = encoder(x)  # mean shape: (1, latent_dim)

        # Compute Jacobian one latent dimension at a time
        for j in range(latent_dim):
            x.grad = None
            mean[0, j].backward(retain_graph=(j < latent_dim - 1))

            grad = x.grad[0]  # (input_dim,)
            grad_sum_signed[j] += grad
            grad_sum_abs[j] += grad.abs()

    # --- Average over samples ------------------------------------------------
    grad_mean_signed = (grad_sum_signed / n_samples).cpu().numpy()
    grad_mean_abs = (grad_sum_abs / n_samples).cpu().numpy()

    # --- Assemble DataFrame --------------------------------------------------
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(feature_names):
        columns[f"{name}_signed"] = grad_mean_signed[:, i]
        columns[f"{name}_absolute"] = grad_mean_abs[:, i]

    df = pd.DataFrame(columns, index=latent_names)
    df.index.name = "latent_dimension"

    return df


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """
    Main function to compute and save gradient attributions.

    Workflow:
    1. Load the cached unified dataset (full set, no train/val split)
    2. Load trained VAE and predictor models
    3. Compute predictor gradient attributions (predictor inputs → outputs)
    4. Compute encoder gradient attributions (raw features → latent means)
    5. Save both results as CSV files
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading dataset from {DATASET_PATH}...")

    dataset = load_dataset(DATASET_PATH)

    # Move tensors to compute device
    dataset.input_df = dataset.input_df.to(DEVICE)
    dataset.context_df = dataset.context_df.to(DEVICE)
    dataset.emi_df = dataset.emi_df.to(DEVICE)

    print(f"  Samples: {dataset.input_df.shape[0]}")
    print(f"  Input features: {len(dataset.input_variable_names)}")
    print(f"  Context features: {len(dataset.context_variable_names)}")

    print("Loading models...")
    vae_model, predictor, latent_dim = load_models(
        dataset=dataset,
        vae_config_path=VAE_CONFIG_PATH,
        predictor_config_path=PREDICTOR_CONFIG_PATH,
        vae_model_path=VAE_MODEL_PATH,
        predictor_model_path=PREDICTOR_MODEL_PATH,
    )
    print(f"  Latent dimension: {latent_dim}")

    # ---- Predictor attributions ---------------------------------------------
    print("\n" + "=" * 70)
    print("PREDICTOR GRADIENT ATTRIBUTIONS")
    print("=" * 70)

    predictor_df = compute_predictor_attributions(
        dataset=dataset,
        vae_model=vae_model,
        predictor=predictor,
        latent_dim=latent_dim,
    )

    predictor_df.to_csv(PREDICTOR_OUTPUT_PATH)
    print(f"\nPredictor attributions saved to {PREDICTOR_OUTPUT_PATH}")
    print(f"  Shape: {predictor_df.shape}  (outputs × 2·inputs)")

    # ---- Encoder attributions -----------------------------------------------
    print("\n" + "=" * 70)
    print("ENCODER GRADIENT ATTRIBUTIONS")
    print("=" * 70)

    encoder_df = compute_encoder_attributions(
        dataset=dataset,
        vae_model=vae_model,
        latent_dim=latent_dim,
    )

    encoder_df.to_csv(ENCODER_OUTPUT_PATH)
    print(f"\nEncoder attributions saved to {ENCODER_OUTPUT_PATH}")
    print(f"  Shape: {encoder_df.shape}  (latent dims × 2·features)")

    print("\nGradient attribution computation complete!")


if __name__ == "__main__":
    main()
