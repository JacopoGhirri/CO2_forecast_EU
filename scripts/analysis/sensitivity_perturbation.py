"""
Perturbation-based sensitivity analysis.

This script performs sensitivity analysis using uniform perturbations to
assess the importance of input variables on emission predictions. It computes
Spearman correlations between perturbed inputs and model outputs.

Workflow:
1. Group variables by base name (collapsing monthly/quarterly variants)
2. Build per-group bounds around baseline values (±30% by default)
3. Sample N perturbations uniformly within bounds
4. Run full model for each perturbation (aggregated across countries)
5. Compute Spearman correlation as importance measure
6. Save results to CSV

Usage:
    python -m scripts.analysis.sensitivity_perturbation

Outputs:
    - data/sensitivity/perturbation_results.csv

Reference:
    Section 4.4 "Sensitivity Analysis" in the paper.
"""

import csv
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from config.data.output_configs import output_configs
from scripts.elements.datasets import DatasetProjections2030
from scripts.elements.models import (
    Decoder,
    EmissionPredictor,
    Encoder,
    FullLatentForecastingModel,
    FullPredictionModel,
    LatentForecaster,
    VAEModel,
)
from scripts.utils import load_config, load_dataset

# =============================================================================
# Configuration
# =============================================================================

# Paths
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
VAE_CONFIG_PATH = Path("config/models/vae_config.yaml")
VAE_MODEL_PATH = Path("data/pytorch_models/vae_model.pth")
PREDICTOR_CONFIG_PATH = Path("config/models/co2_predictor_config.yaml")
PREDICTOR_MODEL_PATH = Path("data/pytorch_models/predictor_model.pth")
FORECASTER_CONFIG_PATH = Path("config/models/latent_forecaster_config.yaml")
FORECASTER_MODEL_PATH = Path("data/pytorch_models/forecaster_model.pth")
OUTPUT_PATH = Path("data/sensitivity/perturbation_results.csv")

# Analysis settings
N_SAMPLES = 2048 * 8  # Number of perturbation samples
PERTURBATION_FRACTION = 0.3  # ±30% around baseline
EU_MODE = True  # Aggregate across all EU27 countries
TARGET_COUNTRY = "DE"  # Used if EU_MODE is False
TARGET_YEAR = 2024
BASELINE_YEAR = 2023

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EU27_COUNTRIES = [
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

SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

# Reproducibility
RANDOM_SEED = 0
TORCH_SEED = 0


# =============================================================================
# Variable Grouping
# =============================================================================


def strip_temporal_suffix(name: str) -> str:
    """
    Remove trailing temporal suffixes (_1, _2, ..., _12) from variable names.

    This collapses monthly or quarterly variants into a single grouped variable.

    Args:
        name: Variable name potentially with temporal suffix.

    Returns:
        Base variable name without suffix.
    """
    return re.sub(r"_\d+$", "", name)


def build_grouped_variable_specs(
    input_names: list[str],
    context_names: list[str],
) -> list[dict]:
    """
    Group input and context variables by their base names.

    Variables like 'temperature_1', 'temperature_2', etc. are collapsed
    into a single group 'temperature' with all their indices.

    Args:
        input_names: List of input variable names.
        context_names: List of context variable names.

    Returns:
        List of variable spec dicts with keys:
            - name: Display name for the variable group
            - type: 'input' or 'context'
            - indices: List of indices in the original array
            - base: Base variable name
    """
    var_specs = []
    seen = {}

    # Group input variables
    for idx, var_name in enumerate(input_names):
        base = strip_temporal_suffix(var_name)
        if base not in seen:
            seen[base] = {
                "name": base,
                "type": "input",
                "indices": [],
                "base": base,
            }
            var_specs.append(seen[base])
        seen[base]["indices"].append(idx)

    # Group context variables (prefix to avoid name collisions)
    for idx, var_name in enumerate(context_names):
        base = strip_temporal_suffix(var_name)
        if base not in seen:
            key_name = f"CONTEXT::{base}"
            seen[base] = {
                "name": key_name,
                "type": "context",
                "indices": [],
                "base": base,
            }
            var_specs.append(seen[base])
        seen[base]["indices"].append(idx)

    return var_specs


# =============================================================================
# Model Loading
# =============================================================================


def load_models(dataset):
    """
    Load all trained models for sensitivity analysis.

    Args:
        dataset: Dataset instance (needed for dimensions).

    Returns:
        Tuple of (full_pred_model, forecaster, latent_dim).
    """
    vae_config = load_config(VAE_CONFIG_PATH)
    predictor_config = load_config(PREDICTOR_CONFIG_PATH)
    forecaster_config = load_config(FORECASTER_CONFIG_PATH)

    input_dim = len(dataset.input_variable_names)
    context_dim = len(dataset.context_variable_names)
    latent_dim = vae_config.vae_latent_dim

    # Build VAE
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
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    vae_model.eval()

    # Build predictor
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

    full_pred_model = FullPredictionModel(vae=vae_model, predictor=predictor)
    full_pred_model.load_state_dict(
        torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE)
    )
    full_pred_model.eval()

    # Build forecaster
    forecaster = LatentForecaster(
        input_dim=predictor_input_dim,
        latent_dim=latent_dim,
        num_blocks=forecaster_config.forecast_num_blocks,
        dim_block=forecaster_config.forecast_dim_block,
        width_block=forecaster_config.forecast_width_block,
        activation=forecaster_config.forecast_activation,
        normalization=forecaster_config.forecast_normalization,
        dropout=forecaster_config.forecast_dropouts,
    ).to(DEVICE)

    full_forecast_model = FullLatentForecastingModel(
        vae=vae_model, forecaster=forecaster
    )
    full_forecast_model.load_state_dict(
        torch.load(FORECASTER_MODEL_PATH, map_location=DEVICE)
    )
    forecaster = full_forecast_model.forecaster
    forecaster.eval()

    return full_pred_model, forecaster, latent_dim


# =============================================================================
# Sensitivity Analysis
# =============================================================================


def run_perturbation_analysis(
    var_specs: list[dict],
    n_samples: int = N_SAMPLES,
    output_path: Path = OUTPUT_PATH,
) -> tuple[dict, dict]:
    """
    Run perturbation-based sensitivity analysis.

    Args:
        var_specs: List of variable specifications from build_grouped_variable_specs.
        n_samples: Number of perturbation samples to generate.
        output_path: Path to save results CSV.

    Returns:
        Tuple of (problem_dict, importance_results).
    """
    import random

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    # Load data and models
    dataset = load_dataset(DATASET_PATH)
    full_pred_model, forecaster, latent_dim = load_models(dataset)
    projection_dataset = DatasetProjections2030(dataset)

    # Determine countries to analyze
    countries = EU27_COUNTRIES if EU_MODE else [TARGET_COUNTRY]

    # Extract baseline values for each country
    baseline_inputs = {}
    baseline_inputs_prev = {}
    baseline_ctx_prev = {}
    baseline_ctx_current = {}
    prev_emissions = {}

    for country in countries:
        idx_baseline = dataset.index_map.get((country, BASELINE_YEAR))
        idx_prev_year = dataset.index_map.get((country, BASELINE_YEAR - 1))

        if idx_baseline is None or idx_prev_year is None:
            raise KeyError(f"Missing baseline data for {country}")

        baseline_inputs[country] = (
            dataset.input_df[idx_baseline].squeeze(0).cpu().numpy()
        )
        baseline_inputs_prev[country] = (
            dataset.input_df[idx_prev_year].squeeze(0).cpu().numpy()
        )

        ctx_prev, ctx_current = projection_dataset.get_from_keys_shifted(
            country, TARGET_YEAR
        )
        baseline_ctx_prev[country] = ctx_prev.squeeze(0).cpu().numpy()
        baseline_ctx_current[country] = ctx_current.squeeze(0).cpu().numpy()

        prev_emi = dataset.emi_df[idx_baseline, :].squeeze(0)
        prev_emissions[country] = (
            prev_emi if isinstance(prev_emi, np.ndarray) else prev_emi.cpu().numpy()
        )

    # Compute bounds for each variable group
    var_names = []
    bounds = []

    for spec in var_specs:
        var_names.append(spec["name"])

        # Aggregate baseline value across countries
        per_country_vals = []
        for country in countries:
            if spec["type"] == "input":
                vals = baseline_inputs[country][spec["indices"]]
            else:
                vals = baseline_ctx_current[country][spec["indices"]]

            agg = float(np.mean(vals)) if len(spec["indices"]) > 1 else float(vals[0])
            per_country_vals.append(agg)

        base_value = float(np.mean(per_country_vals))

        # Compute bounds
        if abs(base_value) < 1e-6:
            lo, hi = (
                base_value - PERTURBATION_FRACTION,
                base_value + PERTURBATION_FRACTION,
            )
        else:
            lo = base_value * (1.0 - PERTURBATION_FRACTION)
            hi = base_value * (1.0 + PERTURBATION_FRACTION)

        if lo > hi:
            lo, hi = hi, lo

        bounds.append([lo, hi])

    problem = {"num_vars": len(var_specs), "names": var_names, "bounds": bounds}
    print(f"Problem setup: {len(var_specs)} variables")

    # Generate uniform perturbation samples
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    param_values = np.random.uniform(
        low=lows, high=highs, size=(n_samples, len(var_specs))
    )

    print(f"Generated {n_samples} perturbation samples")

    # Storage for outputs
    n_sectors = len(SECTORS)
    Y = np.zeros((n_samples, n_sectors), dtype=float)

    def run_sample_for_country(sample_vec: np.ndarray, country: str) -> np.ndarray:
        """Run model for a single sample and country."""
        # Copy baseline values
        input_2023 = baseline_inputs[country].copy()
        input_2022 = baseline_inputs_prev[country].copy()
        ctx_prev = baseline_ctx_prev[country].copy()
        ctx_current = baseline_ctx_current[country].copy()

        # Apply perturbations
        for j, spec in enumerate(var_specs):
            val = float(sample_vec[j])
            if spec["type"] == "input":
                for idx in spec["indices"]:
                    input_2023[idx] = val
            else:
                for idx in spec["indices"]:
                    ctx_current[idx] = val

        # Convert to tensors
        t_input_2023 = torch.tensor(
            input_2023, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        t_input_2022 = torch.tensor(
            input_2022, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        t_ctx_prev = torch.tensor(
            ctx_prev, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        t_ctx_current = torch.tensor(
            ctx_current, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            # Encode inputs to latent space
            latent_prev_mean, _ = full_pred_model.encoder(t_input_2023)
            latent_past_mean, _ = full_pred_model.encoder(t_input_2022)

            # Forecast current latent
            latent_current_mean = forecaster(
                latent_prev_mean, latent_past_mean, t_ctx_current, t_ctx_prev
            )

            # Predict emissions
            stacker = torch.cat(
                (latent_current_mean, t_ctx_current, latent_prev_mean, t_ctx_prev),
                dim=1,
            )
            emission_delta, _ = full_pred_model.predictor(stacker)

            prev_emi_tensor = torch.tensor(
                prev_emissions[country], dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            current_emissions = emission_delta + prev_emi_tensor

            return current_emissions.squeeze(0).cpu().numpy()

    # Run all perturbations
    for idx in range(n_samples):
        sample = param_values[idx]

        # Aggregate across countries
        emissions_acc = np.zeros(n_sectors, dtype=float)
        for country in countries:
            emissions_acc += run_sample_for_country(sample, country)

        Y[idx, :] = emissions_acc / float(len(countries))

        if (idx + 1) % 500 == 0 or idx == n_samples - 1:
            print(f"Completed {idx + 1}/{n_samples} samples")

    # Compute Spearman correlations as importance
    importance_results = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sector", "variable", "importance", "correlation_sign"])

        for s in range(n_sectors):
            sector_outputs = Y[:, s]
            importances = []

            for j, name in enumerate(var_names):
                x = param_values[:, j]

                # Handle constant inputs
                if np.allclose(x, x[0]) or np.allclose(
                    sector_outputs, sector_outputs[0]
                ):
                    corr = 0.0
                else:
                    corr, _ = spearmanr(x, sector_outputs)
                    if np.isnan(corr):
                        corr = 0.0

                importance = float(abs(corr))
                sign = float(np.sign(corr)) if corr != 0.0 else 0.0
                importances.append(importance)

                writer.writerow([SECTORS[s], name, importance, sign])

            importance_results[f"sector_{s}"] = dict(
                zip(var_names, importances, strict=False)
            )

    print(f"Results saved to {output_path}")
    return problem, importance_results


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run perturbation sensitivity analysis."""
    dataset = load_dataset(DATASET_PATH)
    input_names = list(dataset.input_variable_names)
    context_names = list(dataset.context_variable_names)

    var_specs = build_grouped_variable_specs(input_names, context_names)
    print(
        f"Built {len(var_specs)} variable groups from {len(input_names)} inputs + {len(context_names)} contexts"
    )

    run_perturbation_analysis(var_specs, n_samples=N_SAMPLES, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
