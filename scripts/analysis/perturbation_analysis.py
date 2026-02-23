"""
Perturbation-based sensitivity analysis for emission predictions.

This script performs sensitivity analysis using random perturbations and
Spearman correlation to identify which input variables most strongly
influence sectoral emission predictions.

Workflow:
    1. Group variables by base name (monthly/quarterly collapsed)
    2. Build per-group bounds around baseline values
    3. Sample N perturbations uniformly within bounds
    4. Run full model for each perturbation (aggregated across countries)
    5. Compute Spearman correlation between inputs and outputs as importance

Variable Modes:
    - "full": Include both input variables and context variables
    - "inputonly": Include only input variables (exclude context)

Usage:
    python -m scripts.analysis.sensitivity.perturbation_analysis

Outputs:
    For each mode (full/inputonly):
    - data/sensitivity/perturbation_results_{mode}.csv

Reference:
    Section 4 "Methods" discusses sensitivity analysis methodology.
    Figure 4 panel (b) shows Spearman correlation results.
"""

from __future__ import annotations

import csv
import pickle
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

# =============================================================================
# Configuration
# =============================================================================

# Variable mode type
VariableMode = Literal["full", "inputonly"]

# Paths
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
VAE_CONFIG_PATH = Path("config/models/vae_config.yaml")
VAE_MODEL_PATH = Path("data/pytorch_models/vae_model.pth")
PREDICTOR_CONFIG_PATH = Path("config/models/co2_predictor_config.yaml")
PREDICTOR_MODEL_PATH = Path("data/pytorch_models/predictor_model.pth")
FORECASTER_CONFIG_PATH = Path("config/models/latent_forecaster_config.yaml")
FORECASTER_MODEL_PATH = Path("data/pytorch_models/forecaster_model.pth")

OUTPUT_DIR = Path("data/sensitivity")

# Analysis settings
EU_MODE = True  # If True, aggregate across EU27; if False, single country
TARGET_GEO = "PL"  # Used if EU_MODE is False
TARGET_YEAR = 2024
BASELINE_YEAR = 2023

# Which modes to run (can be ["full"], ["inputonly"], or ["full", "inputonly"])
RUN_MODES: list[VariableMode] = ["full", "inputonly"]

# EU27 country codes
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

# Sampling settings
N_SAMPLES = 2048 * 8  # Number of perturbation samples
PERTURBATION_FRACTION = 0.3  # +/- 30% perturbation around baseline
SAMPLE_LATENT = False

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Emission sectors
SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]


# =============================================================================
# Utility Functions
# =============================================================================


def load_config(yaml_path: Path) -> SimpleNamespace:
    """Load configuration from YAML file."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    clean = {
        k: v["value"]
        for k, v in raw.items()
        if not k.startswith("_") and k != "wandb_version"
    }
    return SimpleNamespace(**clean)


def load_dataset(path: Path):
    """Load pickled dataset."""
    with open(path, "rb") as f:
        return pickle.load(f)


def strip_suffix(name: str) -> str:
    """
    Remove trailing month/quarter suffixes from variable names.

    Converts 'variable_name_1', 'variable_name_12' to 'variable_name'
    to group monthly/quarterly variants together.
    """
    return re.sub(r"_\d+$", "", name)


def build_grouped_var_specs(
    input_names: list[str],
    context_names: list[str],
    mode: VariableMode = "full",
) -> list[dict]:
    """
    Build variable specifications with monthly/quarterly variants grouped.

    Args:
        input_names: List of input variable names
        context_names: List of context variable names
        mode: Variable mode - "full" includes context, "inputonly" excludes context

    Returns:
        List of variable spec dictionaries with keys:
            - name: Display name for the variable group
            - type: 'input' or 'context'
            - indices: List of indices for this variable group
            - base: Base variable name
    """
    var_specs = []
    seen = {}

    # Process input variables
    for idx, var in enumerate(input_names):
        base = strip_suffix(var)
        if base not in seen:
            seen[base] = {
                "name": base,
                "type": "input",
                "indices": [],
                "base": base,
            }
            var_specs.append(seen[base])
        seen[base]["indices"].append(idx)

    # Process context variables only if mode is "full"
    if mode == "full":
        for idx, var in enumerate(context_names):
            base = strip_suffix(var)
            key_name = "CONTEXT::" + base
            if base not in seen:
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
        dataset: Unified dataset instance

    Returns:
        Tuple of (full_model, forecast_model, vae_model)
    """
    from config.data.output_configs import output_configs
    from scripts.elements.models import (
        Decoder,
        EmissionPredictor,
        Encoder,
        FullLatentForecastingModel,
        FullPredictionModel,
        LatentForecaster,
        VAEModel,
    )

    vae_cfg = load_config(VAE_CONFIG_PATH)
    pred_cfg = load_config(PREDICTOR_CONFIG_PATH)
    fcast_cfg = load_config(FORECASTER_CONFIG_PATH)

    input_dim = len(dataset.input_variable_names)
    context_dim = len(dataset.context_variable_names)

    # Build VAE
    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=vae_cfg.vae_latent_dim,
        num_blocks=vae_cfg.vae_num_blocks,
        dim_blocks=vae_cfg.vae_dim_blocks,
        activation=vae_cfg.vae_activation,
        normalization=vae_cfg.vae_normalization,
        dropout=vae_cfg.vae_dropouts,
        input_dropout=vae_cfg.vae_input_dropouts,
    )
    decoder = Decoder(
        input_dim=input_dim,
        latent_dim=vae_cfg.vae_latent_dim,
        num_blocks=vae_cfg.vae_num_blocks,
        dim_blocks=vae_cfg.vae_dim_blocks,
        activation=vae_cfg.vae_activation,
        normalization=vae_cfg.vae_normalization,
        dropout=vae_cfg.vae_dropouts,
    )
    vae_model = VAEModel(encoder, decoder).to(DEVICE)
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    vae_model.eval()

    # Build predictor
    predictor = EmissionPredictor(
        input_dim=2 * (vae_cfg.vae_latent_dim + context_dim),
        output_configs=output_configs,
        width_block=pred_cfg.pred_width_block,
        dim_block=pred_cfg.pred_dim_block,
        activation=pred_cfg.pred_activation,
        normalization=pred_cfg.pred_normalization,
        dropout=pred_cfg.pred_dropouts,
        num_blocks=pred_cfg.pred_num_blocks,
        uncertainty=True,
    ).to(DEVICE)

    full_model = FullPredictionModel(vae=vae_model, predictor=predictor)
    full_model.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE))
    full_model.eval()

    # Build forecaster
    forecaster = LatentForecaster(
        input_dim=2 * (vae_model.encoder.latent_dim + context_dim),
        latent_dim=vae_model.encoder.latent_dim,
        width_block=fcast_cfg.forecast_width_block,
        dim_block=fcast_cfg.forecast_dim_block,
        activation=fcast_cfg.forecast_activation,
        normalization=fcast_cfg.forecast_normalization,
        dropout=fcast_cfg.forecast_dropouts,
        num_blocks=fcast_cfg.forecast_num_blocks,
    ).to(DEVICE)

    latent_model = FullLatentForecastingModel(vae=vae_model, forecaster=forecaster)
    latent_model.load_state_dict(torch.load(FORECASTER_MODEL_PATH, map_location=DEVICE))
    forecast_model = latent_model.forecaster
    forecast_model.eval()

    return full_model, forecast_model, vae_model


# =============================================================================
# Main Perturbation Analysis
# =============================================================================


def run_perturbation_analysis(
    var_specs: list[dict],
    mode: VariableMode = "full",
    n_samples: int = N_SAMPLES,
) -> tuple[dict, dict]:
    """
    Run perturbation-based sensitivity analysis.

    Args:
        var_specs: List of variable specifications to analyze
        mode: Variable mode - "full" or "inputonly"
        n_samples: Number of perturbation samples

    Returns:
        Tuple of (problem_dict, importance_results_dict)
    """
    # Output filename includes mode
    output_template = f"perturbation_results_{mode}_iter{{}}.csv"

    # Reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Load data and models
    dataset = load_dataset(DATASET_PATH)
    full_model, forecast_model, vae_model = load_models(dataset)

    list(dataset.input_variable_names)
    list(dataset.context_variable_names)

    # Import projection dataset
    from scripts.elements.datasets import DatasetProjections2030

    projection_dataset = DatasetProjections2030(dataset)

    # Determine countries
    countries = EU27_COUNTRIES if EU_MODE else [TARGET_GEO]

    # Build baseline data per country
    baseline_inputs = {}
    baseline_inputs_prev = {}
    baseline_ctx_prev = {}
    baseline_ctx_cur = {}
    prev_emissions = {}

    for geo in countries:
        idx_baseline = dataset.index_map.get((geo, BASELINE_YEAR))
        idx_baseline_prev = dataset.index_map.get((geo, BASELINE_YEAR - 1))

        if idx_baseline is None or idx_baseline_prev is None:
            raise KeyError(f"No baseline data for {geo}")

        baseline_inputs[geo] = dataset.input_df[idx_baseline].squeeze(0).cpu().numpy()
        baseline_inputs_prev[geo] = (
            dataset.input_df[idx_baseline_prev].squeeze(0).cpu().numpy()
        )

        ctx_prev, ctx_cur = projection_dataset.get_from_keys_shifted(geo, TARGET_YEAR)
        baseline_ctx_prev[geo] = ctx_prev.squeeze(0).cpu().numpy()
        baseline_ctx_cur[geo] = ctx_cur.squeeze(0).cpu().numpy()

        prev_e = dataset.emi_df[dataset.index_map[(geo, BASELINE_YEAR)], :].squeeze(0)
        prev_emissions[geo] = (
            prev_e if isinstance(prev_e, np.ndarray) else prev_e.cpu().numpy()
        )

    # Build bounds per variable group
    var_names = []
    bounds = []

    for spec in var_specs:
        var_names.append(spec["name"])

        # Compute aggregate baseline value across countries
        per_country_vals = []
        for geo in countries:
            if spec["type"] == "input":
                vals = baseline_inputs[geo][spec["indices"]]
            else:
                vals = baseline_ctx_cur[geo][spec["indices"]]

            # Average if multiple indices (monthly/quarterly)
            agg = float(np.mean(vals)) if len(spec["indices"]) > 1 else float(vals[0])
            per_country_vals.append(agg)

        base = float(np.mean(per_country_vals))

        # Set bounds as +/- PERTURBATION_FRACTION around baseline
        if abs(base) < 1e-6:
            lo, hi = base - PERTURBATION_FRACTION, base + PERTURBATION_FRACTION
        else:
            lo, hi = (
                base * (1.0 - PERTURBATION_FRACTION),
                base * (1.0 + PERTURBATION_FRACTION),
            )

        if lo > hi:
            lo, hi = hi, lo

        bounds.append([lo, hi])

    problem_like = {
        "num_vars": len(var_specs),
        "names": var_names,
        "bounds": bounds,
    }
    print(f"Perturbation problem: {len(var_specs)} variables")

    # Generate uniform random samples
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    param_values = np.random.uniform(
        low=lows, high=highs, size=(n_samples, len(var_specs))
    )
    n_runs = param_values.shape[0]
    print(f"Generated {n_runs} perturbation samples")

    # Prepare output array
    n_sectors = len(SECTORS)
    Y = np.zeros((n_runs, n_sectors), dtype=float)

    def run_sample_for_country(sample_vec: np.ndarray, geo: str) -> np.ndarray:
        """
        Run the full prediction pipeline for one perturbation sample and country.

        Copies the baseline input/context arrays for the given country,
        applies the perturbation values from ``sample_vec``, encodes to
        latent space, forecasts the current-year latent, and predicts
        absolute sectoral emissions.

        Args:
            sample_vec: 1-D array of length ``len(var_specs)`` with the
                perturbed values for each variable group.
            geo: Two-letter country code (e.g. ``"PL"``).

        Returns:
            1-D numpy array of shape ``(n_sectors,)`` with predicted
            absolute emissions for each sector.
        """
        in_cur = baseline_inputs[geo].copy()
        in_prev = baseline_inputs_prev[geo].copy()
        ctx_prev = baseline_ctx_prev[geo].copy()
        ctx_cur = baseline_ctx_cur[geo].copy()

        # Apply perturbations
        for j, spec in enumerate(var_specs):
            val = float(sample_vec[j])
            if spec["type"] == "input":
                for ii in spec["indices"]:
                    in_cur[ii] = val
            else:
                for ii in spec["indices"]:
                    ctx_cur[ii] = val

        # To tensors
        t_in_cur = torch.tensor(in_cur, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        t_in_prev = torch.tensor(in_prev, dtype=torch.float32, device=DEVICE).unsqueeze(
            0
        )
        t_ctx_prev = torch.tensor(
            ctx_prev, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        t_ctx_cur = torch.tensor(ctx_cur, dtype=torch.float32, device=DEVICE).unsqueeze(
            0
        )

        with torch.no_grad():
            # Encode — use encoder directly (returns mean, log_var)
            l_prev_mean, _ = full_model.encoder(t_in_cur)
            l_past_mean, _ = full_model.encoder(t_in_prev)

            # Forecast current latent
            latent_cur_mean = forecast_model(
                l_prev_mean, l_past_mean, t_ctx_cur, t_ctx_prev
            )

            # Predict emissions
            stacker = torch.cat(
                (latent_cur_mean, t_ctx_cur, l_prev_mean, t_ctx_prev), dim=1
            )
            emis_delta, _ = full_model.predictor(stacker)

            prev_emi_tensor = torch.tensor(
                prev_emissions[geo], dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            current_emis = emis_delta + prev_emi_tensor

            return current_emis.squeeze(0).cpu().numpy()

    # Run all samples
    for idx in range(n_runs):
        sample = param_values[idx]
        emis_acc = np.zeros((n_sectors,), dtype=float)

        for geo in countries:
            emis_acc += run_sample_for_country(sample, geo)

        Y[idx, :] = emis_acc / float(len(countries))

        if (idx + 1) % 200 == 0 or idx == n_runs - 1:
            print(f"Completed {idx + 1}/{n_runs} runs")

    # Compute importance via Spearman correlation
    importance_results = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / output_template

    with open(out_csv, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["sector", "var", "importance_abs_corr", "pearson_corr_sign"])

        for s in range(n_sectors):
            Ys = Y[:, s]
            importances = []

            for j, name in enumerate(var_names):
                x = param_values[:, j]

                # Compute Spearman correlation robustly
                if np.all(np.isclose(x, x[0])) or np.all(np.isclose(Ys, Ys[0])):
                    corr = 0.0
                else:
                    corr, _ = spearmanr(x, Ys)
                    if np.isnan(corr):
                        corr = 0.0

                score = float(abs(corr))
                sign = float(np.sign(corr)) if corr != 0.0 else 0.0
                importances.append(score)

                writer.writerow([SECTORS[s], name, score, sign])

            importance_results[f"sector_{s}"] = dict(
                zip(var_names, importances, strict=False)
            )

    print(f"Perturbation analysis ({mode}): results written to {out_csv}")
    return problem_like, importance_results


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point for perturbation sensitivity analysis."""
    print("=" * 70)
    print("PERTURBATION-BASED SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Mode: {'EU27 aggregate' if EU_MODE else f'Single country ({TARGET_GEO})'}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Perturbation: ±{PERTURBATION_FRACTION * 100:.0f}%")
    print(f"Variable modes to run: {RUN_MODES}")
    print()

    # Load dataset to get variable names
    dataset = load_dataset(DATASET_PATH)
    input_names = list(dataset.input_variable_names)
    context_names = list(dataset.context_variable_names)

    # Run analysis for each mode
    for mode in RUN_MODES:
        print("\n" + "=" * 70)
        print(f"RUNNING MODE: {mode.upper()}")
        print("=" * 70)

        # Build grouped variable specifications for this mode
        var_specs = build_grouped_var_specs(input_names, context_names, mode=mode)
        print(f"Total variable groups ({mode}): {len(var_specs)}")

        if mode == "inputonly":
            print("(Context variables excluded)")

        # Run analysis
        run_perturbation_analysis(var_specs, mode=mode, n_samples=N_SAMPLES)

    print("\n" + "=" * 70)
    print("Perturbation analysis complete for all modes!")
    print("=" * 70)


def run_single_mode(mode: VariableMode) -> None:
    """
    Run perturbation analysis for a single mode.

    Args:
        mode: Variable mode — ``"full"`` includes both input and context
            variables; ``"inputonly"`` excludes context variables.
    """
    dataset = load_dataset(DATASET_PATH)
    input_names = list(dataset.input_variable_names)
    context_names = list(dataset.context_variable_names)

    var_specs = build_grouped_var_specs(input_names, context_names, mode=mode)
    print(f"Running perturbation analysis ({mode}): {len(var_specs)} variable groups")

    run_perturbation_analysis(var_specs, mode=mode, n_samples=N_SAMPLES)


if __name__ == "__main__":
    main()
