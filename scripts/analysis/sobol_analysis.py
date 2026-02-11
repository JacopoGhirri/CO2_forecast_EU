"""
Sobol sensitivity analysis for emission predictions.

This script performs variance-based global sensitivity analysis using Sobol indices
to identify which input variables most strongly influence sectoral emission predictions.
Total-order Sobol indices (ST) are used to capture each variable's full contribution
to output variance, including all interaction effects with other variables.

The script supports both EU-wide aggregated analysis and single-country analysis,
with options to analyze either emission predictions or model uncertainty.

Variable Modes:
    - "full": Include both input variables and context variables
    - "inputonly": Include only input variables (exclude context)

Usage:
    python -m scripts.analysis.sensitivity.sobol_analysis

Outputs:
    For each mode (full/inputonly):
    - data/sensitivity/sobol_results_{mode}.csv

Reference:
    Section 4 "Methods" discusses sensitivity analysis methodology.
    Figure 4 panel (a) shows Sobol sensitivity results.
"""

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

# SALib for Sobol analysis
try:
    from SALib.analyze import sobol
    from SALib.sample import saltelli
except ImportError as e:
    raise ImportError("SALib not found. Install with: pip install SALib") from e


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
ANALYZE_UNCERTAINTY = (
    True  # If True, analyze uncertainty outputs; if False, analyze emissions
)
EU_MODE = True  # If True, aggregate across EU27; if False, single country
TARGET_GEO = "SE"  # Used if EU_MODE is False
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

# Sobol sampling settings
N_BASE = 1024  # Base sample size for Saltelli sampling
PERTURBATION_FRACTION = 0.3  # +/- 30% perturbation around baseline
SAMPLE_LATENT = False  # Whether to sample from latent distribution

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
            if base not in seen:
                seen[base] = {
                    "name": "CONTEXT::" + base,
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
    # Import here to avoid circular imports
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
# Main Sobol Analysis
# =============================================================================


def run_sobol(
    var_specs: list[dict],
    mode: VariableMode = "full",
) -> tuple[dict, dict]:
    """
    Run Sobol sensitivity analysis for all sectors.

    Computes total-order Sobol indices (ST) to quantify each variable's
    contribution to output variance, including all interaction effects.

    Args:
        var_specs: List of variable specifications to analyze
        mode: Variable mode - "full" or "inputonly"

    Returns:
        Tuple of (problem_dict, sobol_results_dict)
    """
    output_file = f"sobol_results_{mode}.csv"

    # Set random seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(1)

    # Load data and models
    dataset = load_dataset(DATASET_PATH)
    full_model, forecast_model, vae_model = load_models(dataset)

    list(dataset.input_variable_names)
    list(dataset.context_variable_names)

    # Import projection dataset
    from scripts.elements.datasets import DatasetProjections2030

    projection_dataset = DatasetProjections2030(dataset)

    # Determine countries to analyze
    countries = EU27_COUNTRIES if EU_MODE else [TARGET_GEO]

    # Build baseline data for each country
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

    # Build bounds for each variable group
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

    # Define SALib problem
    problem = {
        "num_vars": len(var_specs),
        "names": var_names,
        "bounds": bounds,
    }
    print(f"SALib problem: {len(var_specs)} variables")

    # Generate Saltelli samples
    param_values = saltelli.sample(problem, N_BASE, calc_second_order=False)
    n_runs = param_values.shape[0]
    print(f"Generated {n_runs} samples")

    # Prepare output array
    n_sectors = len(SECTORS)
    Y = np.zeros((n_runs, n_sectors), dtype=float)

    def run_sample_for_country(sample_vec: np.ndarray, geo: str) -> np.ndarray:
        """Run model for a single sample and country."""
        # Copy baselines
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

        # Convert to tensors
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
            # Encode inputs
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
            emis_delta, uncertainties = full_model.predictor(stacker)

            prev_emi_tensor = torch.tensor(
                prev_emissions[geo], dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            current_emis = emis_delta + prev_emi_tensor

            if ANALYZE_UNCERTAINTY:
                return uncertainties.squeeze(0).cpu().numpy()
            else:
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

    # Analyze results and save
    sobol_results = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / output_file

    with open(out_csv, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["sector", "var", "S1", "S1_conf", "ST", "ST_conf"])

        for s in range(n_sectors):
            Ys = Y[:, s]
            Si = sobol.analyze(
                problem, Ys, calc_second_order=False, print_to_console=False
            )
            sobol_results[f"sector_{s}"] = Si

            for j, name in enumerate(var_names):
                writer.writerow(
                    [
                        SECTORS[s],
                        name,
                        Si["S1"][j],
                        Si["S1_conf"][j],
                        Si["ST"][j],
                        Si["ST_conf"][j],
                    ]
                )

    print(f"Results ({mode}) written to {out_csv}")
    return problem, sobol_results


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point for Sobol sensitivity analysis."""
    print("=" * 70)
    print("SOBOL SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Mode: {'EU27 aggregate' if EU_MODE else f'Single country ({TARGET_GEO})'}")
    print(f"Analyzing: {'Uncertainty' if ANALYZE_UNCERTAINTY else 'Emissions'}")
    print(f"Sensitivity index: Total-order (ST)")
    print(f"Base samples: {N_BASE}")
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
        run_sobol(var_specs, mode=mode)

    print("\n" + "=" * 70)
    print("Sobol analysis complete for all modes!")
    print("=" * 70)


def run_single_mode(mode: VariableMode) -> None:
    """
    Run Sobol analysis for a single mode.

    Args:
        mode: Variable mode - "full" or "inputonly"
    """
    dataset = load_dataset(DATASET_PATH)
    input_names = list(dataset.input_variable_names)
    context_names = list(dataset.context_variable_names)

    var_specs = build_grouped_var_specs(input_names, context_names, mode=mode)
    print(f"Running Sobol analysis ({mode}): {len(var_specs)} variable groups")

    run_sobol(var_specs, mode=mode)


if __name__ == "__main__":
    main()