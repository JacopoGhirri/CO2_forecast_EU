"""
Temperature calibration for MC projection uncertainty intervals.

For each (country, sector) pair we find a scalar T such that the rescaled
90% interval  [mean - 1.645·T·σ_mc,  mean + 1.645·T·σ_mc]  achieves
empirical 90% coverage on the historical period 2010-2023.

Procedure
---------
1.  Load the trained models and dataset (mirrors generate_projections.py).
2.  For every observed (country, year) in the historical period run
    N_CAL forward passes through
        encoder  →  reparameterize(μ, σ)  →  predictor
    collecting emission samples.  The current-year emissions come from
    dataset.emi_df (the *scaled* ground truth).
3.  For each (country, sector) accumulate the standardised residuals
        r = (y_true - ŷ_mean) / ŷ_std
    across all years.
4.  Solve for T at the 90 % level:
        T = empirical_90th_percentile_of(|r|) / 1.645
5.  Save calibration_temperatures.csv and apply_calibration.py helper.

Usage
-----
    python -m scripts.calibration.calibrate_uncertainty

Outputs
-------
    data/calibration/calibration_temperatures.csv
        Columns: geo, sector, T, n_obs, coverage_before, coverage_after
    data/calibration/calibrated_projections.csv   (optional, see flag)
        mc_projections.csv with intervals rescaled by T
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config.data.output_configs import output_configs
from scripts.elements.models import (
    Decoder,
    EmissionPredictor,
    Encoder,
    FullLatentForecastingModel,
    FullPredictionModel,
    LatentForecaster,
    VAEModel,
    reparameterize,
)
from scripts.utils import load_config, load_dataset

# =============================================================================
# Configuration
# =============================================================================

SEED = 0
N_CAL = 500  # MC samples per (country, year) for calibration
TARGET_COVERAGE = 0.90
Z_TARGET = 1.6449  # Gaussian z-score for 90 % two-sided interval

# Whether to also write a calibrated version of mc_projections.csv
WRITE_CALIBRATED_PROJECTIONS = True

# Paths
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
VAE_CONFIG_PATH = Path("config/models/vae_config.yaml")
PREDICTOR_CONFIG_PATH = Path("config/models/co2_predictor_config.yaml")
FORECASTER_CONFIG_PATH = Path("config/models/latent_forecaster_config.yaml")
VAE_MODEL_PATH = Path("data/pytorch_models/vae_model.pth")
PREDICTOR_MODEL_PATH = Path("data/pytorch_models/predictor_model.pth")
FORECASTER_MODEL_PATH = Path("data/pytorch_models/forecaster_model.pth")
MC_PROJECTIONS_PATH = Path("data/projections/mc_projections.csv")
OUTPUT_DIR = Path("data/calibration")

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
EMISSION_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model loading  (mirrors generate_projections.py)
# =============================================================================


def set_eval_mode(model: torch.nn.Module) -> None:
    model.eval()


def load_models(dataset):
    vae_config = load_config(VAE_CONFIG_PATH)
    predictor_config = load_config(PREDICTOR_CONFIG_PATH)
    forecaster_config = load_config(FORECASTER_CONFIG_PATH)

    input_dim = len(dataset.input_variable_names)
    context_dim = len(dataset.context_variable_names)
    latent_dim = vae_config.vae_latent_dim

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
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location="cpu"))
    set_eval_mode(vae_model)

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

    full_pred = FullPredictionModel(vae=vae_model, predictor=predictor)
    full_pred.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location="cpu"))
    set_eval_mode(full_pred)

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

    full_forecast = FullLatentForecastingModel(vae=vae_model, forecaster=forecaster)
    full_forecast.load_state_dict(torch.load(FORECASTER_MODEL_PATH, map_location="cpu"))
    set_eval_mode(full_forecast)

    return vae_model, predictor, latent_dim


# =============================================================================
# Per-observation MC sampling  (mirrors project_country inference)
# =============================================================================


def sample_emissions_for_obs(
    country: str,
    year: int,
    dataset,
    vae_model: VAEModel,
    predictor: EmissionPredictor,
    n_samples: int,
) -> np.ndarray | None:
    """
    Draw n_samples emission vectors for a single (country, year) observation
    using the same reparameterization sampling as generate_projections.py.

    The previous-year context is taken from dataset.context_df when
    available, falling back to the current year.

    Returns
    -------
    samples : ndarray of shape (n_samples, n_sectors), in scaled space,
              or None if the observation cannot be found.
    """
    idx = dataset.index_map.get((country, year))
    if idx is None:
        return None

    x_cur = dataset.input_df[idx].unsqueeze(0).to(DEVICE)
    c_cur = dataset.context_df[idx].unsqueeze(0).to(DEVICE)

    prev_idx = dataset.index_map.get((country, year - 1))
    if prev_idx is not None:
        x_prev = dataset.input_df[prev_idx].unsqueeze(0).to(DEVICE)
        c_prev = dataset.context_df[prev_idx].unsqueeze(0).to(DEVICE)
        y_prev = dataset.emi_df[prev_idx].unsqueeze(0).to(DEVICE)
    else:
        x_prev = x_cur
        c_prev = c_cur
        y_prev = dataset.emi_df[idx].unsqueeze(0).to(DEVICE)

    samples = []
    with torch.no_grad():
        # Encode both time steps once; re-sample the latent N_CAL times
        mean_cur, log_var_cur = vae_model.encoder(x_cur)
        mean_prev, log_var_prev = vae_model.encoder(x_prev)

        for _ in range(n_samples):
            z_cur = reparameterize(mean_cur, torch.exp(0.5 * log_var_cur))
            z_prev = reparameterize(mean_prev, torch.exp(0.5 * log_var_prev))

            pred_input = torch.cat((z_cur, c_cur, z_prev, c_prev), dim=1)
            delta, _ = predictor(pred_input)  # (1, n_sectors)
            emission = (delta + y_prev).squeeze(0).cpu().numpy()
            samples.append(emission)

    return np.stack(samples)  # (n_samples, n_sectors)


# =============================================================================
# Calibration
# =============================================================================


def compute_temperature(
    residuals: np.ndarray,
    z_target: float = Z_TARGET,
) -> float:
    """
    Given an array of standardised residuals r = (y-ŷ)/σ_mc,
    return T such that z_target · T equals the empirical quantile
    that achieves 90 % coverage of |r|.

    Equivalently:  T = quantile(|r|, 0.90) / z_target
    """
    return float(np.quantile(np.abs(residuals), TARGET_COVERAGE) / z_target)


def coverage_at_T(residuals: np.ndarray, T: float, z: float = Z_TARGET) -> float:
    """Fraction of |r| ≤ z·T  (should equal TARGET_COVERAGE after calibration)."""
    return float(np.mean(np.abs(residuals) <= z * T))


def run_calibration(dataset, vae_model, predictor):
    """
    Main calibration loop.

    For each country and year in the historical dataset:
      - draw N_CAL emission samples
      - compute mean and std across samples
      - accumulate standardised residual vs ground truth

    Then for each (country, sector) compute T.

    Returns
    -------
    temperatures : dict  { (country, sector) -> T }
    records      : list of dicts for diagnostics CSV
    """
    # Collect all available (country, year) pairs from the dataset
    obs_list = [
        (row["geo"], row["year"])
        for _, row in dataset.keys.iterrows()
        if row["geo"] in EU27_COUNTRIES
    ]

    print(
        f"Calibrating over {len(obs_list)} (country, year) observations "
        f"with {N_CAL} MC samples each ..."
    )

    # residuals[country][sector] = list of floats
    residuals: dict[str, dict[str, list[float]]] = {
        c: {s: [] for s in EMISSION_SECTORS} for c in EU27_COUNTRIES
    }

    for i, (country, year) in enumerate(obs_list):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(obs_list)}  ({country}, {year})")

        # Deterministic seed per observation for reproducibility
        random.seed(SEED + i)
        torch.manual_seed(SEED + i)
        if DEVICE == "cuda":
            torch.cuda.manual_seed(SEED + i)

        samples = sample_emissions_for_obs(
            country, year, dataset, vae_model, predictor, N_CAL
        )
        if samples is None:
            continue

        # Ground truth in scaled space
        idx = dataset.index_map[(country, year)]
        y_true = dataset.emi_df[idx].cpu().numpy()  # (n_sectors,)

        sample_mean = samples.mean(axis=0)  # (n_sectors,)
        sample_std = samples.std(axis=0)  # (n_sectors,)

        for j, sector in enumerate(EMISSION_SECTORS):
            if sample_std[j] < 1e-9:
                # Degenerate: no spread — skip this obs for this sector
                continue
            r = (y_true[j] - sample_mean[j]) / sample_std[j]
            residuals[country][sector].append(r)

    # --- Compute T per (country, sector) ---
    records = []
    temperatures: dict[tuple[str, str], float] = {}

    for country in EU27_COUNTRIES:
        for sector in EMISSION_SECTORS:
            res = np.asarray(residuals[country][sector])
            n = len(res)

            if n < 5:
                # Too few observations: fall back to global T later
                temperatures[(country, sector)] = np.nan
                records.append(
                    dict(
                        geo=country,
                        sector=sector,
                        T=np.nan,
                        n_obs=n,
                        coverage_before=np.nan,
                        coverage_after=np.nan,
                    )
                )
                continue

            # Empirical coverage before calibration (T=1)
            cov_before = coverage_at_T(res, T=1.0)
            T = compute_temperature(res)
            cov_after = coverage_at_T(res, T=T)

            temperatures[(country, sector)] = T
            records.append(
                dict(
                    geo=country,
                    sector=sector,
                    T=round(T, 6),
                    n_obs=n,
                    coverage_before=round(cov_before, 4),
                    coverage_after=round(cov_after, 4),
                )
            )

    # --- Fill NaN T values with global fallback ---
    valid_Ts = [v for v in temperatures.values() if not np.isnan(v)]
    global_T = float(np.median(valid_Ts)) if valid_Ts else 1.0
    print(f"\nGlobal median T (fallback for sparse cells): {global_T:.4f}")

    for key in temperatures:
        if np.isnan(temperatures[key]):
            temperatures[key] = global_T
            # Update record
            for rec in records:
                if rec["geo"] == key[0] and rec["sector"] == key[1]:
                    rec["T"] = round(global_T, 6)
                    rec["note"] = "fallback_global"

    return temperatures, records, global_T


# =============================================================================
# Apply calibration to mc_projections.csv
# =============================================================================


def apply_calibration_to_projections(
    temperatures: dict[tuple[str, str], float],
    dataset,
) -> pd.DataFrame:
    """
    Build calibrated CI columns from mc_projections.csv, stored at the
    (geo, year) level as **total CO2 in kg** (summed across all sectors,
    multiplied by population).

    The calibration transform applied per sector is:
        y_cal_s = mean_s + T_s * (y_mc_s - mean_s)

    For each MC sample, the calibrated sector values are then summed and
    multiplied by population to get a total kg CO2 value.  The 5th/95th
    percentiles are taken across MC samples of this total — which is
    exactly what the boxplot panels compute.  Summing per-sector quantiles
    independently (the previous approach) overstates the width because it
    assumes perfect cross-sector correlation.

    Output columns:
        geo, year,
        total_mean          (kg, mean across MC samples, sum of sectors × pop)
        total_p05           (kg, raw 5th pct of summed total)
        total_p95           (kg, raw 95th pct)
        total_p05_cal       (kg, calibrated 5th pct — matches boxplot whiskers)
        total_p95_cal       (kg, calibrated 95th pct)

    The interactive plot divides by 1e6 to display Mt, consistent with
    how total_CO2_mean is built and plotted.
    """
    print("\nApplying calibration to mc_projections.csv ...")
    df = pd.read_csv(MC_PROJECTIONS_PATH)
    df["geo"] = df["geo"].astype(str)

    # Un-normalise all sectors to physical units (kg CO2 / hab)
    for sector in EMISSION_SECTORS:
        params = dataset.precomputed_scaling_params[sector]
        df[f"{sector}_phys"] = (
            df[f"emissions_{sector}"] * params["std"] + params["mean"]
        ).clip(lower=0)

    # Merge population so we can compute kg totals per MC sample
    pop_hist = pd.read_csv("data/full_timeseries/population.csv")
    pop_proj = pd.read_csv("data/full_timeseries/projections/population.csv")
    pop_df = pd.concat([pop_hist, pop_proj], ignore_index=True)
    pop_df["population"] = pop_df["population:POP_NC"].astype(float)
    pop_df = (
        pop_df[["geo", "year", "population"]]
        .groupby(["geo", "year"], as_index=False)["population"]
        .mean()
    )

    df = df.merge(pop_df, on=["geo", "year"], how="left")

    # Compute per-sector calibrated values for each MC sample, then sum to total
    for sector in EMISSION_SECTORS:
        # Will be filled per (geo, year) group below
        df[f"{sector}_cal"] = np.nan

    rows = []
    for (geo, year), grp in df.groupby(["geo", "year"]):
        pop = grp["population"].iloc[0]
        if np.isnan(pop):
            continue

        # Per-sector calibration: rescale each sample around its sector mean
        total_raw = np.zeros(len(grp))  # sum of raw sector values × pop
        total_cal = np.zeros(len(grp))  # sum of calibrated sector values × pop

        for sector in EMISSION_SECTORS:
            T = temperatures.get((geo, sector), 1.0)
            vals = grp[f"{sector}_phys"].values  # kg/hab, shape (n_mc,)
            mean_s = vals.mean()

            vals_cal = mean_s + T * (vals - mean_s)

            total_raw += vals * pop  # kg total, this sector
            total_cal += vals_cal * pop  # kg total, calibrated

        rows.append(
            dict(
                geo=geo,
                year=year,
                total_mean=round(total_raw.mean(), 2),
                total_p05=round(np.percentile(total_raw, 5), 2),
                total_p95=round(np.percentile(total_raw, 95), 2),
                total_p05_cal=round(np.percentile(total_cal, 5), 2),
                total_p95_cal=round(np.percentile(total_cal, 95), 2),
            )
        )

    return pd.DataFrame(rows)


# =============================================================================
# Diagnostics
# =============================================================================


def print_summary(records: list[dict], global_T: float) -> None:
    df = pd.DataFrame(records)

    print("\n" + "=" * 65)
    print("CALIBRATION SUMMARY")
    print("=" * 65)
    print(f"  Target coverage  : {TARGET_COVERAGE:.0%}")
    print(f"  Global median T  : {global_T:.4f}")
    print(
        f"  T < 1 (deflate)  : {(df['T'] < 1).sum()} cells  (intervals were too wide)"
    )
    print(
        f"  T > 1 (inflate)  : {(df['T'] > 1).sum()} cells  (intervals were too narrow)"
    )
    print(f"  T == 1 (fallback): {(df['T'] == global_T).sum()} cells")

    print("\n  Per-sector median T:")
    for sector in EMISSION_SECTORS:
        sec = df[df["sector"] == sector]["T"]
        print(
            f"    {sector:<16s}  median={sec.median():.3f}  "
            f"[{sec.min():.3f}, {sec.max():.3f}]"
        )

    print("\n  Pre- vs post-calibration coverage (mean across cells with ≥5 obs):")
    valid = df.dropna(subset=["coverage_before", "coverage_after"])
    print(
        f"    Before: {valid['coverage_before'].mean():.3f}  "
        f"After: {valid['coverage_after'].mean():.3f}  "
        f"(target {TARGET_COVERAGE:.2f})"
    )
    print("=" * 65)


# =============================================================================
# Entry point
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading dataset from {DATASET_PATH} ...")
    dataset = load_dataset(DATASET_PATH)
    dataset.input_df = dataset.input_df.cpu()
    dataset.context_df = dataset.context_df.cpu()
    dataset.emi_df = dataset.emi_df.cpu()

    print("Loading models ...")
    vae_model, predictor, latent_dim = load_models(dataset)

    # Move dataset tensors to DEVICE for inference
    dataset.input_df = dataset.input_df.to(DEVICE)
    dataset.context_df = dataset.context_df.to(DEVICE)
    dataset.emi_df = dataset.emi_df.to(DEVICE)

    temperatures, records, global_T = run_calibration(dataset, vae_model, predictor)

    # Save temperature table
    temp_df = pd.DataFrame(records)
    temp_path = OUTPUT_DIR / "calibration_temperatures.csv"
    temp_df.to_csv(temp_path, index=False)
    print(f"\nCalibration temperatures saved to {temp_path}")

    print_summary(records, global_T)

    if WRITE_CALIBRATED_PROJECTIONS:
        dataset.emi_df = dataset.emi_df.cpu()  # back to CPU for pandas ops
        cal_df = apply_calibration_to_projections(temperatures, dataset)
        cal_path = OUTPUT_DIR / "calibrated_projections.csv"
        cal_df.to_csv(cal_path, index=False)
        print(f"Calibrated projections saved to {cal_path}")
        print(
            "\nColumns: geo, year, sector, "
            "median, p05, p95 (raw),  "
            "median_cal, p05_cal, p95_cal (calibrated),  T"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
