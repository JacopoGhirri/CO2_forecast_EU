"""
Decode forecasted latent states back to the original input space.

Uses the IDENTICAL autoregressive latent chain as generate_projections.py
(same seeds, same initialization from 2022-2023, same forecaster logic),
but passes each forecasted latent through the VAE Decoder to recover
the reconstructed input variables, then rescales them to physical units.

Usage:
    python -m scripts.inference.generate_decoded_projections

Outputs:
    data/projections/decoded_projections.csv
    Columns: [mc_sample, geo, year, <input_variable_names>...]
    Values are in the original (unscaled) physical units.
"""

import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config.data.output_configs import output_configs
from scripts.elements.datasets import DatasetProjections2030, DatasetUnified
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
# Configuration — must match generate_projections.py exactly
# =============================================================================

SEED = 0
N_MC_SAMPLES = 10000
CHUNK_SIZE = 100

OUTPUT_PATH = Path("data/projections/decoded_projections.csv")
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")

VAE_CONFIG_PATH = Path("config/models/vae_config.yaml")
PREDICTOR_CONFIG_PATH = Path("config/models/co2_predictor_config.yaml")
FORECASTER_CONFIG_PATH = Path("config/models/latent_forecaster_config.yaml")

VAE_MODEL_PATH = Path("data/pytorch_models/vae_model.pth")
PREDICTOR_MODEL_PATH = Path("data/pytorch_models/predictor_model.pth")
FORECASTER_MODEL_PATH = Path("data/pytorch_models/forecaster_model.pth")

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

PROJECTION_YEARS = range(2024, 2031)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model loading — identical to generate_projections.py
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

    full_pred_model = FullPredictionModel(vae=vae_model, predictor=predictor)
    full_pred_model.load_state_dict(
        torch.load(PREDICTOR_MODEL_PATH, map_location="cpu")
    )
    set_eval_mode(full_pred_model)

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
        torch.load(FORECASTER_MODEL_PATH, map_location="cpu")
    )
    set_eval_mode(full_forecast_model)

    return vae_model, predictor, forecaster, latent_dim


# =============================================================================
# Inverse scaling helpers
# =============================================================================


def build_inverse_scalers(dataset):
    """
    Returns a dict: variable_name -> (mean, std) or (min, max)
    for z-score or min-max scaling respectively.
    Only input variables are included (not context, not emissions).
    """
    scalers = {}
    for var in dataset.input_variable_names:
        params = dataset.precomputed_scaling_params.get(var)
        if params is None:
            continue
        scalers[var] = params
    return scalers


def inverse_scale(
    scaled_values: np.ndarray, scalers: dict, variable_names: list, scaling_type: str
) -> np.ndarray:
    """
    Inverse-transform a (N, D) array of scaled values back to physical units.
    """
    result = scaled_values.copy()
    for j, var in enumerate(variable_names):
        params = scalers.get(var)
        if params is None:
            continue
        if scaling_type == "normalization":
            result[:, j] = scaled_values[:, j] * params["std"] + params["mean"]
        elif scaling_type == "maxmin":
            result[:, j] = (
                scaled_values[:, j] * (params["max"] - params["min"]) + params["min"]
            )
    return result


# =============================================================================
# Per-country projection — identical latent chain to generate_projections.py
# =============================================================================


def project_country_decoded(
    country: str,
    mc_sample: int,
    dataset,
    projection_dataset,
    vae_model: VAEModel,
    forecaster: LatentForecaster,
    latent_dim: int,
) -> list[list]:
    """
    Runs the same autoregressive latent chain as project_country() in
    generate_projections.py, but instead of predicting emissions, decodes
    each forecasted latent back to the input variable space.

    Returns a list of rows, one per year, each containing:
    [mc_sample, country, year, decoded_var_0, decoded_var_1, ...]
    in the SCALED space (before inverse transform — caller rescales).
    """
    results = []

    # ── Initialise from 2022-2023 (identical to generate_projections.py) ──
    idx_2023 = dataset.index_map.get((country, 2023))
    input_2023 = dataset.input_df[idx_2023].unsqueeze(0).to(DEVICE)
    mean_2023, log_var_2023 = vae_model.encoder(input_2023)
    latent_prev = reparameterize(mean_2023, torch.exp(0.5 * log_var_2023))

    idx_2022 = dataset.index_map.get((country, 2022))
    input_2022 = dataset.input_df[idx_2022].unsqueeze(0).to(DEVICE)
    mean_2022, log_var_2022 = vae_model.encoder(input_2022)
    reparameterize(mean_2022, torch.exp(0.5 * log_var_2022))

    avg_log_var = (log_var_2023 + log_var_2022) / 2

    mean_prev = mean_2023
    mean_past = mean_2022

    for year in PROJECTION_YEARS:
        context_prev, context_current = projection_dataset.get_from_keys_shifted(
            country, year
        )
        context_prev = context_prev.unsqueeze(0).to(DEVICE)
        context_current = context_current.unsqueeze(0).to(DEVICE)

        # Forecast latent mean — identical call signature to generate_projections.py
        mean_current = forecaster(mean_prev, mean_past, context_current, context_prev)

        # Sample latent using historical variance — identical to generate_projections.py
        latent_current = reparameterize(mean_current, torch.exp(0.5 * avg_log_var))

        # Decode the forecasted latent to the input variable space
        decoded = vae_model.decoder(latent_current)  # (1, input_dim)

        row = [mc_sample, country, year] + decoded.squeeze(0).cpu().tolist()
        results.append(row)

        # Update chain — identical to generate_projections.py
        latent_prev = latent_current
        mean_past = mean_prev
        mean_prev = mean_current

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading dataset from {DATASET_PATH}...")

    dataset = load_dataset(DATASET_PATH)
    dataset.input_df = dataset.input_df.cpu()
    dataset.context_df = dataset.context_df.cpu()
    dataset.emi_df = dataset.emi_df.cpu()

    scaling_type = getattr(dataset, "scaling_type", "normalization")
    input_variable_names = list(dataset.input_variable_names)
    scalers = build_inverse_scalers(dataset)

    print(f"Input variables: {len(input_variable_names)}")
    print(f"Scaling type: {scaling_type}")

    print("Loading models...")
    vae_model, predictor, forecaster, latent_dim = load_models(dataset)

    projection_dataset = DatasetProjections2030(dataset)

    # ── CSV header ─────────────────────────────────────────────────────────
    # Scaled columns (z-score space)
    scaled_cols = [f"{v}__scaled" for v in input_variable_names]
    # Physical unit columns (inverse-transformed)
    phys_cols = list(input_variable_names)

    header = ["mc_sample", "geo", "year"] + phys_cols + scaled_cols

    with open(OUTPUT_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print(f"Starting {N_MC_SAMPLES} MC samples → {OUTPUT_PATH}")
    print(f"Output: {len(phys_cols)} variables × scaled + physical")

    for chunk_start in range(0, N_MC_SAMPLES, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N_MC_SAMPLES)
        print(f"Processing MC samples {chunk_start}–{chunk_end - 1}...")

        # Collect raw (scaled) rows first, then batch-inverse-transform
        # Structure: list of (mc_sample, country, year, scaled_array)
        raw_rows = []

        for mc_sample in range(chunk_start, chunk_end):
            # Identical seeding to generate_projections.py
            random.seed(SEED + mc_sample)
            torch.manual_seed(SEED + mc_sample)
            if DEVICE == "cuda":
                torch.cuda.manual_seed(SEED + mc_sample)

            with torch.no_grad():
                for country in EU27_COUNTRIES:
                    country_rows = project_country_decoded(
                        country=country,
                        mc_sample=mc_sample,
                        dataset=dataset,
                        projection_dataset=projection_dataset,
                        vae_model=vae_model,
                        forecaster=forecaster,
                        latent_dim=latent_dim,
                    )
                    raw_rows.extend(country_rows)

            if (mc_sample + 1) % 100 == 0:
                print(f"  Completed MC sample {mc_sample}")

        # ── Batch inverse-transform ────────────────────────────────────────
        # Extract the scaled variable arrays from rows
        meta = [(r[0], r[1], r[2]) for r in raw_rows]  # (mc, geo, year)
        scaled_arr = np.array([r[3:] for r in raw_rows], dtype=np.float32)  # (N, D)

        phys_arr = inverse_scale(
            scaled_arr, scalers, input_variable_names, scaling_type
        )

        # ── Write chunk ────────────────────────────────────────────────────
        with open(OUTPUT_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            for i, (mc_s, geo, yr) in enumerate(meta):
                row_out = (
                    [mc_s, geo, yr]
                    + phys_arr[i].tolist()  # physical units
                    + scaled_arr[i].tolist()  # scaled (z-score)
                )
                writer.writerow(row_out)

        print(f"  Saved chunk {chunk_start}–{chunk_end - 1} ({len(raw_rows)} rows)")

    total_rows = N_MC_SAMPLES * len(EU27_COUNTRIES) * len(list(PROJECTION_YEARS))
    print(f"\nDone. Total rows: {total_rows}")
    print(f"Output: {OUTPUT_PATH}")
    print(
        f"Columns: mc_sample, geo, year + "
        f"{len(phys_cols)} physical vars + "
        f"{len(scaled_cols)} scaled vars"
    )


if __name__ == "__main__":
    main()
