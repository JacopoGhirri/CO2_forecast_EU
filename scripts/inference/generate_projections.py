"""
Monte Carlo emission projections to 2030 — with 2024 emission anchor.

Same as generate_projections.py but uses observed 2024 sectoral emissions
as the baseline `emissions_prev` for the 2024 forecast step, instead of
propagating from 2023.

Key changes vs original:
  - load_2024_emissions() reads air_emissions_yearly_full.csv, aggregates
    sectors with the same grouping_structure used in DatasetUnified, and
    applies the *training-set* scaling parameters already stored in the
    dataset object.  No new scaling is computed.
  - project_country() checks whether year==2024 and, if so, replaces
    emissions_prev with the observed 2024 value before the predictor call.
    The autoregressive chain then continues normally from 2025 onward.
"""

import csv
import random
from pathlib import Path

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
# Configuration  (identical to generate_projections.py)
# =============================================================================

SEED = 0
N_MC_SAMPLES = 10000
CHUNK_SIZE = 100

OUTPUT_PATH = Path("data/projections/mc_projections.csv")
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

EMISSION_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]
PROJECTION_YEARS = range(2024, 2031)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 2024 emission anchor
# =============================================================================


def load_2024_emissions(
    dataset: DatasetUnified,
    path_csvs: str = "data/full_timeseries/",
) -> dict[str, torch.Tensor]:
    """
    Load and scale observed 2024 sectoral emissions for all EU27 countries.

    Uses exactly the same aggregation logic as DatasetUnified._load_emissions()
    (level mode, sector grouping from output_configs) and applies the
    *pre-computed* scaling parameters already stored in `dataset` — so the
    returned tensors live in the same scaled space as dataset.emi_df.

    Args:
        dataset: The trained DatasetUnified instance (provides scaling params).
        path_csvs: Directory containing air_emissions_yearly_full.csv.

    Returns:
        Dict mapping country ISO2 code -> 1-D float32 tensor of shape
        (n_sectors,) in scaled emission space, on CPU.
        Countries with missing 2024 data are omitted (caller must handle).
    """
    cfg = output_configs
    assert cfg["mode"] == "level", (
        f"Only 'level' mode is supported for 2024 anchor; got mode='{cfg['mode']}'"
    )
    assert cfg["output"] == "Sectors", (
        "Only 'Sectors' output is supported for 2024 anchor; "
        f"got output='{cfg['output']}'"
    )

    emi_df = pd.read_csv(f"{path_csvs}air_emissions_yearly_full.csv")
    emi_df_2024 = emi_df[emi_df["year"] == 2024].copy()

    if emi_df_2024.empty:
        raise ValueError(
            "No rows with year==2024 found in air_emissions_yearly_full.csv. "
            "Make sure the file contains 2024 data."
        )

    sectors = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]
    measure = cfg["measure"]  # e.g. "KG_HAB"
    emission_type = cfg["emission_type"]  # e.g. "CO2"
    grouping = cfg["grouping_structure"]

    # ---- aggregate raw (unscaled) sector values per country -----------------
    records = {}
    for geo, grp in emi_df_2024.groupby("geo"):
        if geo not in EU27_COUNTRIES:
            continue
        row_vals = {}
        for sector in sectors:
            activities = grouping[sector]
            total = 0.0
            for activity in activities:
                pattern = f"air_emissions_yearly:{emission_type}:{activity}:{measure}"
                cols = [c for c in grp.columns if c == pattern]
                if cols:
                    total += grp[cols].values.sum()
            row_vals[sector] = total
        records[geo] = row_vals

    if not records:
        raise ValueError(
            "After filtering, no EU27 countries have 2024 emission data. "
            "Check the 'geo' codes in the CSV."
        )

    # ---- apply training-set scaling parameters --------------------------------
    # The emission columns are stored in dataset.emission_columns in the same
    # sector order used above.  The scaling params key is just the sector name.
    scaled = {}
    for geo, row_vals in records.items():
        tensor_vals = []
        for sector in dataset.emission_columns:
            # emission_columns may be plain sector names ("HeatingCooling") or
            # include the measure suffix ("HeatingCooling_KG_HAB") depending on
            # whether measure=="both".  Handle both cases.
            raw_sector = sector.split("_")[0] if "_" in sector else sector
            if raw_sector not in row_vals:
                raise KeyError(
                    f"Sector '{raw_sector}' not found in aggregated 2024 data "
                    f"for country '{geo}'. emission_columns={dataset.emission_columns}"
                )
            raw_val = row_vals[raw_sector]

            params = dataset.precomputed_scaling_params.get(sector)
            if params is None:
                raise KeyError(
                    f"No scaling params found for emission column '{sector}'. "
                    "Make sure you are passing the same dataset used for training."
                )

            if dataset.scaling_type == "normalization":
                scaled_val = (raw_val - params["mean"]) / params["std"]
            elif dataset.scaling_type == "maxmin":
                scaled_val = (raw_val - params["min"]) / (params["max"] - params["min"])
            else:
                raise ValueError(f"Unknown scaling_type: {dataset.scaling_type}")

            tensor_vals.append(scaled_val)

        scaled[geo] = torch.tensor(tensor_vals, dtype=torch.float32)

    missing = set(EU27_COUNTRIES) - set(scaled.keys())
    if missing:
        print(
            f"[WARNING] load_2024_emissions: no 2024 data for {sorted(missing)}. "
            "These countries will fall back to the 2023 anchor."
        )

    return scaled


# =============================================================================
# Model loading  (identical to original)
# =============================================================================


def set_eval_mode(model: torch.nn.Module) -> None:
    model.eval()


def load_models(dataset, vae_cfg, pred_cfg, fcast_cfg, vae_path, pred_path, fcast_path):
    vae_config = load_config(vae_cfg)
    pred_config = load_config(pred_cfg)
    fcast_config = load_config(fcast_cfg)

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
    vae_model.load_state_dict(torch.load(vae_path, map_location="cpu"))
    set_eval_mode(vae_model)

    predictor_input_dim = 2 * (latent_dim + context_dim)
    predictor = EmissionPredictor(
        input_dim=predictor_input_dim,
        output_configs=output_configs,
        num_blocks=pred_config.pred_num_blocks,
        dim_block=pred_config.pred_dim_block,
        width_block=pred_config.pred_width_block,
        activation=pred_config.pred_activation,
        normalization=pred_config.pred_normalization,
        dropout=pred_config.pred_dropouts,
        uncertainty=True,
    ).to(DEVICE)
    full_pred_model = FullPredictionModel(vae=vae_model, predictor=predictor)
    full_pred_model.load_state_dict(torch.load(pred_path, map_location="cpu"))
    set_eval_mode(full_pred_model)

    forecaster = LatentForecaster(
        input_dim=predictor_input_dim,
        latent_dim=latent_dim,
        num_blocks=fcast_config.forecast_num_blocks,
        dim_block=fcast_config.forecast_dim_block,
        width_block=fcast_config.forecast_width_block,
        activation=fcast_config.forecast_activation,
        normalization=fcast_config.forecast_normalization,
        dropout=fcast_config.forecast_dropouts,
    ).to(DEVICE)
    full_fcast_model = FullLatentForecastingModel(vae=vae_model, forecaster=forecaster)
    full_fcast_model.load_state_dict(torch.load(fcast_path, map_location="cpu"))
    set_eval_mode(full_fcast_model)

    return vae_model, predictor, forecaster, latent_dim


# =============================================================================
# Projection logic
# =============================================================================


def project_country(
    country: str,
    mc_sample: int,
    dataset: DatasetUnified,
    projection_dataset: DatasetProjections2030,
    vae_model: VAEModel,
    predictor: EmissionPredictor,
    forecaster: LatentForecaster,
    latent_dim: int,
    emissions_2024: dict[str, torch.Tensor],  # NEW: observed 2024 anchor
) -> list[list]:
    """
    Project emissions for one country across 2024-2030.

    For year==2024: `emissions_prev` is replaced by the observed 2024
    value (if available), anchoring the entire autoregressive chain on
    the latest real datum.  All other logic is identical to the original.
    """
    results = []

    # ---- initialise latent states from 2022-2023 historical data ------------
    idx_2023 = dataset.index_map.get((country, 2023))
    input_2023 = dataset.input_df[idx_2023].unsqueeze(0).to(DEVICE)
    mean_2023, log_var_2023 = vae_model.encoder(input_2023)
    latent_prev = reparameterize(mean_2023, torch.exp(0.5 * log_var_2023))

    idx_2022 = dataset.index_map.get((country, 2022))
    input_2022 = dataset.input_df[idx_2022].unsqueeze(0).to(DEVICE)
    mean_2022, log_var_2022 = vae_model.encoder(input_2022)
    # (sample discarded — only needed for avg_log_var)

    avg_log_var = (log_var_2023 + log_var_2022) / 2

    # Default emission baseline: 2023 (same as original script)
    emissions_prev = dataset.emi_df[idx_2023, :].unsqueeze(0).to(DEVICE)

    mean_prev = mean_2023
    mean_past = mean_2022

    # ---- autoregressive projection loop -------------------------------------
    for year in PROJECTION_YEARS:
        context_prev, context_current = projection_dataset.get_from_keys_shifted(
            country, year
        )
        context_prev = context_prev.unsqueeze(0).to(DEVICE)
        context_current = context_current.unsqueeze(0).to(DEVICE)

        # Forecast latent mean for this year
        mean_current = forecaster(mean_prev, mean_past, context_current, context_prev)

        # Sample latent with historical variance
        latent_current = reparameterize(mean_current, torch.exp(0.5 * avg_log_var))

        # ---- 2024 anchor: replace emissions_prev with observed 2024 value ---
        if year == 2024 and country in emissions_2024:
            emissions_prev = emissions_2024[country].unsqueeze(0).to(DEVICE)
        # (For all other years, emissions_prev is carried forward from the
        #  previous iteration — same autoregressive logic as original.)

        # Predict emissions
        predictor_input = torch.cat(
            (latent_current, context_current, latent_prev, context_prev), dim=1
        )
        emission_delta, uncertainty = predictor(predictor_input)
        emissions_current = emission_delta + emissions_prev

        row = (
            [mc_sample, country, year]
            + latent_current.squeeze(0).cpu().tolist()
            + emissions_current.squeeze(0).cpu().tolist()
            + uncertainty.squeeze(0).cpu().tolist()
        )
        results.append(row)

        # Update history
        latent_prev = latent_current
        mean_past = mean_prev
        mean_prev = mean_current
        emissions_prev = emissions_current

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

    print("Loading observed 2024 emissions for anchor...")
    emissions_2024 = load_2024_emissions(dataset)
    print(f"  Loaded 2024 anchor for {len(emissions_2024)} countries.")

    print("Loading models...")
    vae_model, predictor, forecaster, latent_dim = load_models(
        dataset=dataset,
        vae_cfg=VAE_CONFIG_PATH,
        pred_cfg=PREDICTOR_CONFIG_PATH,
        fcast_cfg=FORECASTER_CONFIG_PATH,
        vae_path=VAE_MODEL_PATH,
        pred_path=PREDICTOR_MODEL_PATH,
        fcast_path=FORECASTER_MODEL_PATH,
    )

    projection_dataset = DatasetProjections2030(dataset)

    header = (
        ["mc_sample", "geo", "year"]
        + [f"latent_{i}" for i in range(latent_dim)]
        + [f"emissions_{s}" for s in EMISSION_SECTORS]
        + [f"uncertainty_{s}" for s in EMISSION_SECTORS]
    )

    with open(OUTPUT_PATH, mode="w", newline="") as f:
        csv.writer(f).writerow(header)

    print(f"Starting {N_MC_SAMPLES} MC projections (sequential, single GPU)")
    print(f"Output: {OUTPUT_PATH}")

    for chunk_start in range(0, N_MC_SAMPLES, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N_MC_SAMPLES)
        print(f"Processing MC samples {chunk_start} – {chunk_end - 1}...")

        chunk_results = []

        for mc_sample in range(chunk_start, chunk_end):
            random.seed(SEED + mc_sample)
            torch.manual_seed(SEED + mc_sample)
            if DEVICE == "cuda":
                torch.cuda.manual_seed(SEED + mc_sample)

            with torch.no_grad():
                for country in EU27_COUNTRIES:
                    chunk_results.extend(
                        project_country(
                            country=country,
                            mc_sample=mc_sample,
                            dataset=dataset,
                            projection_dataset=projection_dataset,
                            vae_model=vae_model,
                            predictor=predictor,
                            forecaster=forecaster,
                            latent_dim=latent_dim,
                            emissions_2024=emissions_2024,
                        )
                    )

            if (mc_sample + 1) % 100 == 0:
                print(f"  Completed MC sample {mc_sample}")

        with open(OUTPUT_PATH, mode="a", newline="") as f:
            csv.writer(f).writerows(chunk_results)

        print(f"Saved chunk {chunk_start}–{chunk_end - 1}")

    print(f"\nProjections complete! Results saved to {OUTPUT_PATH}")
    total = N_MC_SAMPLES * len(EU27_COUNTRIES) * len(list(PROJECTION_YEARS))
    print(f"Total rows: {total}")


if __name__ == "__main__":
    main()
