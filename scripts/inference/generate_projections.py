"""
Monte Carlo emission projections to 2030 with 2024 observed anchor.

This script generates Monte Carlo samples of emission projections for EU27
countries from 2024 to 2030. It uses the trained VAE, emission predictor,
and latent forecaster to autoregressively project emissions while capturing
uncertainty through stochastic sampling.

2024 anchoring:
    The 2024 row uses **observed** sectoral emissions from
    air_emissions_yearly_full.csv instead of model predictions. The raw 2024
    values are aggregated with the same NACE grouping logic used by
    DatasetUnified._load_emissions() and normalised with the training-set
    scaling parameters stored in the dataset. Because the observed 2024
    baseline is deterministic, all MC samples share the same emission values
    for 2024 and model uncertainty is set to NaN for that year. The latent
    forecasting chain still runs through 2024 (needed to initialise the
    autoregressive loop for 2025+), and latent columns are kept as-is.

The projections use:
    1. VAE encoder to get initial latent states from 2022-2023 historical data
    2. Latent forecaster to project future latent states
    3. Emission predictor to convert latents to emission predictions

Uncertainty is captured through reparameterization sampling in the latent
space before linking latent variables to emissions. Learned uncertainty
estimates from the predictor are also computed and stored.

Execution model:
    Models and data are loaded once onto a single GPU. MC samples are
    processed sequentially, with each sample seeded deterministically
    (SEED + mc_sample) so that sample i always produces identical results
    regardless of chunk size, resume point, or execution environment.
    This avoids CUDA context issues that arise when spawning multiple
    processes that each try to initialise their own GPU context (common
    on HPC clusters with SLURM/PBS resource isolation).

Usage:
    python -m scripts.inference.generate_projections

Outputs:
    - data/projections/mc_projections.csv: Monte Carlo samples with columns:
        [mc_sample, geo, year, latent_0..N, emissions_by_sector, uncertainty_by_sector]

"""

import csv
import math
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
# Configuration
# =============================================================================

# Reproducibility — each MC sample is seeded as (SEED + mc_sample)
SEED = 0

# Monte Carlo settings
N_MC_SAMPLES = 10000
CHUNK_SIZE = 100  # Write results to CSV every CHUNK_SIZE samples

# Paths
OUTPUT_PATH = Path("data/projections/mc_projections.csv")
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")

VAE_CONFIG_PATH = Path("config/models/vae_config.yaml")
PREDICTOR_CONFIG_PATH = Path("config/models/co2_predictor_config.yaml")
FORECASTER_CONFIG_PATH = Path("config/models/latent_forecaster_config.yaml")

VAE_MODEL_PATH = Path("data/pytorch_models/vae_model.pth")
PREDICTOR_MODEL_PATH = Path("data/pytorch_models/predictor_model.pth")
FORECASTER_MODEL_PATH = Path("data/pytorch_models/forecaster_model.pth")

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

# Emission sectors (must match output_configs)
EMISSION_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

# Projection years
PROJECTION_YEARS = range(2024, 2031)  # 2024 to 2030 inclusive

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 2024 Emission Anchor
# =============================================================================


def load_2024_emissions(
    dataset: DatasetUnified,
    path_csvs: str = "data/full_timeseries/",
) -> dict[str, torch.Tensor]:
    """
    Loads observed 2024 sectoral emissions, aggregated and scaled to match
    the training dataset.

    Replicates the aggregation logic in DatasetUnified._load_emissions():
    reads the full emission CSV, filters to year 2024, groups NACE activities
    into sectors following output_configs['grouping_structure'], and applies
    the pre-computed scaling parameters stored in ``dataset`` so that the
    returned tensors live in the same normalised space as dataset.emi_df.

    Args:
        dataset: Trained DatasetUnified instance whose
            ``precomputed_scaling_params`` and ``emission_columns`` define
            the target scaled space.
        path_csvs: Directory containing air_emissions_yearly_full.csv.

    Returns:
        Dict mapping country ISO-2 code to a 1-D float32 tensor of shape
        (n_sectors,) in scaled emission space, on CPU. Countries absent
        from the 2024 data are omitted; the caller falls back to the 2023
        baseline for those countries.

    Raises:
        ValueError: If the CSV contains no 2024 rows or no EU27 countries
            remain after filtering.
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

    # Aggregate raw (unscaled) sector values per country, mirroring the
    # pandas filter(regex=) approach used in DatasetUnified._load_emissions().
    records: dict[str, dict[str, float]] = {}
    for geo, grp in emi_df_2024.groupby("geo"):
        if geo not in EU27_COUNTRIES:
            continue
        row_vals: dict[str, float] = {}
        for sector in sectors:
            activities = grouping[sector]
            total = 0.0
            for activity in activities:
                col_pattern = (
                    f"air_emissions_yearly:{emission_type}:{activity}:{measure}"
                )
                matching = [c for c in grp.columns if c == col_pattern]
                if matching:
                    total += grp[matching].values.sum()
            row_vals[sector] = total
        records[geo] = row_vals

    if not records:
        raise ValueError(
            "After filtering, no EU27 countries have 2024 emission data. "
            "Check the 'geo' codes in the CSV."
        )

    # Apply training-set scaling parameters so that the returned tensors are
    # directly comparable to dataset.emi_df.
    #
    # dataset.emission_columns lists sector names in the canonical order.
    # When measure != "both" (the standard case) these are plain sector
    # names (e.g. "HeatingCooling"); when measure == "both" they carry a
    # suffix (e.g. "HeatingCooling_KG_HAB").  We look up the raw sector
    # name accordingly.
    measures = ["THS_T", "KG_HAB"] if cfg["measure"] == "both" else [cfg["measure"]]
    col_to_raw_sector: dict[str, str] = {}
    for m in measures:
        for s in sectors:
            col_name = f"{s}_{m}" if len(measures) > 1 else s
            col_to_raw_sector[col_name] = s

    scaled: dict[str, torch.Tensor] = {}
    for geo, row_vals in records.items():
        tensor_vals: list[float] = []
        for col in dataset.emission_columns:
            raw_sector = col_to_raw_sector.get(col)
            if raw_sector is None:
                raise KeyError(
                    f"Emission column '{col}' cannot be mapped back to a raw "
                    f"sector. emission_columns={dataset.emission_columns}"
                )

            raw_val = row_vals[raw_sector]

            params = dataset.precomputed_scaling_params.get(col)
            if params is None:
                raise KeyError(
                    f"No scaling params found for emission column '{col}'. "
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
# Model Loading
# =============================================================================


def set_eval_mode(model: torch.nn.Module) -> None:
    """
    Sets model to evaluation mode.

    Note: In previous code iterations, there was an option to keep dropout active
    during inference for MC dropout uncertainty. This is currently disabled
    but can be re-enabled by uncommenting the dropout activation loop.

    Args:
        model: PyTorch model to set to eval mode.
    """
    model.eval()
    # Uncomment below to enable MC dropout (keeps dropout active during inference)
    # for module in model.modules():
    #     if isinstance(module, torch.nn.Dropout):
    #         module.train()


def load_models(
    dataset: DatasetUnified,
    vae_config_path: Path,
    predictor_config_path: Path,
    forecaster_config_path: Path,
    vae_model_path: Path,
    predictor_model_path: Path,
    forecaster_model_path: Path,
) -> tuple[VAEModel, EmissionPredictor, LatentForecaster, int]:
    """
    Loads all trained models for projection onto a single device.

    All models are loaded to CPU first and then moved to DEVICE once,
    avoiding CUDA context issues in multi-process environments.

    Args:
        dataset: Dataset instance (needed for input/context dimensions).
        vae_config_path: Path to VAE config YAML.
        predictor_config_path: Path to predictor config YAML.
        forecaster_config_path: Path to forecaster config YAML.
        vae_model_path: Path to trained VAE weights.
        predictor_model_path: Path to trained predictor weights.
        forecaster_model_path: Path to trained forecaster weights.

    Returns:
        Tuple of (vae_model, predictor, forecaster, latent_dim).
    """
    vae_config = load_config(vae_config_path)
    predictor_config = load_config(predictor_config_path)
    forecaster_config = load_config(forecaster_config_path)

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
    vae_model.load_state_dict(torch.load(vae_model_path, map_location="cpu"))
    set_eval_mode(vae_model)

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

    # Load full prediction model (contains both VAE and predictor weights)
    full_pred_model = FullPredictionModel(vae=vae_model, predictor=predictor)
    full_pred_model.load_state_dict(
        torch.load(predictor_model_path, map_location="cpu")
    )
    set_eval_mode(full_pred_model)

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

    # Load full forecasting model
    full_forecast_model = FullLatentForecastingModel(
        vae=vae_model, forecaster=forecaster
    )
    full_forecast_model.load_state_dict(
        torch.load(forecaster_model_path, map_location="cpu")
    )
    set_eval_mode(full_forecast_model)

    return vae_model, predictor, forecaster, latent_dim


# =============================================================================
# Projection Logic
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
    emissions_2024: dict[str, torch.Tensor],
) -> list[list]:
    """
    Projects emissions for a single country across all projection years.

    Implements autoregressive projection:
    1. Initialize latent states from 2022-2023 historical data
    2. For each year 2024-2030:
       a. Forecast new latent state using forecaster
       b. Predict emissions using predictor
       c. Update latent history for next iteration

    For the 2024 step the model still runs normally (the latent chain must
    be maintained), but the emission output is overridden with observed 2024
    data so that the autoregressive chain from 2025 onward is anchored to
    reality. Model uncertainty is set to NaN for 2024 since the emission
    values are observed, not predicted.

    Args:
        country: ISO country code (e.g., 'DE').
        mc_sample: Monte Carlo sample index.
        dataset: Historical dataset with 2022-2023 data.
        projection_dataset: Dataset with projected context variables.
        vae_model: Trained VAE model.
        predictor: Trained emission predictor.
        forecaster: Trained latent forecaster.
        latent_dim: Dimensionality of latent space.
        emissions_2024: Dict mapping country code to scaled observed 2024
            emissions tensor. Countries not present fall back to the model
            prediction for 2024.

    Returns:
        List of result rows, one per year. Each row contains:
        [mc_sample, country, year, latent_values..., emissions..., uncertainties...]
    """
    results = []
    n_sectors = len(EMISSION_SECTORS)

    # -------------------------------------------------------------------------
    # Initialize from historical data (2022-2023)
    # -------------------------------------------------------------------------

    # Get 2023 latent state (t-1 for first projection year)
    idx_2023 = dataset.index_map.get((country, 2023))
    input_2023 = dataset.input_df[idx_2023].unsqueeze(0).to(DEVICE)
    mean_2023, log_var_2023 = vae_model.encoder(input_2023)
    latent_prev = reparameterize(mean_2023, torch.exp(0.5 * log_var_2023))

    # Get 2022 latent state (t-2 for first projection year)
    idx_2022 = dataset.index_map.get((country, 2022))
    input_2022 = dataset.input_df[idx_2022].unsqueeze(0).to(DEVICE)
    mean_2022, log_var_2022 = vae_model.encoder(input_2022)
    reparameterize(mean_2022, torch.exp(0.5 * log_var_2022))

    # Average historical latent variance for sampling
    avg_log_var = (log_var_2023 + log_var_2022) / 2

    # Initialize emission history from 2023
    emissions_prev = dataset.emi_df[idx_2023, :].unsqueeze(0).to(DEVICE)

    # Track means for forecaster (which uses means, not samples)
    mean_prev = mean_2023
    mean_past = mean_2022

    # -------------------------------------------------------------------------
    # Autoregressive projection loop
    # -------------------------------------------------------------------------

    for year in PROJECTION_YEARS:
        # Get projected context for current and previous year
        context_prev, context_current = projection_dataset.get_from_keys_shifted(
            country, year
        )
        context_prev = context_prev.unsqueeze(0).to(DEVICE)
        context_current = context_current.unsqueeze(0).to(DEVICE)

        # Forecast latent mean for current year
        mean_current = forecaster(mean_prev, mean_past, context_current, context_prev)

        # Sample latent state using historical variance
        latent_current = reparameterize(mean_current, torch.exp(0.5 * avg_log_var))

        # Predict emissions (runs even for 2024 to keep the code path uniform;
        # the predictor output is discarded for 2024 when observed data exists)
        predictor_input = torch.cat(
            (latent_current, context_current, latent_prev, context_prev), dim=1
        )
        emission_delta, uncertainty = predictor(predictor_input)
        emissions_current = emission_delta + emissions_prev

        # For 2024: override with observed emissions and mark uncertainty as NaN
        if year == 2024 and country in emissions_2024:
            emissions_current = emissions_2024[country].unsqueeze(0).to(DEVICE)
            uncertainty_vals = [math.nan] * n_sectors
        else:
            uncertainty_vals = uncertainty.squeeze(0).cpu().tolist()

        # Store result row
        row = (
            [mc_sample, country, year]
            + latent_current.squeeze(0).cpu().tolist()
            + emissions_current.squeeze(0).cpu().tolist()
            + uncertainty_vals
        )
        results.append(row)

        # Update history for next iteration
        latent_prev = latent_current
        mean_past = mean_prev
        mean_prev = mean_current
        emissions_prev = emissions_current

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """
    Main function to run Monte Carlo projections.

    Workflow:
    1. Load dataset and all models once onto a single GPU
    2. Load observed 2024 emissions and scale to training space
    3. Create CSV with header
    4. Process MC samples sequentially, seeding each deterministically
    5. Write results in chunks to manage memory and allow crash recovery

    Each MC sample is seeded with (SEED + mc_sample) before its forward
    passes, ensuring that sample i always produces identical results
    regardless of chunk boundaries, resume point, or total sample count.
    """
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading dataset from {DATASET_PATH}...")

    # Load dataset once — keep tensors on CPU, move per-sample to GPU
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
        vae_config_path=VAE_CONFIG_PATH,
        predictor_config_path=PREDICTOR_CONFIG_PATH,
        forecaster_config_path=FORECASTER_CONFIG_PATH,
        vae_model_path=VAE_MODEL_PATH,
        predictor_model_path=PREDICTOR_MODEL_PATH,
        forecaster_model_path=FORECASTER_MODEL_PATH,
    )

    projection_dataset = DatasetProjections2030(dataset)

    # Create CSV header
    header = (
        ["mc_sample", "geo", "year"]
        + [f"latent_{i}" for i in range(latent_dim)]
        + [f"emissions_{sector}" for sector in EMISSION_SECTORS]
        + [f"uncertainty_{sector}" for sector in EMISSION_SECTORS]
    )

    # Write header to CSV
    with open(OUTPUT_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print(f"Starting {N_MC_SAMPLES} Monte Carlo projections (sequential, single GPU)")
    print(f"Output: {OUTPUT_PATH}")

    # Process MC samples sequentially in chunks
    for chunk_start in range(0, N_MC_SAMPLES, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N_MC_SAMPLES)
        print(f"Processing MC samples {chunk_start} to {chunk_end - 1}...")

        chunk_results = []

        for mc_sample in range(chunk_start, chunk_end):
            # Deterministic per-sample seeding: sample i always gets the
            # same RNG state, making results reproducible regardless of
            # chunk boundaries or resume point.
            random.seed(SEED + mc_sample)
            torch.manual_seed(SEED + mc_sample)
            if DEVICE == "cuda":
                torch.cuda.manual_seed(SEED + mc_sample)

            with torch.no_grad():
                for country in EU27_COUNTRIES:
                    country_results = project_country(
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
                    chunk_results.extend(country_results)

            if (mc_sample + 1) % 100 == 0:
                print(f"  Completed MC sample {mc_sample}")

        # Write chunk results to CSV
        with open(OUTPUT_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(chunk_results)

        print(f"Saved chunk {chunk_start}-{chunk_end - 1}")

    print(f"\nProjections complete! Results saved to {OUTPUT_PATH}")
    print(
        f"Total rows: {N_MC_SAMPLES * len(EU27_COUNTRIES) * len(list(PROJECTION_YEARS))}"
    )


if __name__ == "__main__":
    main()
