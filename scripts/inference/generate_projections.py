"""
Monte Carlo emission projections to 2030.

This script generates Monte Carlo samples of emission projections for EU27
countries from 2024 to 2030. It uses the trained VAE, emission predictor,
and latent forecaster to autoregressively project emissions while capturing
uncertainty through stochastic sampling.

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

Reference:
    Section 2 "Results" in the paper describes the projection methodology.
"""

import csv
import random
from pathlib import Path

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
) -> list[list]:
    """
    Projects emissions for a single country across all projection years.

    Implements autoregressive projection:
    1. Initialize latent states from 2022-2023 historical data
    2. For each year 2024-2030:
       a. Forecast new latent state using forecaster
       b. Predict emissions using predictor
       c. Update latent history for next iteration

    Args:
        country: ISO country code (e.g., 'DE').
        mc_sample: Monte Carlo sample index.
        dataset: Historical dataset with 2022-2023 data.
        projection_dataset: Dataset with projected context variables.
        vae_model: Trained VAE model.
        predictor: Trained emission predictor.
        forecaster: Trained latent forecaster.
        latent_dim: Dimensionality of latent space.

    Returns:
        List of result rows, one per year. Each row contains:
        [mc_sample, country, year, latent_values..., emissions..., uncertainties...]
    """
    results = []

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

        # Predict emissions
        # Input: [z_t, c_t, z_{t-1}, c_{t-1}]
        predictor_input = torch.cat(
            (latent_current, context_current, latent_prev, context_prev), dim=1
        )
        emission_delta, uncertainty = predictor(predictor_input)
        emissions_current = emission_delta + emissions_prev

        # Store result row
        row = (
            [mc_sample, country, year]
            + latent_current.squeeze(0).cpu().tolist()
            + emissions_current.squeeze(0).cpu().tolist()
            + uncertainty.squeeze(0).cpu().tolist()
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
    2. Create CSV with header
    3. Process MC samples sequentially, seeding each deterministically
    4. Write results in chunks to manage memory and allow crash recovery

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
