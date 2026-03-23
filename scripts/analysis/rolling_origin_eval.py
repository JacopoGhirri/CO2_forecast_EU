"""
Rolling-Origin Evaluation of the Full Emission Projection Pipeline.

For each training cutoff T in {2018, 2019, 2020, 2021}, trains the VAE,
predictor, and forecaster on data up to year T, then autoregressively
projects emissions for T+1, T+2, T+3 and compares against observed values.

This provides a genuine out-of-sample test of the forecasting pipeline
across multiple time horizons, complementing the random-split validation
reported in Table S1.

Note: this script re-trains all three models from scratch for each cutoff,
which is computationally expensive. On a single GPU expect roughly
[training_time * 4] total runtime. Results are written incrementally so
the script can be interrupted and restarted.

Usage:
    python -m scripts.analysis.rolling_origin_eval

Outputs:
    outputs/tables/SI_rolling_origin_results.csv
    outputs/tables/SI_rolling_origin_summary.csv
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config.data.output_configs import output_configs
from scripts.elements.datasets import (
    DatasetForecasting,
    DatasetPrediction,
    DatasetProjections2030,
    DatasetUnified,
)
from scripts.elements.models import (
    Decoder,
    EmissionPredictor,
    Encoder,
    FullLatentForecastingModel,
    FullPredictionModel,
    LatentForecaster,
    VAEModel,
    reparameterize,
    uncertainty_aware_mae_loss,
    uncertainty_aware_mse_loss,
    vae_loss,
)
from scripts.utils import (
    check_nan_gradients,
    count_parameters,
    init_weights,
    load_config,
)

# =============================================================================
# Configuration
# =============================================================================

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training cutoffs to evaluate
CUTOFF_YEARS = [2018, 2019, 2020, 2021]

# How many years ahead to evaluate
MAX_HORIZON = 3

# Data always starts from 2010
DATA_START_YEAR = 2010

# Reduced epochs for rolling eval (full training is expensive x4 cutoffs)
# Set to same as main training if compute allows
VAE_EPOCHS = 5000
PRED_EPOCHS = 5000
FCAST_EPOCHS = 5000
BATCH_SIZE = 128

VARIABLE_FILE = Path("config/data/variable_selection.txt")
VAE_CONFIG_PATH = Path("config/models/vae_config.yaml")
PREDICTOR_CONFIG_PATH = Path("config/models/co2_predictor_config.yaml")
FORECASTER_CONFIG_PATH = Path("config/models/latent_forecaster_config.yaml")

OUTPUT_DIR = Path("outputs/tables")

EU27_COUNTRIES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "EL", "FI",
    "FR", "DE", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
]

EMISSION_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]


# =============================================================================
# Training helpers
# =============================================================================

def train_vae(dataset, vae_cfg, n_epochs: int) -> VAEModel:
    input_dim = len(dataset.input_variable_names)

    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=vae_cfg.vae_latent_dim,
        num_blocks=vae_cfg.vae_num_blocks,
        dim_blocks=vae_cfg.vae_dim_blocks,
        activation=vae_cfg.vae_activation,
        normalization=vae_cfg.vae_normalization,
        dropout=vae_cfg.vae_dropouts,
        input_dropout=vae_cfg.vae_input_dropouts,
    ).to(DEVICE)

    decoder = Decoder(
        input_dim=input_dim,
        latent_dim=vae_cfg.vae_latent_dim,
        num_blocks=vae_cfg.vae_num_blocks,
        dim_blocks=vae_cfg.vae_dim_blocks,
        activation=vae_cfg.vae_activation,
        normalization=vae_cfg.vae_normalization,
        dropout=vae_cfg.vae_dropouts,
    ).to(DEVICE)

    model = VAEModel(encoder, decoder).to(DEVICE)
    model.apply(init_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=vae_cfg.vae_lr,
        weight_decay=vae_cfg.vae_weight_decay,
        eps=1e-6,
    )

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [0.85, 0.15],
        generator=torch.Generator().manual_seed(SEED)
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_recon = None
    best_weights = None
    recon_history = deque(maxlen=10)

    for epoch in range(n_epochs):
        model.train()
        for batch in loader:
            x_cur, _, _, _, _ = batch
            x_cur = x_cur.to(DEVICE)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x_cur)
            recon, kl = vae_loss(x_cur, x_hat, mean, log_var)
            loss = vae_cfg.vae_wr * recon + vae_cfg.vae_wd * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            check_nan_gradients(model)
            optimizer.step()

        # Validation reconstruction for model selection
        model.eval()
        val_recon = []
        with torch.inference_mode():
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
            for batch in val_loader:
                x_cur, _, _, _, _ = batch
                x_cur = x_cur.to(DEVICE)
                x_hat, mean, log_var = model(x_cur)
                recon, _ = vae_loss(x_cur, x_hat, mean, log_var)
                val_recon.append(recon.item())

        recon_history.append(np.mean(val_recon))
        smooth = np.mean(recon_history)
        if best_recon is None or smooth < best_recon:
            best_recon = smooth
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 500 == 0:
            print(f"    VAE epoch {epoch}: recon={np.mean(val_recon):.4f}")

    model.load_state_dict(best_weights)
    return model


def train_predictor(dataset, vae_model, pred_cfg, vae_cfg, n_epochs: int) -> FullPredictionModel:
    from scripts.elements.datasets import DatasetPrediction
    pred_dataset = DatasetPrediction.__new__(DatasetPrediction)
    pred_dataset.__dict__.update(dataset.__dict__)
    pred_dataset.__class__ = DatasetPrediction

    input_dim = len(dataset.input_variable_names)
    context_dim = len(dataset.context_variable_names)
    latent_dim = vae_cfg.vae_latent_dim
    predictor_input_dim = 2 * (latent_dim + context_dim)

    predictor = EmissionPredictor(
        input_dim=predictor_input_dim,
        output_configs=output_configs,
        num_blocks=pred_cfg.pred_num_blocks,
        dim_block=pred_cfg.pred_dim_block,
        width_block=pred_cfg.pred_width_block,
        activation=pred_cfg.pred_activation,
        normalization=pred_cfg.pred_normalization,
        dropout=pred_cfg.pred_dropouts,
        uncertainty=True,
    ).to(DEVICE)
    predictor.apply(init_weights)

    full_model = FullPredictionModel(vae=vae_model, predictor=predictor)

    for p in full_model.encoder.parameters():
        p.requires_grad = False
    for p in full_model.decoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        full_model.predictor.parameters(),
        lr=pred_cfg.pred_lr,
        weight_decay=pred_cfg.pred_wd,
        eps=1e-6,
    )

    train_ds, val_ds = torch.utils.data.random_split(
        pred_dataset, [0.85, 0.15],
        generator=torch.Generator().manual_seed(SEED)
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_loss = None
    best_weights = None
    loss_history = deque(maxlen=50)

    for epoch in range(n_epochs):
        full_model.train()
        for batch in loader:
            x_cur, c_cur, y_cur, x_prev, c_prev, y_prev = [
                b.to(DEVICE) for b in batch
            ]
            optimizer.zero_grad()
            delta_pred, unc, _, _, _, _, _, _ = full_model(x_cur, x_prev, c_cur, c_prev)
            delta_true = y_cur - y_prev
            loss = uncertainty_aware_mse_loss(delta_true, delta_pred, unc, mode=pred_cfg.mode_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
            check_nan_gradients(full_model)
            optimizer.step()

        full_model.eval()
        val_losses = []
        with torch.inference_mode():
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
            for batch in val_loader:
                x_cur, c_cur, y_cur, x_prev, c_prev, y_prev = [
                    b.to(DEVICE) for b in batch
                ]
                delta_pred, unc, _, _, _, _, _, _ = full_model(x_cur, x_prev, c_cur, c_prev)
                delta_true = y_cur - y_prev
                loss = uncertainty_aware_mse_loss(delta_true, delta_pred, unc, mode=pred_cfg.mode_loss)
                val_losses.append(loss.item())

        loss_history.append(np.mean(val_losses))
        smooth = np.mean(loss_history)
        if best_loss is None or smooth < best_loss:
            best_loss = smooth
            best_weights = {k: v.clone() for k, v in full_model.state_dict().items()}

        if epoch % 500 == 0:
            print(f"    Predictor epoch {epoch}: val_loss={np.mean(val_losses):.4f}")

    full_model.load_state_dict(best_weights)
    return full_model


def train_forecaster(dataset, vae_model, fcast_cfg, vae_cfg, n_epochs: int) -> FullLatentForecastingModel:
    fcast_dataset = DatasetForecasting(dataset)

    context_dim = len(dataset.context_variable_names)
    latent_dim = vae_cfg.vae_latent_dim
    forecaster_input_dim = 2 * (latent_dim + context_dim)

    forecaster = LatentForecaster(
        input_dim=forecaster_input_dim,
        latent_dim=latent_dim,
        num_blocks=fcast_cfg.forecast_num_blocks,
        dim_block=fcast_cfg.forecast_dim_block,
        width_block=fcast_cfg.forecast_width_block,
        activation=fcast_cfg.forecast_activation,
        normalization=fcast_cfg.forecast_normalization,
        dropout=fcast_cfg.forecast_dropouts,
    ).to(DEVICE)
    forecaster.apply(init_weights)

    full_model = FullLatentForecastingModel(vae=vae_model, forecaster=forecaster)

    for p in full_model.encoder.parameters():
        p.requires_grad = False
    for p in full_model.vae.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        full_model.forecaster.parameters(),
        lr=fcast_cfg.forecast_lr,
        weight_decay=fcast_cfg.forecast_wd,
        eps=1e-6,
    )

    base_indices = list(range(fcast_dataset.base_length))
    rng = random.Random(SEED)
    rng.shuffle(base_indices)
    n_val = int(fcast_dataset.base_length * 0.15)
    val_indices = base_indices[:n_val]
    train_indices = base_indices[n_val:]

    train_subset = torch.utils.data.Subset(fcast_dataset, train_indices)
    val_subset = torch.utils.data.Subset(fcast_dataset, val_indices)
    loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    best_acc = None
    best_weights = None
    acc_history = deque(maxlen=50)

    for epoch in range(n_epochs):
        full_model.train()
        for batch in loader:
            x_cur, c_cur, x_prev, c_prev, x_past, c_past = [
                b.to(DEVICE) for b in batch
            ]
            optimizer.zero_grad()
            z_forecast = full_model(x_prev, x_past, c_cur, c_prev)
            mean_target, log_var_target = full_model.encoder(x_cur)
            loss = uncertainty_aware_mae_loss(mean_target, z_forecast, log_var_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
            check_nan_gradients(full_model)
            optimizer.step()

        full_model.eval()
        val_accs = []
        with torch.inference_mode():
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
            for batch in val_loader:
                x_cur, c_cur, x_prev, c_prev, x_past, c_past = [
                    b.to(DEVICE) for b in batch
                ]
                z_forecast = full_model(x_prev, x_past, c_cur, c_prev)
                mean_target, _ = full_model.encoder(x_cur)
                val_accs.append((mean_target - z_forecast).pow(2).mean().item())

        acc_history.append(np.mean(val_accs))
        smooth = np.mean(acc_history)
        if best_acc is None or smooth < best_acc:
            best_acc = smooth
            best_weights = {k: v.clone() for k, v in full_model.state_dict().items()}

        if epoch % 500 == 0:
            print(f"    Forecaster epoch {epoch}: val_acc={np.mean(val_accs):.6f}")

    full_model.load_state_dict(best_weights)
    return full_model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_projections(
    cutoff_year: int,
    full_pred_model: FullPredictionModel,
    full_fcast_model: FullLatentForecastingModel,
    train_dataset: DatasetUnified,
    full_dataset: DatasetUnified,
) -> list[dict]:
    """
    Autoregressively project emissions from cutoff_year+1 to
    cutoff_year+MAX_HORIZON and compare against observed values
    from full_dataset. Returns one row per (country, horizon, sector).
    """
    full_pred_model.eval()
    full_fcast_model.eval()

    results = []

    for country in EU27_COUNTRIES:
        # Check we have the anchor years in the training dataset
        idx_t0 = train_dataset.index_map.get((country, cutoff_year))
        idx_tm1 = train_dataset.index_map.get((country, cutoff_year - 1))
        if idx_t0 is None or idx_tm1 is None:
            continue

        with torch.inference_mode():
            # Initialise latent chain from last two observed years
            x_t0 = train_dataset.input_df[idx_t0].unsqueeze(0).to(DEVICE)
            x_tm1 = train_dataset.input_df[idx_tm1].unsqueeze(0).to(DEVICE)

            mean_t0, log_var_t0 = full_fcast_model.encoder(x_t0)
            mean_tm1, log_var_tm1 = full_fcast_model.encoder(x_tm1)

            mean_prev = mean_t0
            mean_past = mean_tm1
            avg_log_var = (log_var_t0 + log_var_tm1) / 2

            # Emission anchor at cutoff year
            emissions_prev = train_dataset.emi_df[idx_t0].unsqueeze(0).to(DEVICE)

            for horizon in range(1, MAX_HORIZON + 1):
                target_year = cutoff_year + horizon

                # Check observed data exists in the full dataset
                idx_obs = full_dataset.index_map.get((country, target_year))
                if idx_obs is None:
                    continue

                # Context: use observed context from full dataset for eval
                # (in a real forecast we would use projected context, but
                # for evaluation we use observed to isolate model error)
                c_cur = full_dataset.context_df[idx_obs].unsqueeze(0).to(DEVICE)
                idx_prev_ctx = full_dataset.index_map.get((country, target_year - 1))
                c_prev = (
                    full_dataset.context_df[idx_prev_ctx].unsqueeze(0).to(DEVICE)
                    if idx_prev_ctx is not None
                    else c_cur
                )

                # Forecast latent
                mean_cur = full_fcast_model.forecaster(
                    mean_prev, mean_past, c_cur, c_prev
                )
                latent_cur = reparameterize(
                    mean_cur, torch.exp(0.5 * avg_log_var)
                )
                latent_prev_sample = reparameterize(
                    mean_prev, torch.exp(0.5 * avg_log_var)
                )

                # Predict emissions
                pred_input = torch.cat(
                    (latent_cur, c_cur, latent_prev_sample, c_prev), dim=1
                )
                delta, _ = full_pred_model.predictor(pred_input)
                emissions_cur = delta + emissions_prev

                # Observed emissions from full dataset
                y_obs = full_dataset.emi_df[idx_obs].unsqueeze(0).to(DEVICE)

                # Compute per-sector errors in scaled space
                residuals = (emissions_cur - y_obs).squeeze(0).cpu().numpy()

                for s_idx, sector in enumerate(EMISSION_SECTORS):
                    results.append({
                        "cutoff_year": cutoff_year,
                        "country": country,
                        "target_year": target_year,
                        "horizon": horizon,
                        "sector": sector,
                        "predicted": float(emissions_cur[0, s_idx].cpu()),
                        "observed": float(y_obs[0, s_idx].cpu()),
                        "residual": float(residuals[s_idx]),
                        "abs_error": float(abs(residuals[s_idx])),
                        "sq_error": float(residuals[s_idx]**2),
                    })

                # Update chain
                mean_past = mean_prev
                mean_prev = mean_cur
                emissions_prev = emissions_cur

    return results


# =============================================================================
# Summary tables
# =============================================================================

def build_summary_tables(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two summary tables:
    1. RMSE by cutoff year and horizon (aggregate across sectors and countries)
    2. RMSE by horizon and sector (averaged across cutoff years)
    """
    # Table 1: cutoff x horizon
    agg = (
        results_df
        .groupby(["cutoff_year", "horizon"])
        .agg(
            MSE=("sq_error", "mean"),
            MAE=("abs_error", "mean"),
            n=("sq_error", "count"),
        )
        .reset_index()
    )
    agg["RMSE"] = np.sqrt(agg["MSE"])

    pivot_rmse = agg.pivot(index="cutoff_year", columns="horizon", values="RMSE")
    pivot_rmse.columns = [f"Horizon +{h}" for h in pivot_rmse.columns]
    pivot_rmse.index.name = "Training cutoff"

    # Add mean row
    pivot_rmse.loc["Mean"] = pivot_rmse.mean()

    # Table 2: sector x horizon (averaged across cutoffs)
    sector_agg = (
        results_df
        .groupby(["sector", "horizon"])
        .agg(MSE=("sq_error", "mean"), MAE=("abs_error", "mean"))
        .reset_index()
    )
    sector_agg["RMSE"] = np.sqrt(sector_agg["MSE"])

    pivot_sector = sector_agg.pivot(index="sector", columns="horizon", values="RMSE")
    pivot_sector.columns = [f"Horizon +{h}" for h in pivot_sector.columns]
    pivot_sector.index.name = "Sector"

    return pivot_rmse, pivot_sector


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ROLLING-ORIGIN EVALUATION")
    print(f"Device: {DEVICE}")
    print(f"Cutoff years: {CUTOFF_YEARS}")
    print(f"Max horizon: {MAX_HORIZON}")
    print("=" * 60)

    # Load configs
    vae_cfg = load_config(VAE_CONFIG_PATH)
    pred_cfg = load_config(PREDICTOR_CONFIG_PATH)
    fcast_cfg = load_config(FORECASTER_CONFIG_PATH)

    # Load variable selection
    with open(VARIABLE_FILE) as f:
        nested_variables = [line.strip() for line in f if line.strip()]

    # Load the full dataset (2010-2023) once — used for observed values
    print("\nLoading full dataset (2010-2023)...")
    full_dataset = DatasetUnified(
        path_csvs="data/full_timeseries/",
        output_configs=output_configs,
        select_years=np.arange(DATA_START_YEAR, 2024),
        select_geo=EU27_COUNTRIES,
        nested_variables=nested_variables,
        with_cuda=True,
        scaling_type="normalization",
    )
    print(f"  Full dataset: {len(full_dataset)} samples")

    all_results = []
    results_path = OUTPUT_DIR / "SI_rolling_origin_results.csv"

    for cutoff_year in CUTOFF_YEARS:
        print(f"\n{'=' * 60}")
        print(f"CUTOFF YEAR: {cutoff_year}")
        print(f"  Training on {DATA_START_YEAR}–{cutoff_year}")
        print(f"  Evaluating on {cutoff_year+1}–{cutoff_year+MAX_HORIZON}")
        print("=" * 60)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # Build training dataset up to cutoff
        print("  Building training dataset...")
        train_dataset = DatasetUnified(
            path_csvs="data/full_timeseries/",
            output_configs=output_configs,
            select_years=np.arange(DATA_START_YEAR, cutoff_year + 1),
            select_geo=EU27_COUNTRIES,
            nested_variables=nested_variables,
            with_cuda=True,
            scaling_type="normalization",
        )
        print(f"  Training dataset: {len(train_dataset)} samples")

        # Train all three models
        print("\n  [1/3] Training VAE...")
        vae_model = train_vae(train_dataset, vae_cfg, VAE_EPOCHS)

        print("\n  [2/3] Training Predictor...")
        full_pred = train_predictor(
            train_dataset, vae_model, pred_cfg, vae_cfg, PRED_EPOCHS
        )

        print("\n  [3/3] Training Forecaster...")
        full_fcast = train_forecaster(
            train_dataset, vae_model, fcast_cfg, vae_cfg, FCAST_EPOCHS
        )

        # Evaluate on held-out years
        print("\n  Evaluating projections...")
        results = evaluate_projections(
            cutoff_year=cutoff_year,
            full_pred_model=full_pred,
            full_fcast_model=full_fcast,
            train_dataset=train_dataset,
            full_dataset=full_dataset,
        )
        all_results.extend(results)

        # Save incrementally so we don't lose progress on interruption
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        print(f"  Saved {len(results)} rows (running total: {len(all_results)})")

        # Print quick summary for this cutoff
        cutoff_df = pd.DataFrame(results)
        for h in range(1, MAX_HORIZON + 1):
            h_df = cutoff_df[cutoff_df["horizon"] == h]
            rmse = np.sqrt(h_df["sq_error"].mean())
            mae = h_df["abs_error"].mean()
            print(f"  Horizon +{h}: RMSE={rmse:.4f}  MAE={mae:.4f}  (n={len(h_df)})")

        # Free GPU memory before next cutoff
        del vae_model, full_pred, full_fcast, train_dataset
        torch.cuda.empty_cache()

    # Build and save summary tables
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(all_results)
    pivot_cutoff, pivot_sector = build_summary_tables(results_df)

    summary_path = OUTPUT_DIR / "SI_rolling_origin_summary.csv"
    pivot_cutoff.to_csv(summary_path)
    print(f"\nSaved summary: {summary_path}")

    sector_path = OUTPUT_DIR / "SI_rolling_origin_by_sector.csv"
    pivot_sector.to_csv(sector_path)
    print(f"Saved sector breakdown: {sector_path}")

    print("\nTable S2: RMSE by cutoff year and horizon")
    print(pivot_cutoff.round(4).to_string())

    print("\nTable S3: RMSE by sector and horizon (averaged across cutoffs)")
    print(pivot_sector.round(4).to_string())


if __name__ == "__main__":
    import numpy as np
    main()