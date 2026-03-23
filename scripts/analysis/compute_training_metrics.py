"""
Training Metrics Summary for Supplementary Information.

Loads the trained models and dataset, then computes and reports:
  - VAE: reconstruction loss, KL divergence, latent space statistics
  - Predictor: emission-space MSE, MAE, R² per sector (train and val)
  - Forecaster: latent-space MSE (train and val)

All metrics are computed on held-out validation splits (85/15, same seed
as training) to ensure comparability with reported training curves.

Usage:
    python -m scripts.analysis.compute_training_metrics

Outputs:
    outputs/tables/SI_training_metrics.csv
    outputs/tables/SI_training_metrics_per_sector.csv
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config.data.output_configs import output_configs
from scripts.elements.datasets import DatasetForecasting, DatasetPrediction, DatasetUnified
from scripts.elements.models import (
    Decoder,
    EmissionPredictor,
    Encoder,
    FullLatentForecastingModel,
    FullPredictionModel,
    LatentForecaster,
    VAEModel,
    reparameterize,
    vae_loss,
)
from scripts.utils import load_config, load_dataset

# =============================================================================
# Configuration
# =============================================================================

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
VAL_SPLIT = 0.15

DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
VAE_CONFIG_PATH = Path("config/models/vae_config.yaml")
PREDICTOR_CONFIG_PATH = Path("config/models/co2_predictor_config.yaml")
FORECASTER_CONFIG_PATH = Path("config/models/latent_forecaster_config.yaml")
VAE_MODEL_PATH = Path("data/pytorch_models/vae_model.pth")
PREDICTOR_MODEL_PATH = Path("data/pytorch_models/predictor_model.pth")
FORECASTER_MODEL_PATH = Path("data/pytorch_models/forecaster_model.pth")
VARIABLE_FILE = Path("config/data/variable_selection.txt")

OUTPUT_DIR = Path("outputs/tables")

EU27_COUNTRIES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "EL", "FI",
    "FR", "DE", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
]

EMISSION_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]


# =============================================================================
# Model loading
# =============================================================================

def load_all_models(dataset):
    vae_cfg = load_config(VAE_CONFIG_PATH)
    pred_cfg = load_config(PREDICTOR_CONFIG_PATH)
    fcast_cfg = load_config(FORECASTER_CONFIG_PATH)

    input_dim = len(dataset.input_variable_names)
    context_dim = len(dataset.context_variable_names)
    latent_dim = vae_cfg.vae_latent_dim

    # VAE
    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_blocks=vae_cfg.vae_num_blocks,
        dim_blocks=vae_cfg.vae_dim_blocks,
        activation=vae_cfg.vae_activation,
        normalization=vae_cfg.vae_normalization,
        dropout=vae_cfg.vae_dropouts,
        input_dropout=vae_cfg.vae_input_dropouts,
    )
    decoder = Decoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_blocks=vae_cfg.vae_num_blocks,
        dim_blocks=vae_cfg.vae_dim_blocks,
        activation=vae_cfg.vae_activation,
        normalization=vae_cfg.vae_normalization,
        dropout=vae_cfg.vae_dropouts,
    )
    vae_model = VAEModel(encoder, decoder).to(DEVICE)
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    vae_model.eval()

    # Predictor
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
    full_pred = FullPredictionModel(vae=vae_model, predictor=predictor)
    full_pred.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE))
    full_pred.eval()

    # Forecaster
    forecaster = LatentForecaster(
        input_dim=predictor_input_dim,
        latent_dim=latent_dim,
        num_blocks=fcast_cfg.forecast_num_blocks,
        dim_block=fcast_cfg.forecast_dim_block,
        width_block=fcast_cfg.forecast_width_block,
        activation=fcast_cfg.forecast_activation,
        normalization=fcast_cfg.forecast_normalization,
        dropout=fcast_cfg.forecast_dropouts,
    ).to(DEVICE)
    full_fcast = FullLatentForecastingModel(vae=vae_model, forecaster=forecaster)
    full_fcast.load_state_dict(torch.load(FORECASTER_MODEL_PATH, map_location=DEVICE))
    full_fcast.eval()

    return vae_model, full_pred, full_fcast, latent_dim


# =============================================================================
# VAE metrics
# =============================================================================

def compute_vae_metrics(vae_model, dataset):
    """
    Reconstruction loss, KL divergence, and latent space statistics
    on the held-out validation split.
    """
    print("\n" + "=" * 60)
    print("VAE METRICS")
    print("=" * 60)

    generator = torch.Generator().manual_seed(SEED)
    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    results = {}
    for split_name, split_ds in [("train", train_ds), ("val", val_ds)]:
        loader = DataLoader(split_ds, batch_size=BATCH_SIZE, shuffle=False)

        recon_losses, kl_losses = [], []
        all_means, all_log_vars = [], []

        with torch.inference_mode():
            for batch in loader:
                x_cur, _, _, _, _ = batch
                x_cur = x_cur.to(DEVICE)

                x_hat, mean, log_var = vae_model(x_cur)
                recon, kl = vae_loss(x_cur, x_hat, mean, log_var)

                recon_losses.append(recon.item())
                kl_losses.append(kl.item())
                all_means.append(mean.cpu())
                all_log_vars.append(log_var.cpu())

        all_means = torch.cat(all_means, dim=0).numpy()
        all_log_vars = torch.cat(all_log_vars, dim=0).numpy()

        results[split_name] = {
            "reconstruction_loss_L1": float(np.mean(recon_losses)),
            "kl_divergence": float(np.mean(kl_losses)),
            # How close the aggregate posterior is to N(0,I)
            "mean_latent_mean": float(np.mean(all_means)),
            "std_latent_mean": float(np.std(all_means)),
            "mean_latent_std": float(np.mean(np.exp(0.5 * all_log_vars))),
            # Active dimensions: fraction with KL > 0.1 nats
            "active_latent_dims": int(
                np.sum(
                    np.mean(
                        -0.5 * (1 + all_log_vars - all_means**2 - np.exp(all_log_vars)),
                        axis=0
                    ) > 0.1
                )
            ),
        }

        print(f"\n  [{split_name}]")
        for k, v in results[split_name].items():
            print(f"    {k:<35s}: {v:.4f}")

    return results


# =============================================================================
# Predictor metrics
# =============================================================================

def compute_predictor_metrics(full_pred_model, dataset):
    """
    Emission-space MSE, MAE, and R² on train and validation splits,
    both aggregate and per sector. Metrics are in the *scaled* emission
    space used during training (z-score normalised kg CO2/hab).
    """
    print("\n" + "=" * 60)
    print("EMISSION PREDICTOR METRICS")
    print("=" * 60)

    # Use DatasetPrediction so we get emissions_prev
    from scripts.elements.datasets import DatasetPrediction
    pred_dataset = DatasetPrediction.__new__(DatasetPrediction)
    pred_dataset.__dict__.update(dataset.__dict__)
    pred_dataset.__class__ = DatasetPrediction

    generator = torch.Generator().manual_seed(SEED)
    n_val = int(len(pred_dataset) * VAL_SPLIT)
    n_train = len(pred_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        pred_dataset, [n_train, n_val], generator=generator
    )

    results = {}
    per_sector_results = {}

    for split_name, split_ds in [("train", train_ds), ("val", val_ds)]:
        loader = DataLoader(split_ds, batch_size=BATCH_SIZE, shuffle=False)

        all_pred, all_true = [], []

        with torch.inference_mode():
            for batch in loader:
                (
                    x_cur, c_cur, y_cur,
                    x_prev, c_prev, y_prev,
                ) = [b.to(DEVICE) for b in batch]

                (
                    delta_pred, _,
                    _, _, _, _, _, _,
                ) = full_pred_model(x_cur, x_prev, c_cur, c_prev)

                y_pred = delta_pred + y_prev
                all_pred.append(y_pred.cpu())
                all_true.append(y_cur.cpu())

        all_pred = torch.cat(all_pred, dim=0).numpy()  # (N, n_sectors)
        all_true = torch.cat(all_true, dim=0).numpy()

        residuals = all_pred - all_true
        ss_res = np.sum(residuals**2, axis=0)
        ss_tot = np.sum((all_true - all_true.mean(axis=0))**2, axis=0)

        # Aggregate metrics
        results[split_name] = {
            "MSE": float(np.mean(residuals**2)),
            "RMSE": float(np.sqrt(np.mean(residuals**2))),
            "MAE": float(np.mean(np.abs(residuals))),
            "R2": float(1 - ss_res.sum() / ss_tot.sum()),
        }

        # Per-sector metrics
        per_sector_results[split_name] = {}
        for i, sector in enumerate(EMISSION_SECTORS):
            r2 = float(1 - ss_res[i] / ss_tot[i]) if ss_tot[i] > 0 else float("nan")
            per_sector_results[split_name][sector] = {
                "MSE": float(np.mean(residuals[:, i]**2)),
                "RMSE": float(np.sqrt(np.mean(residuals[:, i]**2))),
                "MAE": float(np.mean(np.abs(residuals[:, i]))),
                "R2": r2,
            }

        print(f"\n  [{split_name}] Aggregate")
        for k, v in results[split_name].items():
            print(f"    {k:<10s}: {v:.4f}")

        print(f"\n  [{split_name}] Per sector")
        for sector, metrics in per_sector_results[split_name].items():
            print(f"    {sector:<18s}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return results, per_sector_results


# =============================================================================
# Forecaster metrics
# =============================================================================

def compute_forecaster_metrics(full_fcast_model, dataset):
    """
    Latent-space MSE between forecasted z_t and the encoder's mean μ(x_t)
    on train and validation splits.
    """
    print("\n" + "=" * 60)
    print("LATENT FORECASTER METRICS")
    print("=" * 60)

    fcast_dataset = DatasetForecasting(dataset)

    generator = torch.Generator().manual_seed(SEED)
    n_val = int(fcast_dataset.base_length * VAL_SPLIT)
    n_train = fcast_dataset.base_length - n_val

    # Split on the base (non-inflated) length to avoid train/val leakage
    base_indices = list(range(fcast_dataset.base_length))
    rng = random.Random(SEED)
    rng.shuffle(base_indices)
    val_indices = base_indices[:n_val]
    train_indices = base_indices[n_val:]

    results = {}
    for split_name, indices in [("train", train_indices), ("val", val_indices)]:
        subset = torch.utils.data.Subset(fcast_dataset, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

        mse_list, mae_list = [], []

        with torch.inference_mode():
            for batch in loader:
                (
                    x_cur, c_cur,
                    x_prev, c_prev,
                    x_past, c_past,
                ) = [b.to(DEVICE) for b in batch]

                # Target: encoder mean of the current year
                mean_target, _ = full_fcast_model.encoder(x_cur)

                # Prediction: forecasted latent from t-1, t-2
                z_forecast = full_fcast_model(x_prev, x_past, c_cur, c_prev)

                residuals = z_forecast - mean_target
                mse_list.append(residuals.pow(2).mean().item())
                mae_list.append(residuals.abs().mean().item())

        results[split_name] = {
            "latent_MSE": float(np.mean(mse_list)),
            "latent_RMSE": float(np.sqrt(np.mean(mse_list))),
            "latent_MAE": float(np.mean(mae_list)),
        }

        print(f"\n  [{split_name}]")
        for k, v in results[split_name].items():
            print(f"    {k:<20s}: {v:.6f}")

    return results


# =============================================================================
# Save to CSV
# =============================================================================

def save_summary_tables(vae_results, pred_results, pred_per_sector, fcast_results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Flat summary table ---
    rows = []

    for split in ["train", "val"]:
        for k, v in vae_results[split].items():
            rows.append({"model": "VAE", "split": split, "metric": k, "value": round(v, 6)})
        for k, v in pred_results[split].items():
            rows.append({"model": "Predictor", "split": split, "metric": k, "value": round(v, 6)})
        for k, v in fcast_results[split].items():
            rows.append({"model": "Forecaster", "split": split, "metric": k, "value": round(v, 6)})

    summary_df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "SI_training_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # --- Per-sector predictor table ---
    sector_rows = []
    for split in ["train", "val"]:
        for sector, metrics in pred_per_sector[split].items():
            for k, v in metrics.items():
                sector_rows.append({
                    "split": split,
                    "sector": sector,
                    "metric": k,
                    "value": round(v, 6),
                })

    sector_df = pd.DataFrame(sector_rows)
    sector_path = OUTPUT_DIR / "SI_training_metrics_per_sector.csv"
    sector_df.to_csv(sector_path, index=False)
    print(f"Saved per-sector: {sector_path}")

    # --- Pretty print for easy copy-paste into paper ---
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (for SI)")
    print("=" * 60)

    pivot = summary_df.pivot_table(
        index=["model", "metric"], columns="split", values="value"
    )
    print(pivot.to_string())

    print("\n" + "=" * 60)
    print("PER-SECTOR TABLE (for SI, validation split only)")
    print("=" * 60)

    val_sector = sector_df[sector_df["split"] == "val"].pivot_table(
        index="sector", columns="metric", values="value"
    )[["MAE", "RMSE", "MSE", "R2"]]
    print(val_sector.to_string())


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("COMPUTING TRAINING METRICS FOR SI")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_PATH)
    dataset.input_df = dataset.input_df.to(DEVICE)
    dataset.context_df = dataset.context_df.to(DEVICE)
    dataset.emi_df = dataset.emi_df.to(DEVICE)

    print("Loading models...")
    vae_model, full_pred, full_fcast, latent_dim = load_all_models(dataset)
    print(f"  Latent dim: {latent_dim}")

    vae_results = compute_vae_metrics(vae_model, dataset)
    pred_results, pred_per_sector = compute_predictor_metrics(full_pred, dataset)
    fcast_results = compute_forecaster_metrics(full_fcast, dataset)

    save_summary_tables(vae_results, pred_results, pred_per_sector, fcast_results)

    print("\nDone.")


if __name__ == "__main__":
    main()