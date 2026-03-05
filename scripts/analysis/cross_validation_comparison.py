"""
Ablation study: representation quality comparison (Table 3).

Compares six emission-prediction approaches by K-Fold cross-validating
**only the supervised predictor**, while keeping the unsupervised
representation fixed across folds.

Methodological justification
-----------------------------
The VAE, PCA, KPCA, and ICA are all *unsupervised* — they are fitted on
input features only and never see emission labels.  Fitting them on the
full dataset therefore does **not** leak target information into the
representation.  The supervised predictor (which maps representations →
emission deltas) is still properly cross-validated: it is trained on
k-1 folds and evaluated on the held-out fold.

For VAE variants, we replicate the production pipeline exactly:
  - Load pre-trained VAE weights as initialization
  - Freeze encoder/decoder via requires_grad = False
  - But include them in the optimizer at lr * 1e-3 (joint fine-tuning)
  - This matches train_predictor.py behaviour

Training uses the **full k-1 folds** (no internal validation split).
With PATIENCE = EPOCHS the stopper tracks the best training loss and
restores those weights, but never stops early.  This maximises the
training set size (~302 samples) to match production conditions (~322).

Model variants
--------------
  1. Baseline — Direct prediction from raw concatenated inputs.
  2. PCA     — Deterministic linear reduction.
  3. KPCA    — Kernel PCA (RBF) nonlinear reduction.
  4. ICA     — Independent Component Analysis.
  5. VAE (no context) — VAE encoding, predictor sees only latents.
  6. VAE + Context     — Full pipeline with joint fine-tuning (final model).

Usage:
    python -m scripts.analysis.cross_validation_comparison

Outputs:
    - outputs/tables/table3_cv_comparison.csv
"""

from __future__ import annotations

import multiprocessing as mp
import random
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.model_selection import KFold
from tqdm import tqdm

from config.data.output_configs import output_configs
from scripts.elements.datasets import DatasetPrediction
from scripts.elements.models import (
    Decoder,
    EmissionPredictor,
    Encoder,
    FullPredictionModel,
    VAEModel,
    reparameterize,
    uncertainty_aware_mse_loss,
    vae_loss,
)
from scripts.utils import (
    init_weights,
    load_config,
    load_dataset,
    save_dataset,
)

# =============================================================================
# Configuration
# =============================================================================
SEED = 0
N_FOLDS = 5

EPOCHS_VAE = 5000       # trained once on full data
EPOCHS_PREDICTOR = 5000 # per fold (VAE variants with joint fine-tuning)
EPOCHS_DIRECT = 5000    # per fold (baseline / det. reduction predictors)
STOPPER_WINDOW = 50
PATIENCE = 5000         # effectively no early stopping
LATENT_DIM = 10
REDUCTION_DIM = LATENT_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
VARIABLE_FILE = Path("config/data/variable_selection.txt")
CHECKPOINT_PATH = Path("data/checkpoints/ablation_checkpoint.csv")
OUTPUT_DIR = Path("outputs/tables")

EU27_COUNTRIES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "EL", "FI",
    "FR", "DE", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
]

SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]
VariantName = Literal["baseline", "pca", "kpca", "ica", "vae_no_context", "vae_final"]


# =============================================================================
# Config helpers
# =============================================================================
_config_cache: dict[str, object] = {}


def _get_config(path: str) -> object:
    if path not in _config_cache:
        _config_cache[path] = load_config(path)
    return _config_cache[path]


def _vae_config():
    return _get_config("config/models/vae_config.yaml")


def _pred_config():
    return _get_config("config/models/co2_predictor_config.yaml")


# =============================================================================
# EarlyStopper (tracks training loss — no validation split)
# =============================================================================
class EarlyStopper:
    def __init__(self, window: int = STOPPER_WINDOW, patience: int = PATIENCE):
        self.window = window
        self.patience = patience
        self.history: deque[float] = deque(maxlen=window)
        self.best_smooth: float = float("inf")
        self.best_weights: dict | None = None
        self._epochs_without_improvement: int = 0

    def step(self, loss: float, model: nn.Module) -> bool:
        self.history.append(loss)
        smooth = sum(self.history) / len(self.history)
        if smooth < self.best_smooth:
            self.best_smooth = smooth
            self.best_weights = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        return self._epochs_without_improvement >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            model.to(DEVICE)

    @property
    def epochs_without_improvement(self) -> int:
        return self._epochs_without_improvement


# =============================================================================
# Dataset loading
# =============================================================================
def get_or_create_dataset() -> DatasetPrediction:
    if DATASET_PATH.exists():
        print(f"Loading cached dataset from {DATASET_PATH}")
        ds = load_dataset(DATASET_PATH)
        if not isinstance(ds, DatasetPrediction):
            ds.__class__ = DatasetPrediction
        return ds

    with open(VARIABLE_FILE) as f:
        nested_variables = [line.strip() for line in f if line.strip()]

    print("Creating dataset from raw CSVs...")
    dataset = DatasetPrediction(
        path_csvs="data/full_timeseries/",
        output_configs=output_configs,
        select_years=np.arange(2010, 2023 + 1),
        select_geo=EU27_COUNTRIES,
        nested_variables=nested_variables,
        with_cuda=False,
        scaling_type="normalization",
    )
    save_dataset(dataset, DATASET_PATH)
    print(f"Dataset saved to {DATASET_PATH}")
    return dataset


# =============================================================================
# GPU-resident flat tensors
# =============================================================================
class FlatGPUData:
    """Pre-paired, GPU-resident tensors for a specific index subset."""

    def __init__(self, dataset: DatasetPrediction, indices: list[int] | np.ndarray):
        indices = list(indices)
        prev_indices = []
        for i in indices:
            geo = dataset.keys.iloc[i, 0]
            year = dataset.keys.iloc[i, 1]
            prev_idx = dataset.index_map.get((geo, year - 1))
            prev_indices.append(prev_idx if prev_idx is not None else i)

        idx_t = torch.tensor(indices, dtype=torch.long)
        idx_t1 = torch.tensor(prev_indices, dtype=torch.long)

        self.x_t = dataset.input_df[idx_t].to(DEVICE)
        self.c_t = dataset.context_df[idx_t].to(DEVICE)
        self.y_t = dataset.emi_df[idx_t].to(DEVICE)
        self.x_t1 = dataset.input_df[idx_t1].to(DEVICE)
        self.c_t1 = dataset.context_df[idx_t1].to(DEVICE)
        self.y_t1 = dataset.emi_df[idx_t1].to(DEVICE)
        self.n = len(indices)

    def batches(self, batch_size: int, shuffle: bool = False):
        perm = torch.randperm(self.n, device=DEVICE) if shuffle else torch.arange(self.n, device=DEVICE)
        for start in range(0, self.n, batch_size):
            idx = perm[start : start + batch_size]
            if shuffle and len(idx) < batch_size:
                continue
            yield (
                self.x_t[idx], self.c_t[idx], self.y_t[idx],
                self.x_t1[idx], self.c_t1[idx], self.y_t1[idx],
            )


class FlatGPUDataReduced:
    """FlatGPUData with pre-reduced x features (PCA / KPCA / ICA)."""

    def __init__(self, dataset, Z_all: torch.Tensor, indices):
        indices = list(indices)
        prev_indices = []
        for i in indices:
            geo = dataset.keys.iloc[i, 0]
            year = dataset.keys.iloc[i, 1]
            prev_idx = dataset.index_map.get((geo, year - 1))
            prev_indices.append(prev_idx if prev_idx is not None else i)

        idx_t = torch.tensor(indices, dtype=torch.long)
        idx_t1 = torch.tensor(prev_indices, dtype=torch.long)

        self.x_t = Z_all[idx_t].to(DEVICE)
        self.c_t = dataset.context_df[idx_t].to(DEVICE)
        self.y_t = dataset.emi_df[idx_t].to(DEVICE)
        self.x_t1 = Z_all[idx_t1].to(DEVICE)
        self.c_t1 = dataset.context_df[idx_t1].to(DEVICE)
        self.y_t1 = dataset.emi_df[idx_t1].to(DEVICE)
        self.n = len(indices)

    def batches(self, batch_size: int, shuffle: bool = False):
        perm = torch.randperm(self.n, device=DEVICE) if shuffle else torch.arange(self.n, device=DEVICE)
        for start in range(0, self.n, batch_size):
            idx = perm[start : start + batch_size]
            if shuffle and len(idx) < batch_size:
                continue
            yield (
                self.x_t[idx], self.c_t[idx], self.y_t[idx],
                self.x_t1[idx], self.c_t1[idx], self.y_t1[idx],
            )


def _auto_batch_size(n: int) -> int:
    return n if n <= 2048 else 2048


# =============================================================================
# Model builders
# =============================================================================
def _build_vae(input_dim: int) -> VAEModel:
    config = _vae_config()
    encoder = Encoder(
        input_dim=input_dim, latent_dim=config.vae_latent_dim,
        num_blocks=config.vae_num_blocks, dim_blocks=config.vae_dim_blocks,
        activation=config.vae_activation, normalization=config.vae_normalization,
        dropout=config.vae_dropouts, input_dropout=config.vae_input_dropouts,
    )
    decoder = Decoder(
        input_dim=input_dim, latent_dim=config.vae_latent_dim,
        num_blocks=config.vae_num_blocks, dim_blocks=config.vae_dim_blocks,
        activation=config.vae_activation, normalization=config.vae_normalization,
        dropout=config.vae_dropouts,
    )
    vae = VAEModel(encoder, decoder)
    vae.apply(init_weights)
    return vae


def _build_predictor(input_dim: int, uncertainty: bool = True) -> EmissionPredictor:
    config = _pred_config()
    pred = EmissionPredictor(
        input_dim=input_dim, output_configs=output_configs,
        num_blocks=config.pred_num_blocks, dim_block=config.pred_dim_block,
        width_block=config.pred_width_block, activation=config.pred_activation,
        normalization=config.pred_normalization, dropout=config.pred_dropouts,
        uncertainty=uncertainty,
    )
    pred.apply(init_weights)
    return pred


def _get_pred_optimizer(params, pred_config=None):
    if pred_config is None:
        pred_config = _pred_config()
    cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
           "radam": torch.optim.RAdam}.get(pred_config.pred_optimizer.lower(), torch.optim.Adam)
    return cls(params, lr=pred_config.pred_lr, weight_decay=pred_config.pred_wd, eps=1e-6)


# =============================================================================
# Load pre-trained VAE weights
# =============================================================================
def train_vae_full(dataset: DatasetPrediction) -> VAEModel:
    """Train VAE on the entire dataset (unsupervised — no label leakage)."""
    input_dim = dataset.input_df.shape[1]
    vae = _build_vae(input_dim).to(DEVICE)
    config = _vae_config()

    n = len(dataset)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    val_size = int(n * 0.15)
    val_idx = perm[:val_size].tolist()
    train_idx = perm[val_size:].tolist()

    train_data = FlatGPUData(dataset, train_idx)
    val_data = FlatGPUData(dataset, val_idx)
    bs = _auto_batch_size(train_data.n)

    optimizer = torch.optim.AdamW(
        vae.parameters(), lr=config.vae_lr,
        weight_decay=config.vae_weight_decay, eps=1e-6,
    )
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    stopper = EarlyStopper(window=STOPPER_WINDOW, patience=PATIENCE)

    pbar = tqdm(range(EPOCHS_VAE), desc="  VAE (full data)", leave=False, ncols=110)
    for epoch in pbar:
        vae.train()
        trn_loss = 0.0; nb = 0
        for batch in train_data.batches(bs, shuffle=True):
            x = batch[0]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                x_hat, mean, log_var = vae(x)
                recon, kl = vae_loss(x, x_hat, mean, log_var)
                loss = config.vae_wr * recon + config.vae_wd * kl
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            trn_loss += loss.item(); nb += 1

        vae.eval()
        val_recon = 0.0; nv = 0
        with torch.inference_mode():
            for batch in val_data.batches(val_data.n):
                x = batch[0]
                with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                    x_hat, mean, log_var = vae(x)
                    recon, _ = vae_loss(x, x_hat, mean, log_var)
                val_recon += recon.item(); nv += 1
        val_recon /= max(nv, 1)

        should_stop = stopper.step(val_recon, vae)
        pbar.set_postfix(
            trn=f"{trn_loss/max(nb,1):.4f}", val=f"{val_recon:.4f}",
            pat=f"{stopper.epochs_without_improvement}/{PATIENCE}",
        )
        if should_stop:
            break

    stopper.restore(vae)
    pbar.close()
    return vae


def load_pretrained_vae(dataset: DatasetPrediction) -> dict:
    """Load production VAE weights; fall back to training from scratch.
    Returns the state_dict (CPU) so each fold can initialize a fresh copy.
    """
    model_path = Path("data/pytorch_models/vae_model.pth")
    if model_path.exists():
        print(f"  Loading pre-trained VAE from {model_path}")
        return torch.load(model_path, map_location="cpu")
    print("  No pre-trained VAE found — training on full data...")
    vae = train_vae_full(dataset)
    return {k: v.cpu() for k, v in vae.state_dict().items()}


# =============================================================================
# Predictor training — direct (baseline, PCA, KPCA, ICA)
# Train on full train fold, stopper tracks training loss.
# =============================================================================
def _train_predictor_direct(predictor, train_data, epochs, desc, loss_mode="factor"):
    pred_config = _pred_config()
    optimizer = _get_pred_optimizer(predictor.parameters(), pred_config)
    stopper = EarlyStopper(window=STOPPER_WINDOW, patience=PATIENCE)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    bs = _auto_batch_size(train_data.n)

    pbar = tqdm(range(epochs), desc=desc, leave=False, ncols=110)
    for epoch in pbar:
        predictor.train()
        trn_loss = 0.0; nb = 0
        for batch in train_data.batches(bs, shuffle=True):
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = batch
            inp = torch.cat([x_t, c_t, x_t1, c_t1], dim=1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                delta_pred, unc = predictor(inp)
                delta_true = y_t - y_t1
                loss = uncertainty_aware_mse_loss(delta_true, delta_pred, unc, mode=loss_mode)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            trn_loss += loss.item(); nb += 1

        avg_loss = trn_loss / max(nb, 1)
        should_stop = stopper.step(avg_loss, predictor)
        pbar.set_postfix(
            trn=f"{avg_loss:.4f}",
            pat=f"{stopper.epochs_without_improvement}/{PATIENCE}",
        )
        if should_stop:
            break

    stopper.restore(predictor)
    pbar.close()
    return predictor


# =============================================================================
# VAE + Context predictor — replicates train_predictor.py exactly
# =============================================================================
def _train_predictor_with_vae(full_model, train_data, epochs, desc, loss_mode="factor"):
    for p in full_model.encoder.parameters():
        p.requires_grad = False
    for p in full_model.decoder.parameters():
        p.requires_grad = False

    pred_config = _pred_config()
    lr = pred_config.pred_lr

    optimizer_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
                     "radam": torch.optim.RAdam}.get(
        pred_config.pred_optimizer.lower(), torch.optim.Adam
    )
    optimizer = optimizer_cls(
        [
            {"params": full_model.encoder.parameters(), "lr": lr * 1e-3},
            {"params": full_model.decoder.parameters(), "lr": lr * 1e-3},
            {"params": full_model.predictor.parameters(), "lr": lr},
        ],
        weight_decay=pred_config.pred_wd,
        eps=1e-6,
    )

    stopper = EarlyStopper(window=STOPPER_WINDOW, patience=PATIENCE)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    bs = _auto_batch_size(train_data.n)

    pbar = tqdm(range(epochs), desc=desc, leave=False, ncols=110)
    for epoch in pbar:
        full_model.train()
        trn_loss = 0.0; nb = 0
        for batch in train_data.batches(bs, shuffle=True):
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = batch
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                delta_pred, unc, *_ = full_model(x_t, x_t1, c_t, c_t1)
                delta_true = y_t - y_t1
                loss = uncertainty_aware_mse_loss(delta_true, delta_pred, unc, mode=loss_mode)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            trn_loss += loss.item(); nb += 1

        avg_loss = trn_loss / max(nb, 1)
        should_stop = stopper.step(avg_loss, full_model)
        pbar.set_postfix(
            trn=f"{avg_loss:.4f}",
            pat=f"{stopper.epochs_without_improvement}/{PATIENCE}",
        )
        if should_stop:
            break

    stopper.restore(full_model)
    pbar.close()
    return full_model


# =============================================================================
# VAE no-context predictor
# =============================================================================
def _train_predictor_vae_no_ctx(full_model, train_data, epochs, loss_mode="factor"):
    for p in full_model.vae.parameters():
        p.requires_grad = False

    pred_config = _pred_config()
    lr = pred_config.pred_lr

    optimizer_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
                     "radam": torch.optim.RAdam}.get(
        pred_config.pred_optimizer.lower(), torch.optim.Adam
    )
    optimizer = optimizer_cls(
        [
            {"params": full_model.encoder.parameters(), "lr": lr * 1e-3},
            {"params": full_model.predictor.parameters(), "lr": lr},
        ],
        weight_decay=pred_config.pred_wd,
        eps=1e-6,
    )

    stopper = EarlyStopper(window=STOPPER_WINDOW, patience=PATIENCE)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    bs = _auto_batch_size(train_data.n)

    def _forward_no_ctx(batch):
        x_t, _c_t, y_t, x_t1, _c_t1, y_t1 = batch
        mean_t, logvar_t = full_model.encoder(x_t)
        mean_t1, logvar_t1 = full_model.encoder(x_t1)
        z_t = reparameterize(mean_t, torch.exp(0.5 * logvar_t))
        z_t1 = reparameterize(mean_t1, torch.exp(0.5 * logvar_t1))
        inp = torch.cat([z_t, z_t1], dim=1)
        delta_pred, unc = full_model.predictor(inp)
        delta_true = y_t - y_t1
        return delta_pred, unc, delta_true

    pbar = tqdm(range(epochs), desc="    vae_no_ctx", leave=False, ncols=110)
    for epoch in pbar:
        full_model.train()
        full_model.vae.eval()
        trn_loss = 0.0; nb = 0
        for batch in train_data.batches(bs, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                delta_pred, unc, delta_true = _forward_no_ctx(batch)
                loss = uncertainty_aware_mse_loss(delta_true, delta_pred, unc, mode=loss_mode)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            trn_loss += loss.item(); nb += 1

        avg_loss = trn_loss / max(nb, 1)
        should_stop = stopper.step(avg_loss, full_model)
        pbar.set_postfix(
            trn=f"{avg_loss:.4f}",
            pat=f"{stopper.epochs_without_improvement}/{PATIENCE}",
        )
        if should_stop:
            break

    stopper.restore(full_model)
    pbar.close()
    return full_model


# =============================================================================
# Evaluation
# =============================================================================
def _compute_metrics(preds, targets, uncs, loss_mode="factor"):
    n_sectors = preds.shape[1]
    metrics = {}
    maes, mses, corrs = [], [], []
    for i in range(n_sectors):
        mae_i = float(np.mean(np.abs(preds[:, i] - targets[:, i])))
        mse_i = float(np.mean((preds[:, i] - targets[:, i]) ** 2))
        try:
            corr_i = float(pearsonr(targets[:, i], preds[:, i])[0])
        except ValueError:
            corr_i = float("nan")
        metrics[f"mae_{i}"] = mae_i
        metrics[f"mse_{i}"] = mse_i
        metrics[f"corr_{i}"] = corr_i
        maes.append(mae_i); mses.append(mse_i); corrs.append(corr_i)

    metrics["mae"] = float(np.mean(maes))
    metrics["mse"] = float(np.mean(mses))
    metrics["corr"] = float(np.nanmean(corrs))

    preds_t = torch.tensor(preds, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)
    uncs_t = torch.tensor(uncs, dtype=torch.float32)
    metrics["loss_pred"] = float(
        uncertainty_aware_mse_loss(targets_t, preds_t, uncs_t, mode=loss_mode).item()
    )
    return metrics


def _evaluate_direct(predictor, test_data, loss_mode="factor"):
    predictor.eval()
    all_p, all_t, all_u = [], [], []
    with torch.inference_mode():
        for batch in test_data.batches(test_data.n):
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = batch
            inp = torch.cat([x_t, c_t, x_t1, c_t1], dim=1)
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                dp, unc = predictor(inp)
            dt = y_t - y_t1
            all_p.append(dp.cpu()); all_t.append(dt.cpu()); all_u.append(unc.cpu())
    return _compute_metrics(
        torch.cat(all_p).numpy(), torch.cat(all_t).numpy(),
        torch.cat(all_u).numpy(), loss_mode,
    )


def _evaluate_vae_variant(full_model, test_data, loss_mode="factor"):
    full_model.eval()
    all_p, all_t, all_u = [], [], []
    with torch.inference_mode():
        for batch in test_data.batches(test_data.n):
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = batch
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                delta_pred, unc, *_ = full_model(x_t, x_t1, c_t, c_t1)
            dt = y_t - y_t1
            all_p.append(delta_pred.cpu()); all_t.append(dt.cpu()); all_u.append(unc.cpu())
    return _compute_metrics(
        torch.cat(all_p).numpy(), torch.cat(all_t).numpy(),
        torch.cat(all_u).numpy(), loss_mode,
    )


def _evaluate_vae_no_ctx(full_model, test_data, loss_mode="factor"):
    full_model.eval()
    all_p, all_t, all_u = [], [], []
    with torch.inference_mode():
        for batch in test_data.batches(test_data.n):
            x_t, _c_t, y_t, x_t1, _c_t1, y_t1 = batch
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                mean_t, logvar_t = full_model.encoder(x_t)
                mean_t1, logvar_t1 = full_model.encoder(x_t1)
                z_t = reparameterize(mean_t, torch.exp(0.5 * logvar_t))
                z_t1 = reparameterize(mean_t1, torch.exp(0.5 * logvar_t1))
                inp = torch.cat([z_t, z_t1], dim=1)
                dp, unc = full_model.predictor(inp)
            dt = y_t - y_t1
            all_p.append(dp.cpu()); all_t.append(dt.cpu()); all_u.append(unc.cpu())
    return _compute_metrics(
        torch.cat(all_p).numpy(), torch.cat(all_t).numpy(),
        torch.cat(all_u).numpy(), loss_mode,
    )


# =============================================================================
# Single fold — train on full k-1 folds, test on held-out fold
# =============================================================================
def run_fold(
    fold_id: int,
    dataset: DatasetPrediction,
    train_idx: list[int],
    test_idx: list[int],
    vae_weights: dict,
    Z_pca: torch.Tensor,
    Z_kpca: torch.Tensor,
    Z_ica: torch.Tensor,
):
    context_dim = dataset.context_df.shape[1]
    input_dim = dataset.input_df.shape[1]
    vae_cfg = _vae_config()
    pred_cfg = _pred_config()
    latent_dim = vae_cfg.vae_latent_dim
    loss_mode = pred_cfg.mode_loss
    results = []

    train_data = FlatGPUData(dataset, train_idx)
    test_data = FlatGPUData(dataset, test_idx)

    def _record(name, metrics):
        metrics["fold"] = fold_id
        metrics["variant"] = name
        results.append(metrics)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---- 1. Baseline ----
    print(f"    baseline")
    pred = _build_predictor(2 * (input_dim + context_dim)).to(DEVICE)
    pred = _train_predictor_direct(pred, train_data, EPOCHS_DIRECT, "    baseline", loss_mode)
    _record("baseline", _evaluate_direct(pred, test_data, loss_mode))
    del pred

    # ---- 2-4. Deterministic reductions ----
    for name, Z in [("pca", Z_pca), ("kpca", Z_kpca), ("ica", Z_ica)]:
        print(f"    {name}")
        tr_d = FlatGPUDataReduced(dataset, Z, train_idx)
        te_d = FlatGPUDataReduced(dataset, Z, test_idx)
        pred = _build_predictor(2 * (REDUCTION_DIM + context_dim)).to(DEVICE)
        pred = _train_predictor_direct(pred, tr_d, EPOCHS_PREDICTOR, f"    {name}", loss_mode)
        _record(name, _evaluate_direct(pred, te_d, loss_mode))
        del pred, tr_d, te_d

    # ---- 5. VAE no context ----
    print(f"    vae_no_context")
    vae_nc = _build_vae(input_dim).to(DEVICE)
    vae_nc.load_state_dict(vae_weights)
    vae_nc.to(DEVICE)
    pred_nc = _build_predictor(2 * latent_dim).to(DEVICE)
    fm_nc = FullPredictionModel(vae=vae_nc, predictor=pred_nc).to(DEVICE)
    fm_nc = _train_predictor_vae_no_ctx(fm_nc, train_data, EPOCHS_PREDICTOR, loss_mode)
    _record("vae_no_context", _evaluate_vae_no_ctx(fm_nc, test_data, loss_mode))
    del fm_nc, vae_nc, pred_nc

    # ---- 6. VAE + Context (production pipeline) ----
    print(f"    vae_final")
    vae_f = _build_vae(input_dim).to(DEVICE)
    vae_f.load_state_dict(vae_weights)
    vae_f.to(DEVICE)
    pred_f = _build_predictor(2 * (latent_dim + context_dim)).to(DEVICE)
    fm_f = FullPredictionModel(vae=vae_f, predictor=pred_f).to(DEVICE)
    fm_f = _train_predictor_with_vae(fm_f, train_data, EPOCHS_PREDICTOR, "    vae_final", loss_mode)
    _record("vae_final", _evaluate_vae_variant(fm_f, test_data, loss_mode))
    del fm_f, vae_f, pred_f

    del train_data, test_data
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return results


# =============================================================================
# Output formatting
# =============================================================================
def format_table(summary):
    header = (
        f"{'Metric':<18}"
        f"{'Baseline':>16}"
        f"{'PCA':>16}"
        f"{'KPCA':>16}"
        f"{'ICA':>16}"
        f"{'VAE':>16}"
        f"{'Final':>16}"
    )
    separator = "-" * len(header)
    rows = []
    for metric, symbol in [
        ("mae", "MAE ↓"), ("mse", "MSE ↓"),
        ("corr", "Pearson ρ ↑"), ("loss_pred", "ℒ_pred ↓"),
    ]:
        row_str = f"{symbol:<18}"
        for variant in ["baseline", "pca", "kpca", "ica", "vae_no_context", "vae_final"]:
            sub = summary[summary["variant"] == variant]
            m = sub[metric].mean(); s = sub[metric].std()
            row_str += f"{m:>8.3f}({s:.3f})"
        rows.append(row_str)
    return "\n".join([separator, header, separator] + rows + [separator])


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("ABLATION STUDY: Representation Quality Comparison (Table 3)")
    print("=" * 70)
    print()
    print("Design: unsupervised representations fitted ONCE on full data;")
    print("        supervised predictor K-fold cross-validated per fold.")
    print("        VAE variants use joint fine-tuning (production pipeline).")
    print("        Full k-1 folds used for training (no internal val split).")
    print()
    print(f"Folds: {N_FOLDS}")
    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    print(f"Predictor epochs: {EPOCHS_PREDICTOR}, VAE epochs: {EPOCHS_VAE}")
    print()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    dataset = get_or_create_dataset()
    for attr in ("input_df", "context_df", "emi_df"):
        t = getattr(dataset, attr)
        if t.is_cuda:
            setattr(dataset, attr, t.cpu())

    n = len(dataset)
    print(f"Dataset: {n} samples, {dataset.input_df.shape[1]} input features, "
          f"{dataset.context_df.shape[1]} context features, "
          f"{dataset.emi_df.shape[1]} emission sectors")

    # ------------------------------------------------------------------
    # 2. Fit all unsupervised representations on FULL data
    # ------------------------------------------------------------------
    print("\n--- Fitting unsupervised representations on full data ---")

    X_all_np = dataset.input_df.cpu().numpy()

    print("  PCA...")
    pca = PCA(n_components=REDUCTION_DIM, random_state=SEED).fit(X_all_np)
    Z_pca = torch.tensor(pca.transform(X_all_np), dtype=torch.float32)
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    print("  KPCA...")
    kpca = KernelPCA(n_components=REDUCTION_DIM, kernel="rbf",
                     random_state=SEED, fit_inverse_transform=False).fit(X_all_np)
    Z_kpca = torch.tensor(kpca.transform(X_all_np), dtype=torch.float32)

    print("  ICA...")
    ica = FastICA(n_components=REDUCTION_DIM, random_state=SEED, max_iter=500).fit(X_all_np)
    Z_ica = torch.tensor(ica.transform(X_all_np), dtype=torch.float32)

    print("  VAE...")
    vae_weights = load_pretrained_vae(dataset)

    print(f"\n  All representations ready (dim={REDUCTION_DIM})")

    # ------------------------------------------------------------------
    # 3. K-Fold CV — train on full k-1 folds, test on held-out fold
    # ------------------------------------------------------------------
    print(f"\n--- {N_FOLDS}-Fold Cross-Validation ---")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_indices = np.arange(n)
    folds = list(kf.split(all_indices))

    all_results = []
    start_fold = 0

    if CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt = pd.read_csv(CHECKPOINT_PATH)
        all_results = ckpt.to_dict("records")
        start_fold = max(ckpt["fold"].unique()) + 1

    for fold_id in tqdm(range(start_fold, N_FOLDS), desc="Folds", ncols=100):
        train_idx, test_idx = folds[fold_id]
        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()

        print(f"\n  FOLD {fold_id + 1}/{N_FOLDS} "
              f"(train={len(train_idx)}, test={len(test_idx)})")

        fold_results = run_fold(
            fold_id, dataset, train_idx, test_idx,
            vae_weights, Z_pca, Z_kpca, Z_ica,
        )
        all_results.extend(fold_results)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).to_csv(CHECKPOINT_PATH, index=False)

    # ------------------------------------------------------------------
    # 4. Results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "table3_cv_comparison.csv"
    results_df.to_csv(output_path, index=False)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print("\n" + "=" * 70)
    print("TABLE 3: Representation quality comparison")
    print(f"Predictor cross-validated ({N_FOLDS}-fold); "
          f"representations pre-trained on full data (unsupervised).")
    print("VAE variants include joint fine-tuning (production pipeline).")
    print("=" * 70)
    print(format_table(results_df))

    print("\n\nDetailed summary:")
    for variant in ["baseline", "pca", "kpca", "ica", "vae_no_context", "vae_final"]:
        sub = results_df[results_df["variant"] == variant]
        print(f"\n  {variant}:")
        for metric in ["mae", "mse", "corr", "loss_pred"]:
            print(f"    {metric:>10}: {sub[metric].mean():.4f} ± {sub[metric].std():.4f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()