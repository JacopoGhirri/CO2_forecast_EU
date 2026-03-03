"""
K-Fold Cross-validation comparison of model variants (Table 3).

Performs K-Fold CV to compare six emission prediction approaches. In each fold,
the held-out fold serves as the test set, and the remaining data is further
split randomly into train/validation sets to support early stopping.

Model variants:
  1. Baseline (No latent space) — Direct prediction from raw concatenated
     inputs (x_t, x_{t-1}, c_t, c_{t-1}) to emission deltas.
  2. PCA — Deterministic linear reduction followed by emission prediction.
  3. KPCA — Kernel PCA (RBF) as a nonlinear deterministic reduction.
  4. ICA — Independent Component Analysis as a deterministic reduction.
  5. VAE (no context) — VAE encoding without context variables.
  6. VAE + Context (Final) — Full pipeline with VAE encoding and context.

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
from torch.utils.data import DataLoader, Subset

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
# Fraction of the training pool (k-1 folds) reserved for validation / early stopping
INTERNAL_VAL_RATIO = 0.15

BATCH_SIZE = 128
EPOCHS_VAE = 3000
EPOCHS_PREDICTOR = 3000
EPOCHS_DIRECT = 3000
STOPPER_WINDOW = 50
LATENT_DIM = 10
REDUCTION_DIM = LATENT_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

NUM_WORKERS: int = 0
PIN_MEMORY: bool = False

DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
VARIABLE_FILE = Path("config/data/variable_selection.txt")
CHECKPOINT_PATH = Path("data/checkpoints/ablation_checkpoint.csv")
OUTPUT_DIR = Path("outputs/tables")

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

SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]
VariantName = Literal["baseline", "pca", "kpca", "ica", "vae_no_context", "vae_final"]


# =============================================================================
# Cached configuration loading
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
# EarlyStopper
# =============================================================================
class EarlyStopper:
    """Tracks smoothed validation loss and stores best model weights."""

    def __init__(self, window: int = STOPPER_WINDOW):
        self.history: deque[float] = deque(maxlen=window)
        self.best_smooth: float = float("inf")
        self.best_weights: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> None:
        self.history.append(val_loss)
        smooth = sum(self.history) / len(self.history)
        if smooth < self.best_smooth:
            self.best_smooth = smooth
            self.best_weights = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    def restore(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            model.to(DEVICE)


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


def _ensure_cpu_dataset(dataset: DatasetPrediction) -> None:
    global PIN_MEMORY, NUM_WORKERS
    tensor_attrs = ["input_df", "context_df", "emi_df"]
    is_cuda = any(
        hasattr(dataset, attr) and getattr(dataset, attr).is_cuda
        for attr in tensor_attrs
    )
    if is_cuda:
        print("  Dataset tensors on CUDA — moving to CPU for DataLoader compatibility")
        for attr in tensor_attrs:
            if hasattr(dataset, attr):
                setattr(dataset, attr, getattr(dataset, attr).cpu())
        for attr in dir(dataset):
            if not attr.startswith("_"):
                val = getattr(dataset, attr, None)
                if isinstance(val, torch.Tensor) and val.is_cuda:
                    setattr(dataset, attr, val.cpu())
    if DEVICE == "cuda":
        PIN_MEMORY, NUM_WORKERS = True, 4
    else:
        PIN_MEMORY, NUM_WORKERS = False, 0


# =============================================================================
# Model builders
# =============================================================================
def _build_vae(input_dim: int) -> VAEModel:
    config = _vae_config()
    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=config.vae_latent_dim,
        num_blocks=config.vae_num_blocks,
        dim_blocks=config.vae_dim_blocks,
        activation=config.vae_activation,
        normalization=config.vae_normalization,
        dropout=config.vae_dropouts,
        input_dropout=config.vae_input_dropouts,
    )
    decoder = Decoder(
        input_dim=input_dim,
        latent_dim=config.vae_latent_dim,
        num_blocks=config.vae_num_blocks,
        dim_blocks=config.vae_dim_blocks,
        activation=config.vae_activation,
        normalization=config.vae_normalization,
        dropout=config.vae_dropouts,
    )
    vae = VAEModel(encoder, decoder)
    vae.apply(init_weights)
    return vae


def _build_predictor(input_dim: int, uncertainty: bool = True) -> EmissionPredictor:
    config = _pred_config()
    predictor = EmissionPredictor(
        input_dim=input_dim,
        output_configs=output_configs,
        num_blocks=config.pred_num_blocks,
        dim_block=config.pred_dim_block,
        width_block=config.pred_width_block,
        activation=config.pred_activation,
        normalization=config.pred_normalization,
        dropout=config.pred_dropouts,
        uncertainty=uncertainty,
    )
    predictor.apply(init_weights)
    return predictor


def _get_pred_optimizer(params_or_groups, pred_config=None):
    if pred_config is None:
        pred_config = _pred_config()
    lr = pred_config.pred_lr
    wd = pred_config.pred_wd
    optimizer_cls = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "radam": torch.optim.RAdam,
    }.get(pred_config.pred_optimizer.lower(), torch.optim.Adam)
    return optimizer_cls(params_or_groups, lr=lr, weight_decay=wd, eps=1e-6)


# =============================================================================
# Training loops
# =============================================================================
def _train_vae(vae, train_loader, val_loader, epochs, extract_x):
    config = _vae_config()
    wr, wd = config.vae_wr, config.vae_wd

    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=config.vae_lr,
        weight_decay=config.vae_weight_decay,
        eps=1e-6,
    )
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    recon_history: deque[float] = deque(maxlen=STOPPER_WINDOW)
    best_recon_smooth = float("inf")
    best_weights = None

    for _ in range(epochs):
        vae.train()
        for batch in train_loader:
            x = extract_x(batch).to(DEVICE, non_blocking=PIN_MEMORY)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                x_hat, mean, log_var = vae(x)
                recon, kl = vae_loss(x, x_hat, mean, log_var)
                loss = wr * recon + wd * kl
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        # Validation — model selection on smoothed reconstruction loss
        vae.eval()
        val_recon = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                x = extract_x(batch).to(DEVICE, non_blocking=PIN_MEMORY)
                with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                    x_hat, mean, log_var = vae(x)
                    recon, _kl = vae_loss(x, x_hat, mean, log_var)
                val_recon += recon.item()
        val_recon /= len(val_loader)

        recon_history.append(val_recon)
        smooth = sum(recon_history) / len(recon_history)
        if smooth < best_recon_smooth:
            best_recon_smooth = smooth
            best_weights = {k: v.cpu().clone() for k, v in vae.state_dict().items()}

    if best_weights is not None:
        vae.load_state_dict(best_weights)
        vae.to(DEVICE)
    return vae


def _train_predictor_with_vae(
    full_model, train_loader, val_loader, epochs, loss_mode="factor"
):
    for p in full_model.encoder.parameters():
        p.requires_grad = False
    for p in full_model.decoder.parameters():
        p.requires_grad = False

    pred_config = _pred_config()
    lr = pred_config.pred_lr

    optimizer = _get_pred_optimizer(
        [
            {"params": full_model.encoder.parameters(), "lr": lr * 1e-3},
            {"params": full_model.decoder.parameters(), "lr": lr * 1e-3},
            {"params": full_model.predictor.parameters(), "lr": lr},
        ],
        pred_config,
    )

    stopper = EarlyStopper(window=STOPPER_WINDOW)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    for _ in range(epochs):
        full_model.train()
        for batch in train_loader:
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = [
                b.to(DEVICE, non_blocking=PIN_MEMORY) for b in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                delta_pred, unc, *_ = full_model(x_t, x_t1, c_t, c_t1)
                delta_true = y_t - y_t1
                loss = uncertainty_aware_mse_loss(
                    delta_true, delta_pred, unc, mode=loss_mode
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        full_model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                x_t, c_t, y_t, x_t1, c_t1, y_t1 = [
                    b.to(DEVICE, non_blocking=PIN_MEMORY) for b in batch
                ]
                with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                    delta_pred, unc, *_ = full_model(x_t, x_t1, c_t, c_t1)
                    delta_true = y_t - y_t1
                    val_loss += uncertainty_aware_mse_loss(
                        delta_true, delta_pred, unc, mode=loss_mode
                    ).item()
        val_loss /= len(val_loader)
        stopper.step(val_loss, full_model)

    stopper.restore(full_model)
    return full_model


def _train_predictor_no_context(full_model, train_loader, val_loader, epochs):
    for p in full_model.vae.parameters():
        p.requires_grad = False

    pred_config = _pred_config()
    optimizer = _get_pred_optimizer(full_model.predictor.parameters(), pred_config)
    stopper = EarlyStopper(window=STOPPER_WINDOW)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    def _forward_no_ctx(batch):
        x_t, _c_t, y_t, x_t1, _c_t1, y_t1 = [
            b.to(DEVICE, non_blocking=PIN_MEMORY) for b in batch
        ]
        mean_t, logvar_t = full_model.encoder(x_t)
        mean_t1, logvar_t1 = full_model.encoder(x_t1)
        z_t = reparameterize(mean_t, torch.exp(0.5 * logvar_t))
        z_t1 = reparameterize(mean_t1, torch.exp(0.5 * logvar_t1))
        inp = torch.cat([z_t, z_t1], dim=1)
        delta_pred, unc = full_model.predictor(inp)
        delta_true = y_t - y_t1
        return delta_pred, unc, delta_true

    for _ in range(epochs):
        full_model.train()
        full_model.vae.eval()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                delta_pred, unc, delta_true = _forward_no_ctx(batch)
                loss = uncertainty_aware_mse_loss(
                    delta_true, delta_pred, unc, mode="factor"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        full_model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                    delta_pred, unc, delta_true = _forward_no_ctx(batch)
                    val_loss += uncertainty_aware_mse_loss(
                        delta_true, delta_pred, unc, mode="factor"
                    ).item()
        val_loss /= len(val_loader)
        stopper.step(val_loss, full_model)

    stopper.restore(full_model)
    return full_model


def _train_direct_predictor(
    predictor, train_loader, val_loader, epochs, context_dim, loss_mode="factor"
):
    pred_config = _pred_config()
    optimizer = _get_pred_optimizer(predictor.parameters(), pred_config)
    stopper = EarlyStopper(window=STOPPER_WINDOW)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    for _ in range(epochs):
        predictor.train()
        for batch in train_loader:
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = [
                b.to(DEVICE, non_blocking=PIN_MEMORY) for b in batch
            ]
            inp = torch.cat([x_t, c_t, x_t1, c_t1], dim=1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                delta_pred, unc = predictor(inp)
                delta_true = y_t - y_t1
                loss = uncertainty_aware_mse_loss(
                    delta_true, delta_pred, unc, mode=loss_mode
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        predictor.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                x_t, c_t, y_t, x_t1, c_t1, y_t1 = [
                    b.to(DEVICE, non_blocking=PIN_MEMORY) for b in batch
                ]
                inp = torch.cat([x_t, c_t, x_t1, c_t1], dim=1)
                with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                    delta_pred, unc = predictor(inp)
                    delta_true = y_t - y_t1
                    val_loss += uncertainty_aware_mse_loss(
                        delta_true, delta_pred, unc, mode=loss_mode
                    ).item()
        val_loss /= len(val_loader)
        stopper.step(val_loss, predictor)

    stopper.restore(predictor)
    return predictor


def _fit_deterministic_reduction(method, X_train, n_components):
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=SEED)
    elif method == "kpca":
        reducer = KernelPCA(
            n_components=n_components,
            kernel="rbf",
            random_state=SEED,
            fit_inverse_transform=False,
        )
    elif method == "ica":
        reducer = FastICA(
            n_components=n_components,
            random_state=SEED,
            max_iter=500,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    reducer.fit(X_train)
    return reducer


# =============================================================================
# Pre-reduced dataset wrapper
# =============================================================================
class _PreReducedDataset:
    """Wraps DatasetPrediction, substituting pre-computed reduced features."""

    def __init__(self, original: DatasetPrediction, Z_all: torch.Tensor):
        self.original = original
        self.Z_all = Z_all
        self._original_input_df = original.input_df

    def __len__(self) -> int:
        return len(self.original)

    def __getitem__(self, idx: int):
        self.original.input_df = self.Z_all
        try:
            result = self.original[idx]
        finally:
            self.original.input_df = self._original_input_df
        return result


# =============================================================================
# Evaluation helpers
# =============================================================================
def _compute_metrics(preds, targets, uncs, loss_mode):
    n_sectors = preds.shape[1]
    metrics = {}
    maes, mses, corrs = [], [], []
    for i in range(n_sectors):
        mae_i = float(np.mean(np.abs(preds[:, i] - targets[:, i])))
        mse_i = float(np.mean((preds[:, i] - targets[:, i]) ** 2))
        try:
            corr_i, _ = pearsonr(targets[:, i], preds[:, i])
            corr_i = float(corr_i)
        except ValueError:
            corr_i = float("nan")
        metrics[f"mae_{i}"] = mae_i
        metrics[f"mse_{i}"] = mse_i
        metrics[f"corr_{i}"] = corr_i
        maes.append(mae_i)
        mses.append(mse_i)
        corrs.append(corr_i)

    metrics["mae"] = float(np.mean(maes))
    metrics["mse"] = float(np.mean(mses))
    metrics["corr"] = float(np.nanmean(corrs))

    preds_t = torch.tensor(preds, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)
    uncs_t = torch.tensor(uncs, dtype=torch.float32)
    loss_pred = uncertainty_aware_mse_loss(targets_t, preds_t, uncs_t, mode=loss_mode)
    metrics["loss_pred"] = float(loss_pred.item())
    return metrics


def _evaluate_vae_variant(full_model, test_loader, loss_mode="factor"):
    full_model.eval()
    all_preds, all_targets, all_uncs = [], [], []
    with torch.inference_mode():
        for batch in test_loader:
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = [b.to(DEVICE) for b in batch]
            delta_pred, unc, *_ = full_model(x_t, x_t1, c_t, c_t1)
            delta_true = y_t - y_t1
            all_preds.append(delta_pred.cpu())
            all_targets.append(delta_true.cpu())
            all_uncs.append(unc.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    uncs = torch.cat(all_uncs).numpy()
    return _compute_metrics(preds, targets, uncs, loss_mode)


def _evaluate_vae_no_context(full_model, test_loader):
    full_model.eval()
    all_preds, all_targets, all_uncs = [], [], []
    with torch.inference_mode():
        for batch in test_loader:
            x_t, _c_t, y_t, x_t1, _c_t1, y_t1 = [b.to(DEVICE) for b in batch]
            mean_t, logvar_t = full_model.encoder(x_t)
            mean_t1, logvar_t1 = full_model.encoder(x_t1)
            z_t = reparameterize(mean_t, torch.exp(0.5 * logvar_t))
            z_t1 = reparameterize(mean_t1, torch.exp(0.5 * logvar_t1))
            inp = torch.cat([z_t, z_t1], dim=1)
            delta_pred, unc = full_model.predictor(inp)
            delta_true = y_t - y_t1
            all_preds.append(delta_pred.cpu())
            all_targets.append(delta_true.cpu())
            all_uncs.append(unc.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    uncs = torch.cat(all_uncs).numpy()
    return _compute_metrics(preds, targets, uncs, loss_mode="factor")


def _evaluate_direct_variant(predictor, test_loader, loss_mode="factor"):
    predictor.eval()
    all_preds, all_targets, all_uncs = [], [], []
    with torch.inference_mode():
        for batch in test_loader:
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = [b.to(DEVICE) for b in batch]
            inp = torch.cat([x_t, c_t, x_t1, c_t1], dim=1)
            delta_pred, unc = predictor(inp)
            delta_true = y_t - y_t1
            all_preds.append(delta_pred.cpu())
            all_targets.append(delta_true.cpu())
            all_uncs.append(unc.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    uncs = torch.cat(all_uncs).numpy()
    return _compute_metrics(preds, targets, uncs, loss_mode)


# =============================================================================
# DataLoader factory
# =============================================================================
def _make_loaders(dataset, train_idx, val_idx, test_idx):
    kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    train_l = DataLoader(Subset(dataset, train_idx), shuffle=True, drop_last=True, **kwargs)
    val_l = DataLoader(Subset(dataset, val_idx), shuffle=False, **kwargs)
    test_l = DataLoader(Subset(dataset, test_idx), shuffle=False, **kwargs)
    return train_l, val_l, test_l


# =============================================================================
# Per-variant runners
# =============================================================================
def run_baseline(dataset, train_idx, val_idx, test_idx):
    input_dim = dataset.input_df.shape[1]
    context_dim = dataset.context_df.shape[1]
    predictor = _build_predictor(
        input_dim=2 * (input_dim + context_dim), uncertainty=True
    ).to(DEVICE)
    train_l, val_l, test_l = _make_loaders(dataset, train_idx, val_idx, test_idx)
    predictor = _train_direct_predictor(
        predictor, train_l, val_l, EPOCHS_DIRECT, context_dim
    )
    return _evaluate_direct_variant(predictor, test_l)


def run_deterministic_latent(method, dataset, train_idx, val_idx, test_idx):
    context_dim = dataset.context_df.shape[1]

    # Fit reducer on training inputs only (no data leakage)
    X_train = dataset.input_df[train_idx].cpu().numpy()
    reducer = _fit_deterministic_reduction(method, X_train, REDUCTION_DIM)

    # Pre-compute reduced features for ALL rows (needed because
    # DatasetPrediction pairing can reference any row as x_{t-1})
    Z_all = torch.tensor(
        reducer.transform(dataset.input_df.cpu().numpy()),
        dtype=torch.float32,
    )

    pre_ds = _PreReducedDataset(dataset, Z_all)
    train_l = DataLoader(
        Subset(pre_ds, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_l = DataLoader(
        Subset(pre_ds, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_l = DataLoader(
        Subset(pre_ds, test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    predictor = _build_predictor(
        input_dim=2 * (REDUCTION_DIM + context_dim), uncertainty=True
    ).to(DEVICE)
    predictor = _train_direct_predictor(
        predictor, train_l, val_l, EPOCHS_PREDICTOR, context_dim
    )
    return _evaluate_direct_variant(predictor, test_l)


# =============================================================================
# Single fold orchestrator
# =============================================================================
def run_single_fold(fold_id, dataset, train_val_pool_idx, test_idx):
    """Run all six model variants on a single fold.

    Args:
        fold_id: Fold index (used as seed offset for the internal val split).
        dataset: Full dataset.
        train_val_pool_idx: Indices of the k-1 training folds (numpy array).
        test_idx: Indices of the held-out test fold (numpy array).

    Returns:
        List of 6 metric dicts, one per variant.
    """
    # --- Internal train/val split from the training pool ---
    pool_indices = train_val_pool_idx.tolist()
    pool_size = len(pool_indices)
    val_size = int(pool_size * INTERNAL_VAL_RATIO)

    rng = np.random.default_rng(SEED + fold_id)
    shuffled = pool_indices.copy()
    rng.shuffle(shuffled)

    val_idx = shuffled[:val_size]
    train_idx = shuffled[val_size:]
    test_idx = test_idx.tolist()

    # Shared DataLoaders for variants that use the original features
    train_l, val_l, test_l = _make_loaders(dataset, train_idx, val_idx, test_idx)

    results = []

    def _record(name, metrics):
        metrics["fold"] = fold_id
        metrics["variant"] = name
        results.append(metrics)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- 1. Baseline ---
    print(f"  Fold {fold_id}: baseline...")
    _record("baseline", run_baseline(dataset, train_idx, val_idx, test_idx))

    # --- 2-4. Deterministic reductions ---
    for method in ("pca", "kpca", "ica"):
        print(f"  Fold {fold_id}: {method}...")
        _record(
            method,
            run_deterministic_latent(method, dataset, train_idx, val_idx, test_idx),
        )

    # --- 5-6. VAE variants (shared VAE training) ---
    print(f"  Fold {fold_id}: training shared VAE...")
    input_dim = dataset.input_df.shape[1]
    context_dim = dataset.context_df.shape[1]
    vae = _build_vae(input_dim).to(DEVICE)
    vae = _train_vae(vae, train_l, val_l, EPOCHS_VAE, extract_x=lambda b: b[0])

    # Deep-copy VAE weights so the two predictor runs don't interfere
    vae_weights = {k: v.cpu().clone() for k, v in vae.state_dict().items()}
    vae_cfg = _vae_config()
    latent_dim = vae_cfg.vae_latent_dim

    # 5. VAE no context
    print(f"  Fold {fold_id}: vae_no_context...")
    vae_nc = _build_vae(input_dim).to(DEVICE)
    vae_nc.load_state_dict(vae_weights)
    vae_nc.to(DEVICE)
    pred_nc = _build_predictor(input_dim=2 * latent_dim, uncertainty=True).to(DEVICE)
    fm_nc = FullPredictionModel(vae=vae_nc, predictor=pred_nc).to(DEVICE)
    fm_nc = _train_predictor_no_context(fm_nc, train_l, val_l, EPOCHS_PREDICTOR)
    _record("vae_no_context", _evaluate_vae_no_context(fm_nc, test_l))
    del fm_nc, vae_nc, pred_nc

    # 6. VAE + Context (Final)
    print(f"  Fold {fold_id}: vae_final...")
    vae_f = _build_vae(input_dim).to(DEVICE)
    vae_f.load_state_dict(vae_weights)
    vae_f.to(DEVICE)
    pred_f = _build_predictor(
        input_dim=2 * (latent_dim + context_dim), uncertainty=True
    ).to(DEVICE)
    fm_f = FullPredictionModel(vae=vae_f, predictor=pred_f).to(DEVICE)
    fm_f = _train_predictor_with_vae(fm_f, train_l, val_l, EPOCHS_PREDICTOR)
    _record("vae_final", _evaluate_vae_variant(fm_f, test_l))
    del fm_f, vae_f, pred_f, vae, vae_weights

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
        ("mae", "MAE ↓"),
        ("mse", "MSE ↓"),
        ("corr", "Pearson ρ ↑"),
        ("loss_pred", "ℒ_pred ↓"),
    ]:
        row_str = f"{symbol:<18}"
        for variant in [
            "baseline",
            "pca",
            "kpca",
            "ica",
            "vae_no_context",
            "vae_final",
        ]:
            sub = summary[summary["variant"] == variant]
            mean_val = sub[metric].mean()
            std_val = sub[metric].std()
            row_str += f"{mean_val:>8.3f}({std_val:.3f})"
        rows.append(row_str)
    return "\n".join([separator, header, separator] + rows + [separator])


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("K-FOLD CROSS-VALIDATION MODEL COMPARISON (Table 3)")
    print("=" * 70)
    print(f"Folds: {N_FOLDS}")
    print(f"Internal val ratio (from training pool): {INTERNAL_VAL_RATIO:.0%}")
    print(f"Device: {DEVICE}")
    print(f"Latent dim: {LATENT_DIM}")
    print(
        f"Epochs — VAE: {EPOCHS_VAE}, Predictor: {EPOCHS_PREDICTOR}, "
        f"Direct: {EPOCHS_DIRECT}"
    )
    print(f"Early-stop window: {STOPPER_WINDOW}")
    print(f"Mixed precision (AMP): {USE_AMP}")
    print()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = get_or_create_dataset()
    _ensure_cpu_dataset(dataset)
    print(f"Dataset: {len(dataset)} samples")
    print(f"  Input features: {dataset.input_df.shape[1]}")
    print(f"  Context features: {dataset.context_df.shape[1]}")
    print(f"  Emission sectors: {dataset.emi_df.shape[1]}")
    print()

    all_results = []
    start_fold = 0

    # Resume from checkpoint if available
    if CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
        all_results = checkpoint_df.to_dict("records")
        completed_folds = set(checkpoint_df["fold"].unique())
        start_fold = max(completed_folds) + 1
        print(
            f"  Found {len(completed_folds)} completed folds, "
            f"resuming from fold {start_fold}"
        )

    # K-Fold split — deterministic via random_state
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_indices = np.arange(len(dataset))
    folds = list(kf.split(all_indices))

    for fold_id in range(start_fold, N_FOLDS):
        train_pool, test_idx = folds[fold_id]

        print(f"\n{'=' * 50}")
        print(
            f"FOLD {fold_id + 1}/{N_FOLDS}  "
            f"(train pool: {len(train_pool)}, test: {len(test_idx)})"
        )
        print(f"{'=' * 50}")

        fold_results = run_single_fold(fold_id, dataset, train_pool, test_idx)
        all_results.extend(fold_results)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Save checkpoint after every fold
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).to_csv(CHECKPOINT_PATH, index=False)

    results_df = pd.DataFrame(all_results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "table3_cv_comparison.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"Checkpoint removed: {CHECKPOINT_PATH}")

    print("\n" + "=" * 70)
    print("TABLE 3: Performance comparison across model variants")
    print(f"mean (std) across {N_FOLDS}-fold CV.")
    print("=" * 70)
    print(format_table(results_df))

    print("\n\nDetailed summary:")
    for variant in ["baseline", "pca", "kpca", "ica", "vae_no_context", "vae_final"]:
        sub = results_df[results_df["variant"] == variant]
        print(f"\n  {variant}:")
        for metric in ["mae", "mse", "corr", "loss_pred"]:
            print(
                f"    {metric:>10}: {sub[metric].mean():.4f} ± {sub[metric].std():.4f}"
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
