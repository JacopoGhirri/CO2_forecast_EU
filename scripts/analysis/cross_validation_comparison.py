"""
Cross-validation comparison of model variants (Table 3).

Performs repeated random train/val/test splitting to compare six emission
prediction approaches on identical data partitions:

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
    check_nan_gradients,
    init_weights,
    load_config,
    load_dataset,
    save_dataset,
)

# =============================================================================
# Configuration
# =============================================================================
SEED = 0
N_SPLITS = 100
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 128

# Training epochs — aligned with standalone training scripts (5000 epochs).
# Predictor converges by ~epoch 2750; 3000 gives adequate room while
# keeping total compute tractable (100 splits × 6 variants).
EPOCHS_VAE = 3000
EPOCHS_PREDICTOR = 3000
EPOCHS_DIRECT = 3000

# Early-stopping smoothing window — matches training scripts (maxlen=50)
STOPPER_WINDOW = 50

LATENT_DIM = 10
REDUCTION_DIM = LATENT_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
VARIABLE_FILE = Path("config/data/variable_selection.txt")
OUTPUT_DIR = Path("outputs/tables")

EU27_COUNTRIES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "EL", "FI", "FR", "DE",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
]

# Emission sectors (from output_configs)
SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

# Model variant identifier type
VariantName = Literal["baseline", "pca", "kpca", "ica", "vae_no_context", "vae_final"]


# =============================================================================
# EarlyStopper
# =============================================================================
class EarlyStopper:
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


# =============================================================================
# Model builders
# =============================================================================
def _build_vae(input_dim: int) -> VAEModel:
    config = load_config("config/models/vae_config.yaml")
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
    config = load_config("config/models/co2_predictor_config.yaml")
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
    """Helper to build the predictor optimizer matching train_predictor.py."""
    if pred_config is None:
        pred_config = load_config("config/models/co2_predictor_config.yaml")
    lr = pred_config.pred_lr
    wd = pred_config.pred_wd
    optimizer_cls = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "radam": torch.optim.RAdam,
    }.get(pred_config.pred_optimizer.lower(), torch.optim.Adam)
    return optimizer_cls(params_or_groups, weight_decay=wd, eps=1e-6)


# =============================================================================
# Training loops
# =============================================================================
def _train_vae(vae, train_loader, val_loader, epochs, extract_x):
    """Train VAE matching train_vae.py: AdamW, smoothed recon loss selection."""
    config = load_config("config/models/vae_config.yaml")
    wr, wd = config.vae_wr, config.vae_wd

    optimizer = torch.optim.AdamW(
        vae.parameters(), lr=config.vae_lr,
        weight_decay=config.vae_weight_decay, eps=1e-6,
    )

    recon_history: deque[float] = deque(maxlen=STOPPER_WINDOW)
    best_recon_smooth = float("inf")
    best_weights = None

    for _ in range(epochs):
        vae.train()
        for batch in train_loader:
            x = extract_x(batch).to(DEVICE)
            optimizer.zero_grad()
            x_hat, mean, log_var = vae(x)
            recon, kl = vae_loss(x, x_hat, mean, log_var)
            loss = wr * recon + wd * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            check_nan_gradients(vae)
            optimizer.step()

        vae.eval()
        val_recon = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                x = extract_x(batch).to(DEVICE)
                x_hat, mean, log_var = vae(x)
                recon, kl = vae_loss(x, x_hat, mean, log_var)
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


def _train_predictor_with_vae(full_model, train_loader, val_loader, epochs,
                               loss_mode="factor"):
    """Train predictor with frozen VAE, matching train_predictor.py optimizer."""
    for p in full_model.encoder.parameters():
        p.requires_grad = False
    for p in full_model.decoder.parameters():
        p.requires_grad = False

    pred_config = load_config("config/models/co2_predictor_config.yaml")
    lr = pred_config.pred_lr

    optimizer = _get_pred_optimizer([
        {"params": full_model.encoder.parameters(), "lr": lr * 1e-3},
        {"params": full_model.decoder.parameters(), "lr": lr * 1e-3},
        {"params": full_model.predictor.parameters(), "lr": lr},
    ], pred_config)

    stopper = EarlyStopper(window=STOPPER_WINDOW)

    for _ in range(epochs):
        full_model.train()
        for batch in train_loader:
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            delta_pred, unc, *_ = full_model(x_t, x_t1, c_t, c_t1)
            delta_true = y_t - y_t1
            loss = uncertainty_aware_mse_loss(delta_true, delta_pred, unc, mode=loss_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
            check_nan_gradients(full_model)
            optimizer.step()

        full_model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                x_t, c_t, y_t, x_t1, c_t1, y_t1 = [b.to(DEVICE) for b in batch]
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
    """Train predictor without context: input is [z_t, z_{t-1}] only."""
    for p in full_model.vae.parameters():
        p.requires_grad = False

    pred_config = load_config("config/models/co2_predictor_config.yaml")
    optimizer = _get_pred_optimizer(
        full_model.predictor.parameters(), pred_config
    )
    stopper = EarlyStopper(window=STOPPER_WINDOW)

    def _forward_no_ctx(batch):
        x_t, _c_t, y_t, x_t1, _c_t1, y_t1 = [b.to(DEVICE) for b in batch]
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
        full_model.vae.eval()  # keep encoder in eval (no dropout)
        for batch in train_loader:
            optimizer.zero_grad()
            delta_pred, unc, delta_true = _forward_no_ctx(batch)
            loss = uncertainty_aware_mse_loss(delta_true, delta_pred, unc, mode="factor")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
            check_nan_gradients(full_model)
            optimizer.step()

        full_model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                delta_pred, unc, delta_true = _forward_no_ctx(batch)
                val_loss += uncertainty_aware_mse_loss(
                    delta_true, delta_pred, unc, mode="factor"
                ).item()
        val_loss /= len(val_loader)
        stopper.step(val_loss, full_model)

    stopper.restore(full_model)
    return full_model


def _train_direct_predictor(predictor, train_loader, val_loader, epochs,
                             context_dim, loss_mode="factor"):
    """Baseline: predict from raw [x_t, c_t, x_{t-1}, c_{t-1}]."""
    pred_config = load_config("config/models/co2_predictor_config.yaml")
    optimizer = _get_pred_optimizer(predictor.parameters(), pred_config)
    stopper = EarlyStopper(window=STOPPER_WINDOW)

    for _ in range(epochs):
        predictor.train()
        for batch in train_loader:
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = [b.to(DEVICE) for b in batch]
            inp = torch.cat([x_t, c_t, x_t1, c_t1], dim=1)
            optimizer.zero_grad()
            delta_pred, unc = predictor(inp)
            delta_true = y_t - y_t1
            loss = uncertainty_aware_mse_loss(delta_true, delta_pred, unc, mode=loss_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            check_nan_gradients(predictor)
            optimizer.step()

        predictor.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                x_t, c_t, y_t, x_t1, c_t1, y_t1 = [b.to(DEVICE) for b in batch]
                inp = torch.cat([x_t, c_t, x_t1, c_t1], dim=1)
                delta_pred, unc = predictor(inp)
                delta_true = y_t - y_t1
                val_loss += uncertainty_aware_mse_loss(
                    delta_true, delta_pred, unc, mode=loss_mode
                ).item()
        val_loss /= len(val_loader)
        stopper.step(val_loss, predictor)

    stopper.restore(predictor)
    return predictor


def _train_predictor_with_deterministic_latent(
    reducer, predictor, train_loader, val_loader, dataset, epochs,
    loss_mode="factor",
):
    """Train predictor using pre-fitted sklearn reducer."""
    pred_config = load_config("config/models/co2_predictor_config.yaml")
    optimizer = _get_pred_optimizer(predictor.parameters(), pred_config)
    stopper = EarlyStopper(window=STOPPER_WINDOW)

    def _encode(x):
        z = reducer.transform(x.cpu().numpy())
        return torch.tensor(z, dtype=torch.float32, device=DEVICE)

    for _ in range(epochs):
        predictor.train()
        for batch in train_loader:
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = [b.to(DEVICE) for b in batch]
            z_t, z_t1 = _encode(x_t), _encode(x_t1)
            inp = torch.cat([z_t, c_t, z_t1, c_t1], dim=1)
            optimizer.zero_grad()
            delta_pred, unc = predictor(inp)
            delta_true = y_t - y_t1
            loss = uncertainty_aware_mse_loss(delta_true, delta_pred, unc, mode=loss_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            check_nan_gradients(predictor)
            optimizer.step()

        predictor.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                x_t, c_t, y_t, x_t1, c_t1, y_t1 = [b.to(DEVICE) for b in batch]
                z_t, z_t1 = _encode(x_t), _encode(x_t1)
                inp = torch.cat([z_t, c_t, z_t1, c_t1], dim=1)
                delta_pred, unc = predictor(inp)
                delta_true = y_t - y_t1
                val_loss += uncertainty_aware_mse_loss(
                    delta_true, delta_pred, unc, mode=loss_mode
                ).item()
        val_loss /= len(val_loader)
        stopper.step(val_loss, predictor)

    stopper.restore(predictor)
    return predictor
# =============================================================================
# Deterministic reduction fitting
# =============================================================================
def _fit_deterministic_reduction(method, X_train, n_components):
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=SEED)
    elif method == "kpca":
        reducer = KernelPCA(
            n_components=n_components, kernel="rbf",
            random_state=SEED, fit_inverse_transform=False,
        )
    elif method == "ica":
        reducer = FastICA(
            n_components=n_components, random_state=SEED, max_iter=500,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    reducer.fit(X_train)
    return reducer


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


def _evaluate_deterministic_variant(reducer, predictor, test_loader, loss_mode="factor"):
    predictor.eval()
    all_preds, all_targets, all_uncs = [], [], []
    def _encode(x):
        z = reducer.transform(x.cpu().numpy())
        return torch.tensor(z, dtype=torch.float32, device=DEVICE)
    with torch.inference_mode():
        for batch in test_loader:
            x_t, c_t, y_t, x_t1, c_t1, y_t1 = [b.to(DEVICE) for b in batch]
            z_t, z_t1 = _encode(x_t), _encode(x_t1)
            inp = torch.cat([z_t, c_t, z_t1, c_t1], dim=1)
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
# Per-split variant runners
# =============================================================================
def _make_loaders(dataset, train_idx, val_idx, test_idx):
    return (
        DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False),
    )


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
    X_train = dataset.input_df[train_idx].cpu().numpy()
    reducer = _fit_deterministic_reduction(method, X_train, REDUCTION_DIM)
    predictor = _build_predictor(
        input_dim=2 * (REDUCTION_DIM + context_dim), uncertainty=True
    ).to(DEVICE)
    train_l, val_l, test_l = _make_loaders(dataset, train_idx, val_idx, test_idx)
    predictor = _train_predictor_with_deterministic_latent(
        reducer, predictor, train_l, val_l, dataset, EPOCHS_PREDICTOR
    )
    return _evaluate_deterministic_variant(reducer, predictor, test_l)


def run_vae_variant(dataset, train_idx, val_idx, test_idx, include_context):
    input_dim = dataset.input_df.shape[1]
    context_dim = dataset.context_df.shape[1]
    effective_context_dim = context_dim if include_context else 0

    train_l, val_l, test_l = _make_loaders(dataset, train_idx, val_idx, test_idx)

    # Step 1: Train VAE
    vae = _build_vae(input_dim).to(DEVICE)
    vae = _train_vae(vae, train_l, val_l, EPOCHS_VAE, extract_x=lambda b: b[0])

    # Step 2: Build and train predictor with frozen VAE
    vae_config = load_config("config/models/vae_config.yaml")
    latent_dim = vae_config.vae_latent_dim
    predictor = _build_predictor(
        input_dim=2 * (latent_dim + effective_context_dim), uncertainty=True
    ).to(DEVICE)
    full_model = FullPredictionModel(vae=vae, predictor=predictor).to(DEVICE)

    if not include_context:
        full_model = _train_predictor_no_context(
            full_model, train_l, val_l, EPOCHS_PREDICTOR
        )
        return _evaluate_vae_no_context(full_model, test_l)
    else:
        full_model = _train_predictor_with_vae(
            full_model, train_l, val_l, EPOCHS_PREDICTOR
        )
        return _evaluate_vae_variant(full_model, test_l)


# =============================================================================
# Single split orchestrator
# =============================================================================
def run_single_split(split_id, dataset):
    total = len(dataset)
    val_size = int(total * VAL_RATIO)
    test_size = int(total * TEST_RATIO)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(SEED + split_id)
    indices = torch.randperm(total, generator=generator).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    results = []

    variants = [
        ("baseline",       lambda: run_baseline(dataset, train_idx, val_idx, test_idx)),
        ("pca",            lambda: run_deterministic_latent("pca", dataset, train_idx, val_idx, test_idx)),
        ("kpca",           lambda: run_deterministic_latent("kpca", dataset, train_idx, val_idx, test_idx)),
        ("ica",            lambda: run_deterministic_latent("ica", dataset, train_idx, val_idx, test_idx)),
        ("vae_no_context", lambda: run_vae_variant(dataset, train_idx, val_idx, test_idx, include_context=False)),
        ("vae_final",      lambda: run_vae_variant(dataset, train_idx, val_idx, test_idx, include_context=True)),
    ]

    for name, runner in variants:
        print(f"  Split {split_id}: {name}...")
        metrics = runner()
        metrics["split"] = split_id
        metrics["variant"] = name
        results.append(metrics)

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
    print("CROSS-VALIDATION MODEL COMPARISON (Table 3)")
    print("=" * 70)
    print(f"Splits: {N_SPLITS}")
    print(f"Train/Val/Test: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    print(f"Device: {DEVICE}")
    print(f"Latent dim: {LATENT_DIM}")
    print(f"Epochs — VAE: {EPOCHS_VAE}, Predictor: {EPOCHS_PREDICTOR}, "
          f"Direct: {EPOCHS_DIRECT}")
    print(f"Early-stop window: {STOPPER_WINDOW}")
    print()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = get_or_create_dataset()
    print(f"Dataset: {len(dataset)} samples")
    print(f"  Input features: {dataset.input_df.shape[1]}")
    print(f"  Context features: {dataset.context_df.shape[1]}")
    print(f"  Emission sectors: {dataset.emi_df.shape[1]}")
    print()

    all_results = []
    for split_id in range(N_SPLITS):
        print(f"\n{'=' * 50}")
        print(f"SPLIT {split_id + 1}/{N_SPLITS}")
        print(f"{'=' * 50}")
        split_results = run_single_split(split_id, dataset)
        all_results.extend(split_results)

        if (split_id + 1) % 10 == 0:
            df_so_far = pd.DataFrame(all_results)
            print(f"\n--- Progress after {split_id + 1} splits ---")
            for variant in df_so_far["variant"].unique():
                sub = df_so_far[df_so_far["variant"] == variant]
                print(
                    f"  {variant:>20}: MAE={sub['mae'].mean():.4f} "
                    f"MSE={sub['mse'].mean():.4f} "
                    f"ρ={sub['corr'].mean():.4f} "
                    f"ℒ={sub['loss_pred'].mean():.4f}"
                )

    results_df = pd.DataFrame(all_results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "table3_cv_comparison.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")

    print("\n" + "=" * 70)
    print("TABLE 3: Performance comparison across model variants")
    print(f"mean (std) across {N_SPLITS} random "
          f"{TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%} splits.")
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