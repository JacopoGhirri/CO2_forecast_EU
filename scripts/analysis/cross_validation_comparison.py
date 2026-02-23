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

# Automatic mixed precision — gives ~1.5-2× speedup on CUDA for free
USE_AMP = DEVICE == "cuda"

# DataLoader workers and pin_memory are configured after dataset loading
# (see _configure_data_loading) because they depend on whether the dataset
# tensors reside on CPU or CUDA.  CUDA tensors cannot be pinned and cannot
# be shared across worker processes on Windows.
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
# Config files are read from disk once and reused across all splits/variants,
# avoiding repeated YAML parsing (~600 splits × variants × stages).

_config_cache: dict[str, object] = {}


def _get_config(path: str) -> object:
    """Return a cached config namespace, loading from YAML only on first call.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        object: Parsed config namespace (same type as ``load_config`` returns).
    """
    if path not in _config_cache:
        _config_cache[path] = load_config(path)
    return _config_cache[path]


def _vae_config():
    """Return the cached VAE configuration namespace."""
    return _get_config("config/models/vae_config.yaml")


def _pred_config():
    """Return the cached predictor configuration namespace."""
    return _get_config("config/models/co2_predictor_config.yaml")


# =============================================================================
# EarlyStopper
# =============================================================================
class EarlyStopper:
    """Tracks smoothed validation loss and stores best model weights.

    Uses a rolling window to smooth noisy validation curves and saves
    a deep copy of the model state dict whenever the smoothed loss
    improves.  Matches the early-stopping logic in train_vae.py and
    train_predictor.py (deque with maxlen=STOPPER_WINDOW).

    Args:
        window (int): Size of the rolling average window.  Defaults to
            ``STOPPER_WINDOW``.

    Attributes:
        history (deque[float]): Rolling buffer of recent validation losses.
        best_smooth (float): Lowest smoothed loss observed so far.
        best_weights (dict | None): State dict snapshot corresponding to
            ``best_smooth``, stored on CPU.
    """

    def __init__(self, window: int = STOPPER_WINDOW):
        self.history: deque[float] = deque(maxlen=window)
        self.best_smooth: float = float("inf")
        self.best_weights: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> None:
        """Record a validation loss and update best weights if improved.

        Args:
            val_loss (float): Raw (unsmoothed) validation loss for the
                current epoch.
            model (nn.Module): Model whose state dict will be saved if the
                smoothed loss improves.
        """
        self.history.append(val_loss)
        smooth = sum(self.history) / len(self.history)
        if smooth < self.best_smooth:
            self.best_smooth = smooth
            self.best_weights = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    def restore(self, model: nn.Module) -> None:
        """Load the best weights back into the model and move to DEVICE.

        Args:
            model (nn.Module): Model to restore.  Modified in-place.
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            model.to(DEVICE)


# =============================================================================
# Dataset loading
# =============================================================================
def get_or_create_dataset() -> DatasetPrediction:
    """Load the cached prediction dataset, or create it from raw CSVs.

    If the cached dataset is a ``DatasetUnified`` (not ``DatasetPrediction``),
    it is converted by swapping its ``__class__``, since ``DatasetPrediction``
    only overrides ``__getitem__`` to return the 6-tuple
    ``(x_t, c_t, y_t, x_{t-1}, c_{t-1}, y_{t-1})``.

    Returns:
        DatasetPrediction: The loaded (or freshly created) prediction dataset
            with normalised inputs and paired time-step rows.
    """
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
    """Move dataset tensors to CPU if they reside on CUDA, and configure
    DataLoader settings accordingly.

    ``pin_memory=True`` requires CPU tensors and ``num_workers > 0``
    cannot share CUDA tensors across processes (especially on Windows).
    This function detects whether the dataset was loaded/created with
    CUDA tensors and moves them to CPU so that the DataLoader can
    safely use pinned memory and multi-process loading.

    After calling this function the module-level ``PIN_MEMORY`` and
    ``NUM_WORKERS`` globals are set to appropriate values.

    Args:
        dataset (DatasetPrediction): The dataset to check/fix.
            Modified in-place (tensor attributes are replaced with
            CPU copies if originally on CUDA).
    """
    global PIN_MEMORY, NUM_WORKERS

    # Check if any tensor attribute is on CUDA
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
        # Also move any paired/offset tensors if they exist
        for attr in dir(dataset):
            if not attr.startswith("_"):
                val = getattr(dataset, attr, None)
                if isinstance(val, torch.Tensor) and val.is_cuda:
                    setattr(dataset, attr, val.cpu())

    # Now that data is on CPU, we can safely use pin_memory and workers
    if DEVICE == "cuda":
        PIN_MEMORY = True
        NUM_WORKERS = 4
    else:
        PIN_MEMORY = False
        NUM_WORKERS = 0


# =============================================================================
# Model builders
# =============================================================================
def _build_vae(input_dim: int) -> VAEModel:
    """Build a fresh VAE with the architecture specified in vae_config.yaml.

    Constructs an Encoder and Decoder with matching block structure,
    wraps them in a ``VAEModel``, and applies ``init_weights``.

    Args:
        input_dim (int): Dimensionality of the input feature vector x_t.

    Returns:
        VAEModel: Randomly initialised VAE (not yet moved to DEVICE).
    """
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
    """Build a fresh emission predictor from co2_predictor_config.yaml.

    Args:
        input_dim (int): Dimensionality of the predictor input vector.
            Depends on the variant (e.g. 2*(latent_dim + context_dim) for
            VAE+Context, 2*(input_dim + context_dim) for Baseline).
        uncertainty (bool): Whether the predictor outputs an uncertainty
            head alongside the mean prediction.  Defaults to ``True``.

    Returns:
        EmissionPredictor: Randomly initialised predictor (not yet moved
            to DEVICE).
    """
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
    """Build the predictor optimizer matching train_predictor.py.

    Reads the optimizer class (adam / adamw / radam), learning rate, and
    weight decay from the predictor config.  Supports both a flat parameter
    iterable and a list of param-group dicts (for multi-LR setups).

    Args:
        params_or_groups: Either an iterable of ``torch.nn.Parameter`` or a
            list of param-group dicts (each with ``"params"`` and ``"lr"``
            keys) to pass to the optimizer constructor.
        pred_config (optional): Pre-loaded predictor config namespace.  If
            ``None``, loads from ``co2_predictor_config.yaml``.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.
    """
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
    """Train the VAE on reconstruction + KL loss.

    Optimizer and loss weights match train_vae.py exactly:
      - AdamW with lr=vae_lr, weight_decay=vae_weight_decay, eps=1e-6
      - Loss = vae_wr * reconstruction + vae_wd * KL divergence
      - Gradient clipping at max_norm=1.0
      - Model selection on smoothed *reconstruction* loss (window=50)

    Args:
        vae (VAEModel): Fresh VAEModel already on DEVICE.
        train_loader (DataLoader): Training DataLoader yielding dataset
            tuples from ``DatasetPrediction``.
        val_loader (DataLoader): Validation DataLoader.
        epochs (int): Number of training epochs.
        extract_x (callable): Function that takes a batch tuple and returns
            the input tensor ``x_t`` to reconstruct (e.g. ``lambda b: b[0]``).

    Returns:
        VAEModel: The VAE with best weights (by smoothed reconstruction loss)
            restored and placed on DEVICE.
    """
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

        vae.eval()
        val_recon = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                x = extract_x(batch).to(DEVICE, non_blocking=PIN_MEMORY)
                with torch.amp.autocast(DEVICE, enabled=USE_AMP):
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


def _train_predictor_with_vae(
    full_model, train_loader, val_loader, epochs, loss_mode="factor"
):
    """Train the emission predictor while keeping VAE weights frozen.

    Optimizer setup matches train_predictor.py exactly:
      - Frozen VAE encoder/decoder params still in optimizer groups at
        lr * 1e-3 (reproducing the training script's param-group structure).
      - Predictor params at full lr from co2_predictor_config.yaml.
      - Optimizer class and weight_decay from config.
      - Gradient clipping at max_norm=1.0.
      - Model selection on smoothed validation loss (window=50).

    The forward pass uses ``FullPredictionModel.forward()`` which encodes
    x_t and x_{t-1} via the frozen VAE, concatenates the latent vectors
    with context, and predicts emission deltas.

    Args:
        full_model (FullPredictionModel): Model containing the pre-trained
            VAE and a fresh predictor, already on DEVICE.
        train_loader (DataLoader): Training DataLoader yielding 6-tuples.
        val_loader (DataLoader): Validation DataLoader.
        epochs (int): Number of training epochs.
        loss_mode (str): Mode for ``uncertainty_aware_mse_loss``.
            Defaults to ``"factor"``.

    Returns:
        FullPredictionModel: Model with best weights restored.
    """
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
    """Train the predictor without context variables.

    This variant tests the contribution of the probabilistic latent space
    alone.  The predictor receives ``[z_t, z_{t-1}]`` (no context
    concatenated).  The VAE encoder is frozen and kept in eval mode to
    disable dropout / stochastic layers.

    Only predictor parameters are updated; the optimizer uses the same
    class, learning rate, and weight decay as the full-context variant.

    Args:
        full_model (FullPredictionModel): Model containing the pre-trained
            VAE and a fresh predictor (input_dim = 2 * latent_dim),
            already on DEVICE.
        train_loader (DataLoader): Training DataLoader yielding 6-tuples.
        val_loader (DataLoader): Validation DataLoader.
        epochs (int): Number of training epochs.

    Returns:
        FullPredictionModel: Model with best weights restored.
    """
    for p in full_model.vae.parameters():
        p.requires_grad = False

    pred_config = _pred_config()
    optimizer = _get_pred_optimizer(full_model.predictor.parameters(), pred_config)
    stopper = EarlyStopper(window=STOPPER_WINDOW)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    def _forward_no_ctx(batch):
        """Encode x_t and x_{t-1}, concatenate latents without context, and predict."""
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
        full_model.vae.eval()  # keep encoder in eval (no dropout)
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
    """Train the Baseline (direct) predictor that bypasses any latent space.

    The predictor receives the raw concatenation ``[x_t, c_t, x_{t-1}, c_{t-1}]``
    and predicts emission deltas directly.  Uses the same optimizer config
    (class, lr, weight_decay) as the predictor training script.

    Args:
        predictor (EmissionPredictor): Fresh predictor on DEVICE with
            ``input_dim = 2 * (input_dim + context_dim)``.
        train_loader (DataLoader): Training DataLoader yielding 6-tuples.
        val_loader (DataLoader): Validation DataLoader.
        epochs (int): Number of training epochs.
        context_dim (int): Dimensionality of the context vector (used only
            for documentation / debugging; not directly consumed).
        loss_mode (str): Mode for ``uncertainty_aware_mse_loss``.
            Defaults to ``"factor"``.

    Returns:
        EmissionPredictor: Predictor with best weights restored.
    """
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
    """Fit a deterministic dimensionality reduction on training data.

    Args:
        method (str): One of ``"pca"``, ``"kpca"``, or ``"ica"``.
        X_train (np.ndarray): Training input array, shape ``(N_train, D)``.
        n_components (int): Target latent dimensionality.

    Returns:
        sklearn transformer: Fitted reducer with a ``.transform()`` method.

    Raises:
        ValueError: If ``method`` is not one of the supported strings.
    """
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
# Evaluation helpers
# =============================================================================
def _compute_metrics(preds, targets, uncs, loss_mode):
    """Compute per-sector and aggregate evaluation metrics.

    Metrics computed for each of the ``n_sectors`` output columns:
      - MAE (Mean Absolute Error)
      - MSE (Mean Squared Error)
      - Pearson ρ (correlation between predicted and true deltas)

    Additionally computes:
      - Aggregate MAE, MSE, and ρ (mean across sectors)
      - ℒ_pred (uncertainty-aware prediction loss over all sectors)

    Args:
        preds (np.ndarray): Predicted emission deltas, shape ``(N, n_sectors)``.
        targets (np.ndarray): True emission deltas, shape ``(N, n_sectors)``.
        uncs (np.ndarray): Predicted uncertainties, shape ``(N, n_sectors)``.
        loss_mode (str): Mode for ``uncertainty_aware_mse_loss``.

    Returns:
        dict: Metric dictionary with keys ``mae_0`` … ``mae_{n-1}``,
            ``mse_0`` … ``mse_{n-1}``, ``corr_0`` … ``corr_{n-1}``,
            ``mae``, ``mse``, ``corr``, and ``loss_pred``.
    """
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
    """Evaluate a VAE-based variant (VAE + Context) on the test set.

    Runs inference through ``FullPredictionModel.forward()`` which encodes
    x_t and x_{t-1}, concatenates latents with context, and predicts
    emission deltas.

    Args:
        full_model (FullPredictionModel): Trained model on DEVICE.
        test_loader (DataLoader): Test DataLoader yielding 6-tuples.
        loss_mode (str): Mode for ``uncertainty_aware_mse_loss``.
            Defaults to ``"factor"``.

    Returns:
        dict: Metric dictionary (see ``_compute_metrics``).
    """
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
    """Evaluate the VAE no-context variant on the test set.

    Manually encodes x_t and x_{t-1} through the frozen VAE encoder,
    concatenates ``[z_t, z_{t-1}]`` (without context), and predicts
    emission deltas via the predictor head.

    Args:
        full_model (FullPredictionModel): Trained model on DEVICE.
        test_loader (DataLoader): Test DataLoader yielding 6-tuples.

    Returns:
        dict: Metric dictionary (see ``_compute_metrics``).
    """
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
    """Evaluate the Baseline (direct) variant on the test set.

    Concatenates raw inputs ``[x_t, c_t, x_{t-1}, c_{t-1}]`` and predicts
    emission deltas directly.

    Args:
        predictor (EmissionPredictor): Trained predictor on DEVICE.
        test_loader (DataLoader): Test DataLoader yielding 6-tuples.
        loss_mode (str): Mode for ``uncertainty_aware_mse_loss``.
            Defaults to ``"factor"``.

    Returns:
        dict: Metric dictionary (see ``_compute_metrics``).
    """
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


def _make_loaders(dataset, train_idx, val_idx, test_idx):
    """Create train, validation, and test DataLoaders for a single split.

    Args:
        dataset (DatasetPrediction): Full dataset.
        train_idx (list[int]): Training sample indices.
        val_idx (list[int]): Validation sample indices.
        test_idx (list[int]): Test sample indices.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train (shuffled),
            validation, and test DataLoaders with ``BATCH_SIZE``.
    """
    return (
        DataLoader(
            Subset(dataset, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
        DataLoader(
            Subset(dataset, val_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
        DataLoader(
            Subset(dataset, test_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
    )


def run_baseline(dataset, train_idx, val_idx, test_idx):
    """Run the Baseline (no latent space) variant for one split.

    Builds a predictor that takes the raw concatenation
    ``[x_t, c_t, x_{t-1}, c_{t-1}]`` and predicts emission deltas
    directly, bypassing any dimensionality reduction.

    Args:
        dataset (DatasetPrediction): Full dataset.
        train_idx (list[int]): Training sample indices.
        val_idx (list[int]): Validation sample indices.
        test_idx (list[int]): Test sample indices.

    Returns:
        dict: Metric dictionary (see ``_compute_metrics``).
    """
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
    """Run a deterministic latent space variant (PCA / KPCA / ICA) for one split.

    Fits the dimensionality reduction on training inputs only, then
    **pre-computes** the reduced representations for *all* rows in
    ``dataset.input_df`` (since ``DatasetPrediction.__getitem__`` pairs
    row i with row i-1, any row can appear as x_{t-1}).

    This avoids the expensive CPU ↔ numpy round-trip on every batch of
    every epoch (~6000 transform calls → 1 bulk transform).

    The predictor trains on ``[z_t, c_t, z_{t-1}, c_{t-1}]`` using a
    lightweight ``_PreReducedDataset`` that swaps ``input_df`` for the
    pre-reduced tensor.

    Args:
        method (str): One of ``"pca"``, ``"kpca"``, or ``"ica"``.
        dataset (DatasetPrediction): Full dataset.
        train_idx (list[int]): Training sample indices.
        val_idx (list[int]): Validation sample indices.
        test_idx (list[int]): Test sample indices.

    Returns:
        dict: Metric dictionary (see ``_compute_metrics``).
    """
    context_dim = dataset.context_df.shape[1]

    # Fit reducer on training inputs only (no data leakage)
    X_train = dataset.input_df[train_idx].cpu().numpy()
    reducer = _fit_deterministic_reduction(method, X_train, REDUCTION_DIM)

    # Pre-compute reduced features for ALL rows (needed because the
    # DatasetPrediction pairing can reference any row as x_{t-1}).
    Z_all = torch.tensor(
        reducer.transform(dataset.input_df.cpu().numpy()),
        dtype=torch.float32,
    )

    # Create a shallow wrapper that substitutes input_df with Z_all
    pre_ds = _PreReducedDataset(dataset, Z_all)

    train_l = DataLoader(Subset(pre_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_l = DataLoader(Subset(pre_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_l = DataLoader(Subset(pre_ds, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    # Build and train predictor — uses the same direct training loop
    # since the data is already reduced (no per-batch transform needed).
    predictor = _build_predictor(
        input_dim=2 * (REDUCTION_DIM + context_dim), uncertainty=True
    ).to(DEVICE)
    predictor = _train_direct_predictor(
        predictor, train_l, val_l, EPOCHS_PREDICTOR, context_dim
    )
    return _evaluate_direct_variant(predictor, test_l)


class _PreReducedDataset:
    """Lightweight dataset wrapper that substitutes pre-computed reduced features.

    Yields the same 6-tuple as ``DatasetPrediction`` but replaces the
    raw input features (x_t, x_{t-1}) with their pre-computed reduced
    counterparts (z_t, z_{t-1}).  Context and emission tensors are
    passed through unchanged from the original dataset.

    The wrapper directly indexes into a pre-computed ``Z_all`` tensor
    using the same row-pairing logic as ``DatasetPrediction``.  Since
    ``Z_all`` covers all rows in ``input_df``, both x_t and x_{t-1}
    are correctly substituted regardless of the internal pairing.

    This replaces ~6000 per-batch sklearn ``transform()`` calls per
    variant with a single bulk transform at split initialisation.

    Args:
        original (DatasetPrediction): The original dataset (provides
            context, emission data, and the time-step pairing logic).
        Z_all (torch.Tensor): Pre-reduced features for all rows in
            ``original.input_df``, shape ``(N_total, REDUCTION_DIM)``.
    """

    def __init__(self, original: DatasetPrediction, Z_all: torch.Tensor):
        self.original = original
        self.Z_all = Z_all
        # Temporarily swap input_df so the original __getitem__ returns
        # reduced features directly.  We store the original to restore later.
        self._original_input_df = original.input_df

    def __len__(self) -> int:
        """Return the length of the original dataset."""
        return len(self.original)

    def __getitem__(self, idx: int):
        """Return ``(z_t, c_t, y_t, z_{t-1}, c_{t-1}, y_{t-1})`` tuple.

        Temporarily swaps the original dataset's ``input_df`` with
        ``Z_all``, delegates to the original ``__getitem__`` (which
        preserves the time-step pairing logic), then restores the
        original ``input_df``.

        Args:
            idx (int): Sample index (same indexing as the original dataset).

        Returns:
            tuple[Tensor, ...]: 6-tuple with reduced features in place of
                raw inputs.
        """
        # Swap input_df → Z_all, call original __getitem__, swap back
        self.original.input_df = self.Z_all
        try:
            result = self.original[idx]
        finally:
            self.original.input_df = self._original_input_df
        return result


# =============================================================================
# Single split orchestrator
# =============================================================================
def run_single_split(split_id, dataset):
    """Run all six model variants on a single train/val/test split.

    Generates a deterministic random partition using ``SEED + split_id``
    and trains each variant from scratch on the same split.

    The VAE is trained **once** and shared between the ``vae_no_context``
    and ``vae_final`` variants (they differ only in whether context is
    concatenated with the latents).  This halves the most expensive
    training stage per split.

    Variant order:
      1. Baseline (direct)
      2. PCA
      3. KPCA
      4. ICA
      5. VAE (no context)  }  share a single VAE
      6. VAE + Context      }

    Args:
        split_id (int): Split index (used as seed offset for reproducibility).
        dataset (DatasetPrediction): Full dataset.

    Returns:
        list[dict]: List of 6 metric dictionaries, one per variant.  Each
            dict additionally contains ``"split"`` and ``"variant"`` keys.
    """
    total = len(dataset)
    val_size = int(total * VAL_RATIO)
    test_size = int(total * TEST_RATIO)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(SEED + split_id)
    indices = torch.randperm(total, generator=generator).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    # Shared DataLoaders — created once, reused across applicable variants
    train_l, val_l, test_l = _make_loaders(dataset, train_idx, val_idx, test_idx)

    results = []

    def _record(name, metrics):
        """Append metrics with variant and split metadata."""
        metrics["split"] = split_id
        metrics["variant"] = name
        results.append(metrics)
        # Free GPU memory between variants
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- 1. Baseline ---
    print(f"  Split {split_id}: baseline...")
    _record("baseline", run_baseline(dataset, train_idx, val_idx, test_idx))

    # --- 2-4. Deterministic reductions ---
    for method in ("pca", "kpca", "ica"):
        print(f"  Split {split_id}: {method}...")
        _record(
            method,
            run_deterministic_latent(method, dataset, train_idx, val_idx, test_idx),
        )

    # --- 5-6. VAE variants (shared VAE training) ---
    print(f"  Split {split_id}: training shared VAE...")
    input_dim = dataset.input_df.shape[1]
    context_dim = dataset.context_df.shape[1]
    vae = _build_vae(input_dim).to(DEVICE)
    vae = _train_vae(vae, train_l, val_l, EPOCHS_VAE, extract_x=lambda b: b[0])

    # Deep-copy VAE weights so the two predictor training runs don't interfere
    vae_weights = {k: v.cpu().clone() for k, v in vae.state_dict().items()}

    vae_cfg = _vae_config()
    latent_dim = vae_cfg.vae_latent_dim

    # 5. VAE no context
    print(f"  Split {split_id}: vae_no_context...")
    vae_nc = _build_vae(input_dim).to(DEVICE)
    vae_nc.load_state_dict(vae_weights)
    vae_nc.to(DEVICE)
    pred_nc = _build_predictor(input_dim=2 * latent_dim, uncertainty=True).to(DEVICE)
    fm_nc = FullPredictionModel(vae=vae_nc, predictor=pred_nc).to(DEVICE)
    fm_nc = _train_predictor_no_context(fm_nc, train_l, val_l, EPOCHS_PREDICTOR)
    _record("vae_no_context", _evaluate_vae_no_context(fm_nc, test_l))
    del fm_nc, vae_nc, pred_nc  # free memory

    # 6. VAE + Context (Final)
    print(f"  Split {split_id}: vae_final...")
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
    """Format the results DataFrame as an ASCII table matching Table 3 layout.

    Columns correspond to the six model variants.  Rows show MAE, MSE,
    Pearson ρ, and ℒ_pred as ``mean(std)`` across all splits.

    Args:
        summary (pd.DataFrame): Long-format DataFrame with columns
            ``variant``, ``mae``, ``mse``, ``corr``, ``loss_pred``, etc.

    Returns:
        str: Multi-line formatted ASCII table string.
    """
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
    """Main entry point for the cross-validation model comparison.

    Workflow:
      1. Load or create the unified prediction dataset.
      2. Resume from checkpoint if a previous run was interrupted.
      3. For each of ``N_SPLITS`` random partitions:
         a. Train and evaluate all 6 model variants from scratch.
         b. Collect per-split metrics.
         c. Save checkpoint CSV (crash recovery).
         d. Free GPU memory between splits.
      4. Print progress every 10 splits.
      5. Aggregate results into mean ± std summary.
      6. Save full per-split CSV, remove checkpoint, and print Table 3.
    """
    print("=" * 70)
    print("CROSS-VALIDATION MODEL COMPARISON (Table 3)")
    print("=" * 70)
    print(f"Splits: {N_SPLITS}")
    print(f"Train/Val/Test: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    print(f"Device: {DEVICE}")
    print(f"Latent dim: {LATENT_DIM}")
    print(
        f"Epochs — VAE: {EPOCHS_VAE}, Predictor: {EPOCHS_PREDICTOR}, "
        f"Direct: {EPOCHS_DIRECT}"
    )
    print(f"Early-stop window: {STOPPER_WINDOW}")
    print(f"Mixed precision (AMP): {USE_AMP}")
    print(f"DataLoader workers: {NUM_WORKERS}, pin_memory: {PIN_MEMORY}")
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
    start_split = 0

    # --- Resume from checkpoint if available ---
    if CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
        all_results = checkpoint_df.to_dict("records")
        completed_splits = set(checkpoint_df["split"].unique())
        start_split = max(completed_splits) + 1
        print(
            f"  Found {len(completed_splits)} completed splits, "
            f"resuming from split {start_split}"
        )

    for split_id in range(start_split, N_SPLITS):
        print(f"\n{'=' * 50}")
        print(f"SPLIT {split_id + 1}/{N_SPLITS}")
        print(f"{'=' * 50}")
        split_results = run_single_split(split_id, dataset)
        all_results.extend(split_results)

        # Free GPU memory between splits
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Save checkpoint after every split (crash recovery)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).to_csv(CHECKPOINT_PATH, index=False)

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

    # Remove checkpoint after successful completion
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"Checkpoint removed: {CHECKPOINT_PATH}")

    print("\n" + "=" * 70)
    print("TABLE 3: Performance comparison across model variants")
    print(
        f"mean (std) across {N_SPLITS} random "
        f"{TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%} splits."
    )
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
