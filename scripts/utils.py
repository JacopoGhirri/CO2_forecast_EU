"""
Utility functions for EU emission forecasting.

This module provides helper functions for:
- Dataset serialization/deserialization
- Configuration loading from YAML
- Neural network initialization
- Gradient monitoring and NaN handling

Example:
    >>> from scripts.utils import load_config, init_weights
    >>> config = load_config("config/models/vae_config.yaml")
    >>> model.apply(init_weights)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import yaml


def save_dataset(dataset: Any, filepath: str | Path) -> None:
    """
    Saves a dataset object to disk using pickle.

    Args:
        dataset: Dataset object to save (e.g., DatasetUnified instance).
        filepath: Path where the dataset will be saved.

    Example:
        >>> save_dataset(train_dataset, "data/pytorch_datasets/train.pkl")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(filepath: str | Path) -> Any:
    """
    Loads a dataset object from disk.

    Args:
        filepath: Path to the saved dataset file.

    Returns:
        The loaded dataset object.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        >>> dataset = load_dataset("data/pytorch_datasets/train.pkl")
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_config(yaml_path: str | Path) -> SimpleNamespace:
    """
    Loads configuration from a YAML file into a SimpleNamespace.

    This function handles W&B (Weights & Biases) sweep configuration format,
    extracting the 'value' field from each parameter and filtering out
    W&B-specific metadata.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        SimpleNamespace with configuration values accessible as attributes.

    Example:
        >>> config = load_config("config/models/vae_config.yaml")
        >>> print(config.vae_latent_dim)  # 10
        >>> print(config.vae_activation)  # 'gelu'

    Note:
        The YAML file is expected to have the W&B format:
        ```yaml
        param_name:
          value: actual_value
        ```
    """
    with open(yaml_path) as f:
        raw_config = yaml.safe_load(f)

    # Extract values from W&B format, filtering out metadata
    clean_config = {
        key: value["value"]
        for key, value in raw_config.items()
        if not key.startswith("_") and key != "wandb_version"
    }

    return SimpleNamespace(**clean_config)


def check_nan_gradients(model: nn.Module) -> None:
    """
    Checks for NaN gradients and replaces them with zeros.

    This is a safety mechanism to prevent training crashes due to
    numerical instability. NaN gradients can occur with:
    - Very large learning rates
    - Exploding gradients in deep networks
    - Division by zero in custom loss functions

    Args:
        model: PyTorch model to check.

    Side effects:
        Replaces any NaN values in gradients with 0.

    Example:
        >>> loss.backward()
        >>> check_nan_gradients(model)  # Fix any NaN gradients
        >>> optimizer.step()

    Warning:
        While this prevents crashes, frequent NaN gradients indicate
        underlying issues that should be addressed (e.g., learning rate,
        loss function stability, data preprocessing).
    """
    for param in model.parameters():
        if param.grad is not None:
            nan_mask = torch.isnan(param.grad)
            if nan_mask.any():
                param.grad[nan_mask] = 0


def init_weights(module: nn.Module) -> None:
    """
    Initializes neural network weights using orthogonal initialization.

    Orthogonal initialization helps maintain gradient magnitude during
    forward and backward passes, which is particularly beneficial for
    deep networks with residual connections.

    Args:
        module: PyTorch module to initialize.

    Side effects:
        - Linear layers: Orthogonal weight initialization, bias = 0.01
        - Other layers: Unchanged

    Example:
        >>> model = VAEModel(encoder, decoder)
        >>> model.apply(init_weights)

    Note:
        This function is designed to be used with model.apply().
        It only affects nn.Linear layers.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of parameters with requires_grad=True.

    Example:
        >>> model = VAEModel(encoder, decoder)
        >>> print(f"Model has {count_parameters(model):,} parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_module(module: nn.Module) -> None:
    """
    Freezes all parameters in a module (sets requires_grad=False).

    Useful for transfer learning or when training only part of a model.

    Args:
        module: PyTorch module to freeze.

    Example:
        >>> freeze_module(model.encoder)  # Freeze encoder during predictor training
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """
    Unfreezes all parameters in a module (sets requires_grad=True).

    Args:
        module: PyTorch module to unfreeze.

    Example:
        >>> unfreeze_module(model.encoder)  # Allow encoder fine-tuning
    """
    for param in module.parameters():
        param.requires_grad = True
