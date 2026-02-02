import pickle
import torch
import yaml
from types import SimpleNamespace
import torch.nn as nn

def save_datasets(full_dataset, filename='datasets.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(full_dataset, f)


def load_datasets(filename='datasets.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def check_NaN_grad(model):
    for param in model.parameters():
        if param.grad is not None:
            # Replace NaN values with 0
            is_nan = torch.isnan(param.grad)
            if is_nan.any():
                param.grad[is_nan] = 0


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # remove wandb-specific fields
    clean_config = {
        k: v["value"] for k, v in raw_config.items()
        if not k.startswith("_") and k != "wandb_version"
    }

    # convert to namespace
    return SimpleNamespace(**clean_config)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)