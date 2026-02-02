import multiprocessing as mp

# Set the start method at the very beginning of the script
mp.set_start_method("spawn", force=True)

import os
import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.data.output_configs import output_configs
from Scripts.Elements.Datasets import Dataset_unified
from Scripts.Elements.Models import Decoder, Encoder, VAE_loss_function, VAEModel
from Scripts.utils import (
    check_NaN_grad,
    init_weights,
    load_config,
    load_datasets,
    save_datasets,
)

random.seed(0)
torch.manual_seed(0)


def main():

    config = load_config("config/models/VAE_config.yaml")
    batch_size = 128
    vae_num_epochs = 5000
    scaling_type = "normalization"  # 'maxmin' or 'normalization'

    vae_activation = config.vae_activation

    vae_latent_dim = config.vae_latent_dim
    vae_num_blocks = config.vae_num_blocks
    vae_dim_blocks = config.vae_dim_blocks
    vae_normalization = config.vae_normalization
    vae_dropouts = config.vae_dropouts
    vae_input_dropout = config.vae_input_dropouts
    vae_learning_rate = config.vae_lr
    vae_weight_decay = config.vae_weight_decay
    vae_wr = config.vae_wr
    vae_wd = config.vae_wd
    vae_optimizer = config.vae_optimizer

    dataset_filename = "Data/pytorch_datasets/unified_dataset.pkl"
    nested_variable_file = "config/data/variable_selection.pkl"
    with open(nested_variable_file, "r") as file:
        nested_variables = [line.strip() for line in file]

    if os.path.exists(dataset_filename):
        full_dataset = load_datasets(dataset_filename)
        print("Dataset loaded from file.")
    else:
        full_dataset = Dataset_unified(
            path_csvs="Data/full_timeseries/",
            output_configs=output_configs,
            select_years=np.arange(2010, 2023 + 1),
            select_geo=[
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
            ],
            nested_variables=nested_variables,
            with_cuda=True,
            precomputed_scaling_params=None,
            scaling_type=scaling_type,
        )
        # Save the datasets to file
        save_datasets(full_dataset, dataset_filename)
        print("Dataset created and saved to file.")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [0.85, 0.15]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )  # , num_workers=16)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    print("Data setup done")
    input_dim = len(full_dataset.input_variable_names)

    enc = Encoder(
        input_dim=input_dim,
        latent_dim=vae_latent_dim,
        num_blocks=vae_num_blocks,
        dim_blocks=vae_dim_blocks,
        activation=vae_activation,
        normalization=vae_normalization,
        dropouts=vae_dropouts,
        input_dropout=vae_input_dropout,
    ).cuda()
    dec = Decoder(
        input_dim=input_dim,
        latent_dim=vae_latent_dim,
        num_blocks=vae_num_blocks,
        dim_blocks=vae_dim_blocks,
        activation=vae_activation,
        normalization=vae_normalization,
        dropouts=vae_dropouts,
    ).cuda()
    vae_model = VAEModel(enc, dec).cuda()
    vae_model.apply(init_weights)

    vae_criterion = VAE_loss_function

    if vae_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            vae_model.parameters(),
            lr=vae_learning_rate,
            weight_decay=vae_weight_decay,
            eps=1e-6,
        )  # , betas=(beta1, beta2))
    elif vae_optimizer == "adam":
        optimizer = torch.optim.Adam(
            vae_model.parameters(),
            lr=vae_learning_rate,
            weight_decay=vae_weight_decay,
            eps=1e-6,
        )
    elif vae_optimizer == "radam":
        optimizer = torch.optim.RAdam(
            vae_model.parameters(),
            lr=vae_learning_rate,
            weight_decay=vae_weight_decay,
            eps=1e-6,
        )

    print("VAE Model setup done")

    best_val_loss = None
    moving_recon_loss = deque([], maxlen=10)
    best_recon_loss = None
    vae_best_weights = None
    for epoch in range(vae_num_epochs):
        vae_model.train()
        train_loss = 0.0
        train_loss_r = 0.0
        train_loss_d = 0.0
        track_means = 0.0
        for idx, sample in enumerate(train_loader):
            input_current, _, _, _, _ = sample
            optimizer.zero_grad()
            predictions, means, variances = vae_model(input_current)
            temp_r, temp_d = vae_criterion(input_current, predictions, means, variances)
            loss = (
                vae_wr * temp_r + temp_d * vae_wd
            )  # weight_KLD(e=epoch, a=alpha_KD, w=wd, me=num_epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=1.0)
            check_NaN_grad(vae_model)
            train_loss += loss.item()
            train_loss_r += temp_r.item()
            train_loss_d += temp_d.item()
            track_means += torch.mean(means)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        train_loss /= len(train_loader)
        train_loss_r /= len(train_loader)
        train_loss_d /= len(train_loader)
        track_means /= len(train_loader)

        if epoch % 250 == 0:
            print(
                {
                    "vae_epoch": epoch,
                    "vae_train_loss": train_loss,
                    "vae_train_recontruction": train_loss_r,
                    "vae_train_KLD": train_loss_d,
                    "vae_mean_latent_mean": track_means,
                }
            )

        vae_model.eval()

        val_loss = 0.0
        val_loss_r = 0.0
        val_loss_d = 0.0
        with torch.inference_mode():
            for val_idx, val_sample in enumerate(val_loader):
                val_input_current, _, _, _, _ = val_sample
                val_predictions, val_means, val_variances = vae_model(val_input_current)
                temp_r, temp_d = vae_criterion(
                    val_input_current, val_predictions, val_means, val_variances
                )
                val_batch_loss = vae_wr * temp_r + vae_wd * temp_d
                val_loss += val_batch_loss.item()
                val_loss_r += temp_r.item()
                val_loss_d += temp_d.item()

        val_loss /= len(val_loader)
        val_loss_r /= len(val_loader)
        val_loss_d /= len(val_loader)
        moving_recon_loss.append(val_loss_r)
        val_loss_r_smooth = np.mean(moving_recon_loss)
        # print(f'Training Loss after epoch {epoch}: {val_loss}')

        if not (best_val_loss and best_val_loss < val_loss):
            best_val_loss = val_loss
        if not (best_recon_loss and best_recon_loss < val_loss_r_smooth):
            best_recon_loss = val_loss_r_smooth
            vae_best_weights = vae_model.state_dict()

        if epoch % 250 == 0:
            print(
                {
                    "vae_epoch": epoch,
                    "vae_val_loss": val_loss,
                    "vae_val_recontruction_smooth": val_loss_r_smooth,
                    "vae_val_recontruction": val_loss_r,
                    "vae_val_KLD": val_loss_d,
                    "vae_best_val_loss": best_val_loss,
                    "vae_best_recon_loss": best_recon_loss,
                }
            )
    torch.save(vae_best_weights, "Data/pytorch_models/VAE_model.pth")

    print("VAE Model trained")


if __name__ == "__main__":
    main()
