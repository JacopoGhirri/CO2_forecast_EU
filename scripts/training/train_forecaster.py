import multiprocessing as mp
import os
from collections import deque

import numpy as np
import torch

# Set the start method at the very beginning of the script
mp.set_start_method("spawn", force=True)

from torch.utils.data import DataLoader

from config.data.output_configs import output_configs
from Scripts.Elements.Datasets import Dataset_forecasting_latent, Dataset_unified
from Scripts.Elements.Models import (
    Decoder,
    Encoder,
    ForecastModel_Latent,
    Full_Latent_Forecasting_Model,
    VAEModel,
    costum_uncertain_L1_loss_function,
)
from Scripts.utils import (
    check_NaN_grad,
    init_weights,
    load_config,
    load_datasets,
    save_datasets,
)


def main():
    config = load_config("Models/define_config_latenr_AR.yaml")
    vae_config = load_config("Models/define_config_VAE.yaml")

    batch_size = 128
    scaling_type = "normalization"  # 'maxmin'

    vae_activation = vae_config.vae_activation

    vae_latent_dim = vae_config.vae_latent_dim
    vae_num_blocks = vae_config.vae_num_blocks
    vae_dim_blocks = vae_config.vae_dim_blocks
    vae_normalization = vae_config.vae_normalization
    vae_dropouts = vae_config.vae_dropouts
    vae_input_dropout = vae_config.vae_input_dropouts

    dataset_filename = "Data/pytorch_datasets/unified_dataset.pkl"
    nested_variable_file = "config/data/variable_selection.pkl"
    with open(nested_variable_file) as file:
        nested_variables = [line.strip() for line in file]

    if os.path.exists(dataset_filename):
        uni_dataset = load_datasets(dataset_filename)
    else:
        uni_dataset = Dataset_unified(
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
        save_datasets(uni_dataset, dataset_filename)

    full_dataset = Dataset_forecasting_latent(uni_dataset)
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
    input_dim = len(full_dataset.full_dataset.input_variable_names)

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

    vae_model.load_state_dict(torch.load("Data/pytorch_models/VAE_model.pth"))

    forecast_num_epochs = 5001

    forecast_learning_rate = config.forecast_lr
    forecast_width_block = config.forecast_width_block
    forecast_dim_block = config.forecast_dim_block
    forecast_activation = config.forecast_activation
    forecast_normalization = config.forecast_normalization
    forecast_dropouts = config.forecast_dropouts
    forecast_num_blocks = config.forecast_num_blocks
    forecast_weight_decay = config.forecast_wd
    forecast_optimizer = config.forecast_optimizer

    forecast_model = ForecastModel_Latent(
        input_dim=2
        * (
            vae_model.Encoder.latent_dim
            + len(full_dataset.full_dataset.context_variable_names)
        ),
        latent_dim=vae_model.Encoder.latent_dim,
        width_block=forecast_width_block,
        dim_block=forecast_dim_block,
        activation=forecast_activation,
        normalization=forecast_normalization,
        dropouts=forecast_dropouts,
        num_blocks=forecast_num_blocks,
    ).cuda()

    forecast_model.apply(init_weights)

    lat_model = Full_Latent_Forecasting_Model(VAE=vae_model, Forecaster=forecast_model)

    if forecast_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": lat_model.Forecaster.parameters(),
                    "lr": forecast_learning_rate,
                }  # Higher LR for new layers
            ],
            weight_decay=forecast_weight_decay,
            eps=1e-6,
        )
    elif forecast_optimizer == "adam":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": lat_model.Forecaster.parameters(),
                    "lr": forecast_learning_rate,
                }  # Higher LR for new layers
            ],
            weight_decay=forecast_weight_decay,
            eps=1e-6,
        )
    elif forecast_optimizer == "radam":
        optimizer = torch.optim.RAdam(
            [
                {
                    "params": lat_model.Forecaster.parameters(),
                    "lr": forecast_learning_rate,
                }  # Higher LR for new layers
            ],
            weight_decay=forecast_weight_decay,
            eps=1e-6,
        )

    for p in lat_model.Encoder.parameters():
        p.requires_grad = False
    for p in lat_model.VAE.parameters():
        p.requires_grad = False

    best_val_loss = None
    best_val_loss_smooth = None
    best_val_acc = None
    best_val_acc_smooth = None
    val_loss_history = deque(maxlen=50)
    val_acc_history = deque(maxlen=50)

    for epoch in range(forecast_num_epochs):
        lat_model.train()
        train_loss = 0
        for _idx, sample in enumerate(train_loader):
            (
                input_current,
                context_current,
                input_prev,
                context_prev,
                input_past,
                context_past,
            ) = sample
            optimizer.zero_grad()
            forecasted_latent = lat_model(
                input_prev, input_past, context_current, context_prev
            )
            mean_current, log_var_current = lat_model.Encoder(input_current)
            loss = costum_uncertain_L1_loss_function(
                mean_current, forecasted_latent, log_var_current, "regular"
            )

            loss.backward()
            check_NaN_grad(lat_model)
            torch.nn.utils.clip_grad_norm_(lat_model.parameters(), 1)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        lat_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.inference_mode():
            for _val_idx, val_sample in enumerate(val_loader):
                (
                    input_current,
                    context_current,
                    input_prev,
                    context_prev,
                    input_past,
                    context_past,
                ) = val_sample
                forecasted_latent = lat_model(
                    input_prev, input_past, context_current, context_prev
                )
                mean_current, log_var_current = lat_model.Encoder(input_current)
                val_loss_value = costum_uncertain_L1_loss_function(
                    mean_current, forecasted_latent, log_var_current, "regular"
                )
                val_loss += val_loss_value.item()
                val_accuracy += (mean_current - forecasted_latent).pow(2).mean()

        val_loss /= len(val_loader)
        val_accuracy = val_accuracy / len(val_loader)
        val_loss_history.append(val_loss)
        val_loss_smooth = sum(val_loss_history) / len(val_loss_history)
        val_acc_history.append(val_accuracy)
        val_acc_smooth = sum(val_acc_history) / len(val_acc_history)

        if not best_val_loss or best_val_loss > val_loss:
            best_val_loss = val_loss
        if not best_val_loss_smooth or best_val_loss_smooth > val_loss_smooth:
            best_val_loss_smooth = val_loss_smooth
        if not best_val_acc or best_val_acc > val_accuracy:
            best_val_acc = val_accuracy
        if not best_val_acc_smooth or best_val_acc_smooth > val_acc_smooth:
            best_val_acc_smooth = val_acc_smooth
            torch.save(
                lat_model.state_dict(), "Data/pytorch_models/forecast_model_latent.pth"
            )

        if epoch % 250 == 0:
            print(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "pred_val_loss": val_loss,
                    "pred_val_accuracy": val_accuracy,
                    "pred_val_loss_smooth": val_loss_smooth,
                    "pred_val_acc_smooth": val_acc_smooth,
                    "pred_best_val_loss": best_val_loss,
                    "pred_best_val_loss_smooth": best_val_loss_smooth,
                    "pred_best_val_acc": best_val_acc,
                    "pred_best_val_acc_smooth": best_val_acc_smooth,
                }
            )


if __name__ == "__main__":
    main()
