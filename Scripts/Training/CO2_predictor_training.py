import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import os
import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.data.output_configs import output_configs
from Scripts.Elements.Datasets import Dataset_predict_AR
from Scripts.Elements.Models import (
    Decoder,
    Emission_Predictor,
    Encoder,
    Full_Prediction_Model,
    VAE_loss_function,
    VAEModel,
    costum_uncertain_L2_loss_function,
)
from Scripts.utils import (
    check_NaN_grad,
    init_weights,
    load_config,
    load_datasets,
    save_datasets,
)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def main():
    # Load VAE and Predictor configuration from YAML
    vae_config = load_config("config/models/vae_config.yaml")
    pred_config = load_config("config/models/co2_predictor_config.yaml")

    batch_size = 128
    vae_latent_dim = vae_config.vae_latent_dim
    scaling_type = "normalization"

    dataset_filename = "Data/pytorch_datasets/predictor_dataset.pkl"
    nested_variable_file = "nested_variables_reduced_v0.txt"
    with open(nested_variable_file, "r") as file:
        nested_variables = [line.strip() for line in file]

    if os.path.exists(dataset_filename):
        full_dataset = load_datasets(dataset_filename)
    else:
        full_dataset = Dataset_predict_AR(
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
        save_datasets(full_dataset, dataset_filename)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [0.85, 0.15]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = len(full_dataset.input_variable_names)
    enc = Encoder(
        input_dim=input_dim,
        latent_dim=vae_latent_dim,
        num_blocks=vae_config.vae_num_blocks,
        dim_blocks=vae_config.vae_dim_blocks,
        activation=vae_config.vae_activation,
        normalization=vae_config.vae_normalization,
        dropouts=vae_config.vae_dropouts,
        input_dropout=vae_config.vae_input_dropouts,
    ).cuda()
    dec = Decoder(
        input_dim=input_dim,
        latent_dim=vae_latent_dim,
        num_blocks=vae_config.vae_num_blocks,
        dim_blocks=vae_config.vae_dim_blocks,
        activation=vae_config.vae_activation,
        normalization=vae_config.vae_normalization,
        dropouts=vae_config.vae_dropouts,
    ).cuda()
    vae_model = VAEModel(enc, dec).cuda()
    vae_model.load_state_dict(torch.load("Data/pytorch_models/VAE_model.pth"))
    vae_criterion = VAE_loss_function

    pred_model = Emission_Predictor(
        input_dim=2 * (vae_latent_dim + len(full_dataset.context_variable_names)),
        output_configs=output_configs,
        width_block=pred_config.pred_width_block,
        dim_block=pred_config.pred_dim_block,
        activation=pred_config.pred_activation,
        normalization=pred_config.pred_normalization,
        dropouts=pred_config.pred_dropouts,
        num_blocks=pred_config.pred_num_blocks,
        uncertainty=True,
    ).cuda()
    pred_model.apply(init_weights)

    full_model = Full_Prediction_Model(VAE=vae_model, Predictor=pred_model)

    # Optimizer setup
    lr = pred_config.pred_lr
    wd = pred_config.pred_wd
    if pred_config.pred_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [
                {"params": full_model.Encoder.parameters(), "lr": lr * 1e-3},
                {"params": full_model.Decoder.parameters(), "lr": lr * 1e-3},
                {"params": full_model.Predictor.parameters(), "lr": lr},
            ],
            weight_decay=wd,
            eps=1e-6,
        )
    elif pred_config.pred_optimizer == "adam":
        optimizer = torch.optim.Adam(
            [
                {"params": full_model.Encoder.parameters(), "lr": lr * 1e-3},
                {"params": full_model.Decoder.parameters(), "lr": lr * 1e-3},
                {"params": full_model.Predictor.parameters(), "lr": lr},
            ],
            weight_decay=wd,
            eps=1e-6,
        )
    elif pred_config.pred_optimizer == "radam":
        optimizer = torch.optim.RAdam(
            [
                {"params": full_model.Encoder.parameters(), "lr": lr * 1e-3},
                {"params": full_model.Decoder.parameters(), "lr": lr * 1e-3},
                {"params": full_model.Predictor.parameters(), "lr": lr},
            ],
            weight_decay=wd,
            eps=1e-6,
        )

    # Freeze VAE weights
    for p in full_model.Encoder.parameters():
        p.requires_grad = False
    for p in full_model.Decoder.parameters():
        p.requires_grad = False

    pred_num_epochs = 5000
    mode_loss = pred_config.mode_loss

    best_val_loss = None
    best_val_loss_smooth = None
    best_val_acc = None
    best_val_acc_smooth = None
    val_loss_history = deque(maxlen=50)
    val_acc_history = deque(maxlen=50)

    for epoch in range(pred_num_epochs):
        full_model.train()
        train_loss = 0
        for idx, sample in enumerate(train_loader):
            (
                input_current,
                context_current,
                emissions,
                input_prev,
                context_prev,
                emissions_prev,
            ) = sample
            optimizer.zero_grad()
            (
                emissions_predictions_DELTA,
                emission_uncertainty,
                recon_current,
                recon_prev,
                mean_current,
                mean_prev,
                log_var_current,
                log_var_prev,
            ) = full_model(
                input_current, input_prev, context_current, context_prev, emissions_prev
            )
            emission_DELTA = emissions - emissions_prev
            pred_loss_value = costum_uncertain_L2_loss_function(
                emission_DELTA,
                emissions_predictions_DELTA,
                emission_uncertainty,
                mode_loss,
            )
            loss = pred_loss_value
            loss.backward()
            check_NaN_grad(full_model)
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        full_model.eval()
        val_loss = 0.0
        val_uncertainty = 0.0
        val_accuracy = 0.0
        val_reconstruction = 0.0
        with torch.inference_mode():
            for val_sample in val_loader:
                (
                    val_input_current,
                    val_context_current,
                    val_emissions,
                    val_input_prev,
                    val_context_prev,
                    val_emissions_prev,
                ) = val_sample
                (
                    val_emissions_predictions_DELTA,
                    val_emission_uncertainty,
                    val_recon_current,
                    val_recon_prev,
                    val_mean_current,
                    val_mean_prev,
                    val_log_var_current,
                    val_log_var_prev,
                ) = full_model(
                    val_input_current,
                    val_input_prev,
                    val_context_current,
                    val_context_prev,
                    val_emissions_prev,
                )
                val_emission_DELTA = val_emissions - val_emissions_prev
                val_pred_loss_value = costum_uncertain_L2_loss_function(
                    val_emission_DELTA,
                    val_emissions_predictions_DELTA,
                    val_emission_uncertainty,
                    mode_loss,
                )
                val_pred_vae_curr_recon, val_pred_vae_curr_KLD = vae_criterion(
                    val_input_current,
                    val_recon_current,
                    val_mean_current,
                    val_log_var_current,
                )
                val_pred_vae_prev_recon, val_pred_vae_prev_KLD = vae_criterion(
                    val_input_prev, val_recon_prev, val_mean_prev, val_log_var_prev
                )

                val_loss_value = val_pred_loss_value
                val_loss += val_loss_value.item()
                val_uncertainty += val_emission_uncertainty.mean().item()
                val_accuracy += (
                    (
                        val_emissions_predictions_DELTA
                        + val_emissions_prev
                        - val_emissions
                    )
                    .pow(2)
                    .mean()
                    .item()
                )
                val_reconstruction += val_pred_vae_curr_recon.mean().item()

        val_loss /= len(val_loader)
        val_uncertainty /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_reconstruction /= len(val_loader)

        val_loss_history.append(val_loss)
        val_loss_smooth = sum(val_loss_history) / len(val_loss_history)
        val_acc_history.append(val_accuracy)
        val_acc_smooth = sum(val_acc_history) / len(val_acc_history)

        # Print full metrics every 250 epochs
        if epoch % 250 == 0 or epoch == pred_num_epochs - 1:
            print(f"--- Epoch {epoch} ---")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Loss (Smoothed): {val_loss_smooth:.4f}")
            print(f"Validation Uncertainty: {val_uncertainty:.4f}")
            print(f"Validation Accuracy (MSE): {val_accuracy:.4f}")
            print(f"Validation VAE Recon Loss: {val_reconstruction:.4f}")
            print(
                f"Best Smoothed Val Loss so far: {best_val_loss_smooth:.4f}"
                if best_val_loss_smooth
                else "N/A"
            )
            print(
                f"Best Smoothed Val Accuracy so far: {best_val_acc_smooth:.4f}"
                if best_val_acc_smooth
                else "N/A"
            )

        # Save best model
        if not best_val_loss or best_val_loss > val_loss:
            best_val_loss = val_loss
        if not best_val_loss_smooth or best_val_loss_smooth > val_loss_smooth:
            best_val_loss_smooth = val_loss_smooth
            torch.save(
                full_model.state_dict(), "Data/pytorch_models/predictor_model_co2.pth"
            )

        if not best_val_acc or best_val_acc > val_accuracy:
            best_val_acc = val_accuracy
        if not best_val_acc_smooth or best_val_acc_smooth > val_acc_smooth:
            best_val_acc_smooth = val_acc_smooth


if __name__ == "__main__":
    main()
