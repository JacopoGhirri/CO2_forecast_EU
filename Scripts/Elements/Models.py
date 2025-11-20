import torch
import torch.nn as nn
import numpy as np


class residual_block(nn.Module):
    """
    Main building block for models.
    """

    def __init__(self, width, dim_block, activation, normalization, dropout=0):
        super(residual_block, self).__init__()
        layers = []
        for i in range(dim_block - 1):
            layers.append(nn.Linear(width, width))
            # place normalization, if any, at the beginning of the block
            if i == 0:
                if normalization == "batch":
                    layers.append(nn.BatchNorm1d(width))
                elif normalization == "layer":
                    layers.append(nn.LayerNorm(width))
                elif normalization == "none":
                    pass
                else:
                    raise ValueError(f"Unsupported normalization type: {normalization}")
            layers.append(activation)
            # dropout after every layer
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # learn a residual connection
        return x + self.layers(x)


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "None" or activation is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


def reparameterization(mean, var):
    epsilon = torch.randn_like(var)
    z = mean + var * epsilon
    return z


class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_blocks,
        dim_blocks,
        activation,
        normalization="batch",
        dropouts=0,
        input_dropout=0,
    ):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.dim_blocks = dim_blocks
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropouts = dropouts
        self.input_dropout = input_dropout

        pass_sizes = list(
            self.input_dim
            - np.arange(0, self.num_blocks)
            * int((self.input_dim - self.latent_dim) / self.num_blocks)
        )
        pass_sizes.append(self.latent_dim)

        blocks = []
        blocks.append(nn.Dropout(self.input_dropout))
        for i in range(0, len(pass_sizes) - 1):
            s_in = pass_sizes[i]
            s_out = pass_sizes[i + 1]
            blocks.append(nn.Linear(s_in, s_out))
            blocks.append(
                residual_block(
                    s_out,
                    self.dim_blocks,
                    self.activation,
                    self.normalization,
                    self.dropouts,
                )
            )

        self.net = nn.Sequential(*blocks)
        self.mean = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, input):
        net = self.net(input)
        mean = self.mean(net)
        log_var = self.log_var(net)

        return mean, log_var


class Decoder(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_blocks,
        dim_blocks,
        activation,
        normalization="batch",
        dropouts=0,
    ):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.dim_blocks = dim_blocks
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropouts = dropouts

        pass_sizes = list(
            self.input_dim
            - np.arange(0, self.num_blocks)
            * int((self.input_dim - self.latent_dim) / self.num_blocks)
        )
        pass_sizes.append(self.latent_dim)

        pass_sizes = pass_sizes[::-1]

        blocks = []
        for i in range(0, len(pass_sizes) - 1):
            s_in = pass_sizes[i]
            s_out = pass_sizes[i + 1]
            blocks.append(nn.Linear(s_in, s_out))
            blocks.append(
                residual_block(
                    s_out,
                    self.dim_blocks,
                    self.activation,
                    self.normalization,
                    self.dropouts,
                )
            )

        self.net = nn.Sequential(*blocks)
        self.out = nn.Linear(input_dim, input_dim)

    def forward(self, latent):
        # Pass through MLP
        net = self.net(latent)
        out = self.out(net)
        return out


class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        mean, log_var = self.Encoder(x)

        # Reparameterization
        z = reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # Variance from log variance

        # Decode the latent variable
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


class ForecastModel_multiyears(nn.Module):
    def __init__(
        self,
        input_dim,
        output_configs,
        num_blocks,
        dim_block,
        width_block,
        activation,
        normalization="batch",
        dropouts=0,
        uncertainty=True,
    ):
        super(ForecastModel_multiyears, self).__init__()

        self.input_dim = input_dim
        self.dim_block = dim_block
        self.width_block = width_block
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropouts = dropouts
        self.output_configs = output_configs
        self.num_blocks = num_blocks
        self.uncertainty = uncertainty

        if output_configs["output"] == "Sectors":
            self.output_size = 6
        elif output_configs["output"] in ["Total", "TotalECON", "TotalHOUSE"]:
            self.output_size = 1

        blocks = []

        blocks.append(nn.Linear(self.input_dim, self.width_block))
        for _ in range(self.num_blocks):
            blocks.append(
                residual_block(
                    self.width_block,
                    self.dim_block,
                    self.activation,
                    self.normalization,
                    self.dropouts,
                )
            )

        self.net = nn.Sequential(*blocks)

        if self.output_configs["measure"] == "both":
            self.output_layer = nn.Linear(self.width_block, self.output_size * 2)
        else:
            self.output_layer = nn.Linear(self.width_block, self.output_size)

        if self.uncertainty:
            self.forward_selected = self.forward_uncertain
            self.log_var = nn.Linear(self.width_block, self.output_size)
        else:
            self.forward_selected = self.forward_deterministic

    def forward_uncertain(self, inputs):
        mlp_out = self.net(inputs)

        # Final output layer
        output = self.output_layer(mlp_out)
        log_var = self.log_var(mlp_out)

        return output, log_var

    def forward_deterministic(self, inputs):
        mlp_out = self.net(inputs)

        # Final output layer
        output = self.output_layer(mlp_out)

        return output.squeeze(1)

    def forward(self, inputs):
        return self.forward_selected(inputs)


class Full_Model(nn.Module):
    def __init__(self, VAE, Predictor):
        super(Full_Model, self).__init__()
        self.VAE = VAE
        self.Predictor = Predictor
        self.Encoder = self.VAE.Encoder
        self.Decoder = self.VAE.Decoder

    def forward(
        self, input_current, input_prev, context_current, context_prev, emissions_prev
    ):
        mean_current, log_var_current = self.Encoder(input_current)
        mean_prev, log_var_prev = self.Encoder(input_prev)

        # Reparameterization
        z_current = reparameterization(
            mean_current, torch.exp(0.5 * log_var_current)
        )  # Variance from log variance
        z_prev = reparameterization(mean_prev, torch.exp(0.5 * log_var_prev))

        stacker = torch.cat((z_current, context_current, z_prev, context_prev), dim=1)

        # Decode the latent variable
        emissions_delta, emission_uncertainty = self.Predictor(stacker)
        emissions_predictions = emissions_prev + emissions_delta

        # sanity checks
        recon_current = self.Decoder(z_current)
        recon_prev = self.Decoder(z_prev)

        return (
            emissions_delta,
            emission_uncertainty,
            recon_current,
            recon_prev,
            mean_current,
            mean_prev,
            log_var_current,
            log_var_prev,
        )


def VAE_loss_function(x, x_hat, mean, log_var):
    total_reconstruction_loss = nn.functional.l1_loss(x_hat, x, reduce=False)
    total_reconstruction_loss = torch.clamp(total_reconstruction_loss, 0, 5.0)
    total_reconstruction_loss = torch.nanmean(total_reconstruction_loss)

    # KL Divergence term
    KLD = -0.5 * torch.nanmean(1 + log_var - mean.pow(2) - log_var.exp())

    # Combine reconstruction loss and KL divergence
    return total_reconstruction_loss, KLD


def costum_uncertain_forecast_loss_function(
    x, x_hat, log_var_uncertainty, mode="regular"
):
    """
    Computes the uncertainty-aware loss, adapting automatically whether x contains
    one or two representations (e.g., total and per capita values).

    Args:
        x (torch.Tensor): Ground truth, shape (batch_size, num_vars) or (batch_size, 2 * num_vars)
        x_hat (torch.Tensor): Predictions, same shape as x
        log_var_uncertainty (torch.Tensor): Uncertainty estimates, shape (batch_size, num_vars)

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Ensure uncertainty can track gradients
    if log_var_uncertainty.requires_grad and not log_var_uncertainty.is_leaf:
        log_var_uncertainty.retain_grad()

    error = (x - x_hat).abs()
    error = torch.clamp(error, min=-0.0, max=2.0)
    loss = 0.5 * torch.exp(-log_var_uncertainty) * error
    if mode == "regular":
        loss += 0.5 * log_var_uncertainty
    elif mode == "exponential":
        loss += torch.exp(log_var_uncertainty)
    elif mode == "factor":
        loss += 10 * log_var_uncertainty
    elif mode == "quadrexp":
        loss += torch.exp(log_var_uncertainty) ** 2
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    loss = loss.nanmean()

    return loss


def costum_uncertain_predict_loss_function(
    x, x_hat, log_var_uncertainty, mode="regular"
):
    """
    Computes the uncertainty-aware loss, adapting automatically whether x contains
    one or two representations (e.g., total and per capita values).

    Args:
        x (torch.Tensor): Ground truth, shape (batch_size, num_vars) or (batch_size, 2 * num_vars)
        x_hat (torch.Tensor): Predictions, same shape as x
        log_var_uncertainty (torch.Tensor): Uncertainty estimates, shape (batch_size, num_vars)

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Ensure uncertainty can track gradients
    if log_var_uncertainty.requires_grad and not log_var_uncertainty.is_leaf:
        log_var_uncertainty.retain_grad()

    error = torch.square(x - x_hat)
    error = torch.clamp(error, min=-0.0, max=2.0)
    loss = 0.5 * torch.exp(-log_var_uncertainty) * error
    if mode == "regular":
        loss += 0.5 * log_var_uncertainty
    elif mode == "exponential":
        loss += torch.tanh_(log_var_uncertainty)
    elif mode == "factor":
        loss += 0.005 * log_var_uncertainty
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    loss = loss.nanmean()

    return loss


class ForecastModel_Latent(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_blocks,
        dim_block,
        width_block,
        activation,
        normalization="batch",
        dropouts=0,
    ):
        super(ForecastModel_Latent, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dim_block = dim_block
        self.width_block = width_block
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropouts = dropouts
        self.num_blocks = num_blocks

        blocks = []

        blocks.append(nn.Linear(self.input_dim, self.width_block))
        for _ in range(self.num_blocks):
            blocks.append(
                residual_block(
                    self.width_block,
                    self.dim_block,
                    self.activation,
                    self.normalization,
                    self.dropouts,
                )
            )

        self.net = nn.Sequential(*blocks)

        self.output_layer = nn.Linear(self.width_block, self.latent_dim)

    def forward(self, latent_current, latent_prev, context_next, context_current):
        stack = torch.cat(
            (latent_current, latent_prev, context_next, context_current), dim=1
        )
        mlp_out = self.net(stack)

        # Final output layer
        output = latent_current + self.output_layer(mlp_out)

        return output


class Latent_Model(nn.Module):
    def __init__(self, VAE, Forecaster):
        super(Latent_Model, self).__init__()
        self.VAE = VAE
        self.Forecaster = Forecaster
        self.Encoder = self.VAE.Encoder

    def forward(self, input_current, input_prev, context_next, context_current):
        mean_current, log_var_current = self.Encoder(input_current)
        mean_prev, log_var_prev = self.Encoder(input_prev)

        # Reparameterization
        z_current = reparameterization(
            mean_current, torch.exp(0.5 * log_var_current)
        )  # Variance from log variance
        z_prev = reparameterization(mean_prev, torch.exp(0.5 * log_var_prev))

        forecast = self.Forecaster(z_current, z_prev, context_next, context_current)
        return forecast
