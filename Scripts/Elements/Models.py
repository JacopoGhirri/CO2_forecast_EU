import torch
import torch.nn as nn
import numpy as np


class residual_block(nn.Module):
    """
    Main building block for models.
    """

    def __init__(self, width, dim_block, activation, normalization, dropout=0.0):
        """
        Args:
            width: number of neurons in each layer
            dim_block: number of hidden layers in the block
            activation: activation function
            normalization: type of normalization layer, placed only once in the block after the first linear layer
            dropout: numeric value between 0 and 1
        """
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
        # learns a residual connection
        return x + self.layers(x)


def get_activation(activation):
    """
    Utility function to get activation torch module from string.
    """
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
    """
    Utlity function for implementing reparameterization trick.
    """
    epsilon = torch.randn_like(var)
    z = mean + var * epsilon
    return z


class Encoder(nn.Module):
    """
    Encoder model, to obtain latent representation from input variables.
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_blocks,
        dim_blocks,
        activation,
        normalization="batch",
        dropouts=0.0,
        input_dropout=0.0,
    ):
        """
        Args:
            input_dim: number of input variables
            latent_dim: dimensionality of latent representation
            num_blocks: number of residual blocks
            dim_blocks: number of hidden layers in each residual block
            activation: string name of activation function
            normalization: string name of normalization layer
            dropouts: dropout value inside the network, numeric value between 0 and 1
            input_dropout: dropout value applied directly on inputs, numeric value between 0 and 1
        """
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.dim_blocks = dim_blocks
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropouts = dropouts
        self.input_dropout = input_dropout

        #width of residual blocks linearly decreasing from input size to latent representation
        pass_sizes = list(
            self.input_dim
            - np.arange(0, self.num_blocks)
            * int((self.input_dim - self.latent_dim) / self.num_blocks)
        )
        pass_sizes.append(self.latent_dim)

        blocks = []
        #apply input dropout first
        blocks.append(nn.Dropout(self.input_dropout))
        #append individual residual blocks
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
        #separate output layers for mean and log variance
        self.mean = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, input):
        net = self.net(input)
        mean = self.mean(net)
        log_var = self.log_var(net)

        return mean, log_var


class Decoder(nn.Module):
    """
    Decoder model, to obtain reconstructed input variables from latent representation.
    """
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
        """
        Args:
            input_dim: number of input variables
            latent_dim: dimensionality of latent representation
            num_blocks: number of residual blocks
            dim_blocks: number of hidden layers in each residual block
            activation: string name of activation function
            normalization: string name of normalization layer
            dropouts: dropout value inside the network, numeric value between 0 and 1
        """
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.dim_blocks = dim_blocks
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropouts = dropouts

        #residual block structure replicates the behaviour of encoder network, but reverses the order.
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
        net = self.net(latent)
        out = self.out(net)
        return out


class VAEModel(nn.Module):
    """
    Variational Autoencoder model, to obtain latent representation from input variables, and the associated reconstruction..
    """
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

        # Decode the latent variable from a random sample
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


class Emission_Predictor(nn.Module):
    """
    Emission Predictor model, to obtain emission variations from input variables and contexts.
    """
    def __init__(
        self,
        input_dim,
        output_configs,
        num_blocks,
        dim_block,
        width_block,
        activation,
        normalization="batch",
        dropouts=0.0,
        uncertainty=True,
    ):
        """
        Args:
            input_dim: size of input variables
            output_configs: configuration of output emissions
            num_blocks: number of residual blocks
            dim_block: number of hidden layers in each residual block
            width_block: number of neurons in hidden layers in each residual block
            activation: string name of activation function
            normalization: string name of normalization layer
            dropouts: numeric value between 0 and 1
            uncertainty: logical, for training with or without uncertainty
        """
        super(Emission_Predictor, self).__init__()

        self.input_dim = input_dim
        self.dim_block = dim_block
        self.width_block = width_block
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropouts = dropouts
        self.output_configs = output_configs
        self.num_blocks = num_blocks
        self.uncertainty = uncertainty

        #read how many outputs are expected
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

        #check if the output configuration requires both total and per capita emissions, or only one of the two
        if self.output_configs["measure"] == "both":
            self.output_layer = nn.Linear(self.width_block, self.output_size * 2)
        else:
            self.output_layer = nn.Linear(self.width_block, self.output_size)

        #different forward methods depending on self.uncertainty
        #if trained with uncertainty, learn and output also model uncertainty
        if self.uncertainty:
            self.forward_selected = self.forward_uncertain
            self.learned_uncertainty = nn.Linear(self.width_block, self.output_size)
        else:
            self.forward_selected = self.forward_deterministic

    def forward_uncertain(self, inputs):
        mlp_out = self.net(inputs)

        # Final output layer
        output = self.output_layer(mlp_out)
        learned_uncertainty = self.learned_uncertainty(mlp_out)

        return output, learned_uncertainty

    def forward_deterministic(self, inputs):
        mlp_out = self.net(inputs)

        # Final output layer
        output = self.output_layer(mlp_out)

        return output.squeeze(1)

    def forward(self, inputs):
        return self.forward_selected(inputs)


class Full_Prediction_Model(nn.Module):
    """
    Wrapper class for direct emission prediction, combining VAE and Emission Predictors for simplified pipeline.
    Assumes Emission Predictor is configured with uncertainty=True.
    """
    def __init__(self, VAE, Predictor):
        super(Full_Prediction_Model, self).__init__()
        self.VAE = VAE
        self.Predictor = Predictor
        self.Encoder = self.VAE.Encoder
        self.Decoder = self.VAE.Decoder

        assert self.Predictor.uncertainty == True

    def forward(
        self, input_current, input_prev, context_current, context_prev
    ):
        """
        Given inputs and contexts, outputs emission variations, confidence, and VAE outputs for diagnostics.

        Args:
            input_current: input variables at time t
            input_prev: input variables at time t-1
            context_current: context variables at time t
            context_prev: context variables at time t-1

        Returns:
            emission_delta: change in emissions at time t wrt t-1
            emission_uncertainty: learned uncertainty in emissions_delta
            recon_current: reconstructed input variables at time t
            recon_prev: reconstructed input variables at time t-1
            mean_current: mean latent variables at time t
            mean_prev: mean latent variables at time t-1
            log_var_current: log variance of latent variables at time t
            log_var_prev: log variance of latent variables at time t-1

        """
        mean_current, log_var_current = self.Encoder(input_current)
        mean_prev, log_var_prev = self.Encoder(input_prev)

        # Reparameterization
        z_current = reparameterization(
            mean_current, torch.exp(0.5 * log_var_current)
        )  # Variance from log variance
        z_prev = reparameterization(mean_prev, torch.exp(0.5 * log_var_prev))

        stacker = torch.cat((z_current, context_current, z_prev, context_prev), dim=1)

        # Decode the latent variable
        emission_delta, emission_uncertainty = self.Predictor(stacker)

        # sanity checks
        recon_current = self.Decoder(z_current)
        recon_prev = self.Decoder(z_prev)

        return (
            emission_delta,
            emission_uncertainty,
            recon_current,
            recon_prev,
            mean_current,
            mean_prev,
            log_var_current,
            log_var_prev,
        )

class ForecastModel_Latent(nn.Module):
    """
    Model for forecasting the next latent sample from previous latent means, and contexts.
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_blocks,
        dim_block,
        width_block,
        activation,
        normalization="batch",
        dropouts=0.0,
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

        # learn a residual connection
        output = latent_current + self.output_layer(mlp_out)

        return output


class Full_Latent_Forecasting_Model(nn.Module):
    """
    Wrapper class for forecasting latent space directly from input variables and contexts.
    """
    def __init__(self, VAE, Forecaster):
        super(Full_Latent_Forecasting_Model, self).__init__()
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


def VAE_loss_function(x, x_hat, mean, log_var):
    """
    Variational autoencoder loss function.

    Args:
        x: true input variables
        x_hat: reconstructed input variables
        mean: mean of latent variables
        log_var: log variance of latent variables

    Returns:
        total_recon_loss: Mean Absolute Reconstruction Error
        KLD: KL Divergence loss

    """
    total_reconstruction_loss = nn.functional.l1_loss(x_hat, x, reduce=False)
    #avoid unstable learning at first iterations
    total_reconstruction_loss = torch.clamp(total_reconstruction_loss, 0, 5.0)
    total_reconstruction_loss = torch.nanmean(total_reconstruction_loss)

    # KL Divergence term
    KLD = -0.5 * torch.nanmean(1 + log_var - mean.pow(2) - log_var.exp())

    # Combine reconstruction loss and KL divergence
    return total_reconstruction_loss, KLD


def costum_uncertain_L1_loss_function(
    x, x_hat, log_var, mode="regular"
):
    """
    Computes the uncertainty-aware MAE loss, used when forecasting latent spaces

    Args:
        x: target latent sample
        x_hat: forecast latent sample
        log_var: target latent variance

    Returns:
        Scalar loss value
    """
    # Ensure uncertainty can track gradients
    if log_var.requires_grad and not log_var.is_leaf:
        log_var.retain_grad()

    #MAE
    error = (x - x_hat).abs()
    error = torch.clamp(error, min=-0.0, max=2.0)

    #discount errors by latent variance
    loss = 0.5 * torch.exp(-log_var) * error

    #apply loss penalty for latent variance, eventually rescaled
    if mode == "regular":
        loss += 0.5 * log_var
    elif mode == "exponential":
        loss += torch.exp(log_var)
    elif mode == "factor":
        loss += 10 * log_var
    elif mode == "quadrexp":
        loss += torch.exp(log_var) ** 2
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    loss = loss.nanmean()

    return loss


def costum_uncertain_L2_loss_function(
    x, x_hat, learned_uncertainty, mode="regular"
):
    """
    Computes the uncertainty-aware MSE loss, used when predicting emissions

    Args:
        x: ground truth emissions
        x_hat: predicted emissions
        learned_uncertainty: Uncertainty estimates, shape (batch_size, num_vars)

    Returns:
        Scalar loss value
    """
    # Ensure uncertainty can track gradients
    if learned_uncertainty.requires_grad and not learned_uncertainty.is_leaf:
        learned_uncertainty.retain_grad()

    #Squared error
    error = torch.square(x - x_hat)
    error = torch.clamp(error, min=-0.0, max=2.0)

    # discount errors by model confidence
    loss = 0.5 * torch.exp(-learned_uncertainty) * error

    # apply penalty for model uncertainty, eventually rescaled
    if mode == "regular":
        loss += 0.5 * learned_uncertainty
    elif mode == "exponential":
        loss += torch.tanh_(learned_uncertainty)
    elif mode == "factor":
        loss += 0.005 * learned_uncertainty
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    loss = loss.nanmean()

    return loss
