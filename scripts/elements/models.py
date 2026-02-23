"""
Neural network architectures for EU emission forecasting.

This module implements a hierarchical deep learning framework for projecting
sectoral CO2 emissions across EU member states. The architecture consists of:

1. **Variational Autoencoder (VAE)**: Learns compressed latent representations
   of high-dimensional socioeconomic indicators while capturing uncertainty.

2. **Emission Predictor**: Maps VAE latents and context variables to sectoral
   emission predictions with learned uncertainty estimates.

3. **Latent Forecaster**: Projects future latent states for autoregressive
   emission projections.

Reference:
    This implementation supports the paper "Deep Learning Emission Projections
    Challenge European Climate Ambitions" analyzing EU27 decarbonization trajectories.

Example:
    >>> # Build and train the full pipeline
    >>> vae = VAEModel(encoder, decoder)
    >>> predictor = EmissionPredictor(input_dim, output_configs, ...)
    >>> full_model = FullPredictionModel(vae, predictor)
    >>> delta, uncertainty, *diagnostics = full_model(x_t, x_t1, c_t, c_t1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# =============================================================================
# Utility Functions
# =============================================================================


def get_activation(activation: str) -> nn.Module:
    """
    Returns a PyTorch activation module from a string identifier.

    Args:
        activation: Name of the activation function. Supported values:
            'relu', 'gelu', 'selu', 'tanh', 'sigmoid', 'leaky_relu',
            'silu', 'None', or None.

    Returns:
        The corresponding PyTorch activation module.

    Raises:
        ValueError: If the activation name is not recognized.

    Example:
        >>> act = get_activation('gelu')
        >>> act(torch.tensor([-1.0, 0.0, 1.0]))
    """
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "None": nn.Identity,
        None: nn.Identity,
    }

    if activation not in activations:
        raise ValueError(
            f"Unsupported activation function: {activation}. "
            f"Supported: {list(activations.keys())}"
        )

    return activations[activation]()


def reparameterize(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Implements the reparameterization trick for VAE training.

    Samples from N(mean, std^2) in a way that allows backpropagation
    through the sampling operation.

    Args:
        mean: Mean of the latent distribution, shape (batch_size, latent_dim).
        std: Standard deviation (NOT variance), shape (batch_size, latent_dim).

    Returns:
        Sampled latent vectors, shape (batch_size, latent_dim).

    Note:
        The caller should pass std = exp(0.5 * log_var) if working with
        log variance outputs from the encoder.
    """
    epsilon = torch.randn_like(std)
    return mean + std * epsilon


# =============================================================================
# Building Blocks
# =============================================================================


class ResidualBlock(nn.Module):
    """
    Residual block with configurable depth, normalization, and dropout.

    This is the fundamental building block for all networks in this framework.
    It implements a skip connection that learns a residual mapping, which
    facilitates gradient flow and enables training of deeper networks.

    Architecture:
        input -> [Linear -> Norm -> Activation -> Dropout] x (dim_block-1) -> + input

    Attributes:
        layers: Sequential container of linear, normalization, activation,
            and dropout layers.

    Example:
        >>> block = ResidualBlock(width=128, dim_block=3, activation=nn.GELU(),
        ...                       normalization='batch', dropout=0.1)
        >>> x = torch.randn(32, 128)
        >>> out = block(x)  # shape: (32, 128)
    """

    def __init__(
        self,
        width: int,
        dim_block: int,
        activation: nn.Module,
        normalization: str,
        dropout: float = 0.0,
    ):
        """
        Initializes the ResidualBlock.

        Args:
            width: Number of neurons in each layer (input and output dimensions
                must match for the residual connection).
            dim_block: Number of linear layers in the block. The actual number
                of transformations is (dim_block - 1).
            activation: Pre-instantiated activation module (e.g., nn.GELU()).
            normalization: Type of normalization layer. Options:
                - 'batch': BatchNorm1d (good for larger batches)
                - 'layer': LayerNorm (good for smaller batches/variable batch sizes)
                - 'none': No normalization
            dropout: Dropout probability applied after each activation.
                Default is 0.0 (no dropout).

        Raises:
            ValueError: If normalization type is not recognized.
        """
        super().__init__()

        layers = []
        for i in range(dim_block - 1):
            layers.append(nn.Linear(width, width))

            # Apply normalization only after the first linear layer
            if i == 0:
                if normalization == "batch":
                    layers.append(nn.BatchNorm1d(width))
                elif normalization == "layer":
                    layers.append(nn.LayerNorm(width))
                elif normalization == "none":
                    pass
                else:
                    raise ValueError(f"Unsupported normalization: {normalization}")

            layers.append(activation)
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch_size, width).

        Returns:
            Output tensor of shape (batch_size, width), computed as
            x + f(x) where f is the learned transformation.
        """
        return x + self.layers(x)


# =============================================================================
# Variational Autoencoder Components
# =============================================================================


class Encoder(nn.Module):
    """
    VAE Encoder that maps input variables to a probabilistic latent space.

    The encoder progressively reduces dimensionality through a series of
    residual blocks, outputting parameters (mean and log-variance) of a
    Gaussian distribution in latent space. This probabilistic formulation
    captures uncertainty in the learned representations.

    Architecture:
        input -> Dropout -> [Linear -> ResidualBlock] x num_blocks -> mean, log_var

    The width of each block decreases linearly from input_dim to latent_dim.

    Attributes:
        input_dim: Dimensionality of input features.
        latent_dim: Dimensionality of the latent space.
        net: Sequential network of linear layers and residual blocks.
        mean: Linear layer outputting latent means.
        log_var: Linear layer outputting latent log-variances.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_blocks: int,
        dim_blocks: int,
        activation: str,
        normalization: str = "batch",
        dropout: float = 0.0,
        input_dropout: float = 0.0,
    ):
        """
        Initializes the Encoder network.

        Args:
            input_dim: Number of input features (socioeconomic indicators).
            latent_dim: Dimensionality of the latent representation.
            num_blocks: Number of residual blocks in the encoder.
            dim_blocks: Depth of each residual block (number of layers).
            activation: Name of activation function (e.g., 'gelu', 'relu').
            normalization: Type of normalization ('batch', 'layer', 'none').
            dropout: Dropout probability within residual blocks.
            input_dropout: Dropout probability applied directly to inputs.
                This acts as a form of data augmentation during training.
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.dim_blocks = dim_blocks
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropout = dropout
        self.input_dropout = input_dropout

        # Compute layer widths: linearly decreasing from input_dim to latent_dim
        step_size = int((input_dim - latent_dim) / num_blocks)
        layer_widths = [input_dim - i * step_size for i in range(num_blocks)]
        layer_widths.append(latent_dim)

        # Build the network
        blocks = [nn.Dropout(input_dropout)]

        for i in range(len(layer_widths) - 1):
            in_features = layer_widths[i]
            out_features = layer_widths[i + 1]

            blocks.append(nn.Linear(in_features, out_features))
            blocks.append(
                ResidualBlock(
                    width=out_features,
                    dim_block=dim_blocks,
                    activation=self.activation,
                    normalization=normalization,
                    dropout=dropout,
                )
            )

        self.net = nn.Sequential(*blocks)

        # Output heads for mean and log-variance
        self.mean = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of (mean, log_var), each of shape (batch_size, latent_dim).
            These parameterize the approximate posterior q(z|x) = N(mean, exp(log_var)).
        """
        h = self.net(x)
        mean = self.mean(h)
        log_var = self.log_var(h)
        return mean, log_var


class Decoder(nn.Module):
    """
    VAE Decoder that reconstructs input variables from latent representations.

    The decoder mirrors the encoder architecture but reverses the information
    flow, progressively increasing dimensionality from latent_dim to input_dim.

    Architecture:
        latent -> [Linear -> ResidualBlock] x num_blocks -> Linear -> reconstruction

    Attributes:
        input_dim: Dimensionality of reconstructed outputs.
        latent_dim: Dimensionality of the latent space.
        net: Sequential network of linear layers and residual blocks.
        out: Final linear layer for reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_blocks: int,
        dim_blocks: int,
        activation: str,
        normalization: str = "batch",
        dropout: float = 0.0,
    ):
        """
        Initializes the Decoder network.

        Args:
            input_dim: Number of output features (matches encoder input_dim).
            latent_dim: Dimensionality of the latent representation.
            num_blocks: Number of residual blocks in the decoder.
            dim_blocks: Depth of each residual block.
            activation: Name of activation function.
            normalization: Type of normalization.
            dropout: Dropout probability within residual blocks.
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.dim_blocks = dim_blocks
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropout = dropout

        # Compute layer widths (reversed from encoder)
        step_size = int((input_dim - latent_dim) / num_blocks)
        layer_widths = [input_dim - i * step_size for i in range(num_blocks)]
        layer_widths.append(latent_dim)
        layer_widths = layer_widths[::-1]  # Reverse: latent_dim -> input_dim

        # Build the network
        blocks = []
        for i in range(len(layer_widths) - 1):
            in_features = layer_widths[i]
            out_features = layer_widths[i + 1]

            blocks.append(nn.Linear(in_features, out_features))
            blocks.append(
                ResidualBlock(
                    width=out_features,
                    dim_block=dim_blocks,
                    activation=self.activation,
                    normalization=normalization,
                    dropout=dropout,
                )
            )

        self.net = nn.Sequential(*blocks)
        self.out = nn.Linear(input_dim, input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent vectors to reconstructed inputs.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        h = self.net(z)
        return self.out(h)


class VAEModel(nn.Module):
    """
    Variational Autoencoder combining encoder and decoder.

    The VAE learns a probabilistic latent representation of high-dimensional
    socioeconomic indicators. Unlike deterministic dimensionality reduction
    (e.g., PCA), the VAE captures both the central structure of the data and
    the variability around it, enabling uncertainty propagation.

    The model is trained to maximize the Evidence Lower Bound (ELBO):
        ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

    Attributes:
        encoder: Encoder network mapping inputs to latent distributions.
        decoder: Decoder network reconstructing inputs from latent samples.

    Example:
        >>> encoder = Encoder(input_dim=100, latent_dim=10, ...)
        >>> decoder = Decoder(input_dim=100, latent_dim=10, ...)
        >>> vae = VAEModel(encoder, decoder)
        >>> x_hat, mean, log_var = vae(x)
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        """
        Initializes the VAE with pre-configured encoder and decoder.

        Args:
            encoder: Encoder network instance.
            decoder: Decoder network instance.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, sample, and decode.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of:
                - x_hat: Reconstructed inputs, shape (batch_size, input_dim)
                - mean: Latent means, shape (batch_size, latent_dim)
                - log_var: Latent log-variances, shape (batch_size, latent_dim)
        """
        mean, log_var = self.encoder(x)

        # Sample from latent distribution using reparameterization trick
        std = torch.exp(0.5 * log_var)
        z = reparameterize(mean, std)

        # Decode the sampled latent
        x_hat = self.decoder(z)

        return x_hat, mean, log_var


# =============================================================================
# Emission Prediction Components
# =============================================================================


class EmissionPredictor(nn.Module):
    """
    Predicts sectoral emission changes with uncertainty quantification.

    This network maps concatenated latent representations and context variables
    to emission predictions (deltas from previous year) along with learned
    uncertainty estimates. The uncertainty-aware formulation improves robustness
    and helps identify sectors where predictions are less reliable.

    Input composition:
        [z_t, context_t, z_{t-1}, context_{t-1}]

    Where z_t is the latent representation at time t and context includes
    climate and macroeconomic variables.

    Attributes:
        output_size: Number of emission sectors (6 for sectoral, 1 for total).
        uncertainty: Whether to output uncertainty estimates.
        net: Main network body with residual blocks.
        output_layer: Linear layer for emission predictions.
        learned_uncertainty: Linear layer for uncertainty estimates (if enabled).

    Example:
        >>> predictor = EmissionPredictor(
        ...     input_dim=2*(10+5),  # 2 * (latent_dim + context_dim)
        ...     output_configs={'output': 'Sectors', 'measure': 'KG_HAB'},
        ...     num_blocks=2, dim_block=2, width_block=128,
        ...     activation='silu', uncertainty=True
        ... )
        >>> delta, uncertainty = predictor(concatenated_input)
    """

    def __init__(
        self,
        input_dim: int,
        output_configs: dict,
        num_blocks: int,
        dim_block: int,
        width_block: int,
        activation: str,
        normalization: str = "batch",
        dropout: float = 0.0,
        uncertainty: bool = True,
    ):
        """
        Initializes the EmissionPredictor.

        Args:
            input_dim: Size of concatenated input (2 * (latent_dim + context_dim)).
            output_configs: Dictionary specifying output configuration:
                - 'output': 'Sectors' for 6 sectors, or 'Total'/'TotalECON'/'TotalHOUSE' for 1
                - 'measure': 'KG_HAB', 'THS_T', or 'both' (doubles output size)
            num_blocks: Number of residual blocks.
            dim_block: Depth of each residual block.
            width_block: Width (number of neurons) in hidden layers.
            activation: Name of activation function.
            normalization: Type of normalization.
            dropout: Dropout probability.
            uncertainty: If True, output uncertainty estimates alongside predictions.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.dim_block = dim_block
        self.width_block = width_block
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropout = dropout
        self.output_configs = output_configs
        self.uncertainty = uncertainty

        # Determine output size based on configuration
        if output_configs["output"] == "Sectors":
            self.output_size = (
                6  # HeatingCooling, Industry, Land, Mobility, Power, Other
            )
        elif output_configs["output"] in ["Total", "TotalECON", "TotalHOUSE"]:
            self.output_size = 1
        else:
            raise ValueError(f"Unknown output type: {output_configs['output']}")

        # Build the network body
        blocks = [nn.Linear(input_dim, width_block)]
        for _ in range(num_blocks):
            blocks.append(
                ResidualBlock(
                    width=width_block,
                    dim_block=dim_block,
                    activation=self.activation,
                    normalization=normalization,
                    dropout=dropout,
                )
            )
        self.net = nn.Sequential(*blocks)

        # Output layer(s)
        output_multiplier = 2 if output_configs.get("measure") == "both" else 1
        self.output_layer = nn.Linear(width_block, self.output_size * output_multiplier)

        # Uncertainty head (optional)
        if uncertainty:
            self.learned_uncertainty = nn.Linear(width_block, self.output_size)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts emission deltas and optionally uncertainty.

        Args:
            x: Concatenated input tensor of shape (batch_size, input_dim),
               containing [z_t, context_t, z_{t-1}, context_{t-1}].

        Returns:
            If uncertainty=True:
                Tuple of (predictions, uncertainty), each shape (batch_size, output_size)
            If uncertainty=False:
                Predictions tensor of shape (batch_size, output_size)
        """
        h = self.net(x)
        predictions = self.output_layer(h)

        if self.uncertainty:
            uncertainty = self.learned_uncertainty(h)
            return predictions, uncertainty
        else:
            return predictions.squeeze(-1)


class FullPredictionModel(nn.Module):
    """
    End-to-end model combining VAE encoding with emission prediction.

    This wrapper class provides a unified interface for the full prediction
    pipeline: encoding inputs to latent space, and predicting emission changes
    from the latent representations combined with context variables.

    The model predicts emission *deltas* (changes from previous year), not
    absolute emission levels. This formulation improves training stability
    and allows the model to focus on year-over-year dynamics.

    Attributes:
        vae: Trained VAE model (encoder weights may be frozen during predictor training).
        predictor: Emission predictor network.
        encoder: Reference to VAE encoder for convenience.
        decoder: Reference to VAE decoder for convenience.

    Example:
        >>> full_model = FullPredictionModel(vae, predictor)
        >>> (delta, uncertainty, recon_t, recon_t1,
        ...  mean_t, mean_t1, logvar_t, logvar_t1) = full_model(x_t, x_t1, c_t, c_t1)
    """

    def __init__(self, vae: VAEModel, predictor: EmissionPredictor):
        """
        Initializes the full prediction model.

        Args:
            vae: Pre-trained VAEModel instance.
            predictor: EmissionPredictor instance (can be untrained).

        Raises:
            AssertionError: If predictor does not have uncertainty=True.
        """
        super().__init__()

        self.vae = vae
        self.predictor = predictor
        self.encoder = vae.encoder
        self.decoder = vae.decoder

        assert self.predictor.uncertainty, (
            "FullPredictionModel requires EmissionPredictor with uncertainty=True"
        )

    def forward(
        self,
        input_current: torch.Tensor,
        input_prev: torch.Tensor,
        context_current: torch.Tensor,
        context_prev: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward pass through encoding and prediction.

        Args:
            input_current: Input features at time t, shape (batch_size, input_dim).
            input_prev: Input features at time t-1, shape (batch_size, input_dim).
            context_current: Context variables at time t, shape (batch_size, context_dim).
            context_prev: Context variables at time t-1, shape (batch_size, context_dim).

        Returns:
            Tuple of 8 tensors:
                - emission_delta: Predicted emission changes, shape (batch_size, num_sectors)
                - emission_uncertainty: Uncertainty estimates, shape (batch_size, num_sectors)
                - recon_current: Reconstructed inputs at t (for VAE monitoring)
                - recon_prev: Reconstructed inputs at t-1 (for VAE monitoring)
                - mean_current: Latent means at t
                - mean_prev: Latent means at t-1
                - log_var_current: Latent log-variances at t
                - log_var_prev: Latent log-variances at t-1
        """
        # Encode both time steps
        mean_current, log_var_current = self.encoder(input_current)
        mean_prev, log_var_prev = self.encoder(input_prev)

        # Sample from latent distributions
        z_current = reparameterize(mean_current, torch.exp(0.5 * log_var_current))
        z_prev = reparameterize(mean_prev, torch.exp(0.5 * log_var_prev))

        # Concatenate for predictor: [z_t, c_t, z_{t-1}, c_{t-1}]
        predictor_input = torch.cat(
            (z_current, context_current, z_prev, context_prev), dim=1
        )

        # Predict emission deltas
        emission_delta, emission_uncertainty = self.predictor(predictor_input)

        # Decode for reconstruction monitoring (useful during training)
        recon_current = self.decoder(z_current)
        recon_prev = self.decoder(z_prev)

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


# =============================================================================
# Latent Space Forecasting Components
# =============================================================================


class LatentForecaster(nn.Module):
    """
    Forecasts future latent states from historical latent representations.

    This model enables autoregressive projection of emissions by predicting
    the next latent state z_t from previous states z_{t-1}, z_{t-2} and
    context variables. The forecasted latents can then be passed through
    the emission predictor for future emission projections.

    The model learns a residual connection: z_t = z_{t-1} + f(inputs),
    which assumes smooth temporal evolution of the latent space.

    Input composition:
        [z_{t-1}, z_{t-2}, context_t, context_{t-1}]

    Attributes:
        latent_dim: Dimensionality of the latent space to forecast.
        net: Main network body with residual blocks.
        output_layer: Linear layer for latent prediction.

    Example:
        >>> forecaster = LatentForecaster(
        ...     input_dim=2*(10+5),  # 2 * (latent_dim + context_dim)
        ...     latent_dim=10,
        ...     num_blocks=3, dim_block=5, width_block=128,
        ...     activation='gelu'
        ... )
        >>> z_next = forecaster(z_t, z_t1, context_next, context_t)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_blocks: int,
        dim_block: int,
        width_block: int,
        activation: str,
        normalization: str = "batch",
        dropout: float = 0.0,
    ):
        """
        Initializes the LatentForecaster.

        Args:
            input_dim: Size of concatenated input (2 * (latent_dim + context_dim)).
            latent_dim: Dimensionality of the latent space.
            num_blocks: Number of residual blocks.
            dim_block: Depth of each residual block.
            width_block: Width of hidden layers.
            activation: Name of activation function.
            normalization: Type of normalization.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.dim_block = dim_block
        self.width_block = width_block
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropout = dropout

        # Build the network
        blocks = [nn.Linear(input_dim, width_block)]
        for _ in range(num_blocks):
            blocks.append(
                ResidualBlock(
                    width=width_block,
                    dim_block=dim_block,
                    activation=self.activation,
                    normalization=normalization,
                    dropout=dropout,
                )
            )
        self.net = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(width_block, latent_dim)

    def forward(
        self,
        latent_current: torch.Tensor,
        latent_prev: torch.Tensor,
        context_next: torch.Tensor,
        context_current: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forecasts the next latent state.

        Args:
            latent_current: Latent at t-1, shape (batch_size, latent_dim).
            latent_prev: Latent at t-2, shape (batch_size, latent_dim).
            context_next: Context at t (the target time), shape (batch_size, context_dim).
            context_current: Context at t-1, shape (batch_size, context_dim).

        Returns:
            Forecasted latent at t, shape (batch_size, latent_dim).
            Computed as: z_{t-1} + f([z_{t-1}, z_{t-2}, c_t, c_{t-1}])
        """
        stacked = torch.cat(
            (latent_current, latent_prev, context_next, context_current), dim=1
        )
        h = self.net(stacked)

        # Residual connection: predict delta from current latent
        return latent_current + self.output_layer(h)


class FullLatentForecastingModel(nn.Module):
    """
    End-to-end model for latent space forecasting from raw inputs.

    This wrapper combines the VAE encoder with the latent forecaster,
    providing a unified interface for predicting future latent states
    directly from input features.

    Attributes:
        vae: VAE model (only encoder is used).
        forecaster: LatentForecaster network.
        encoder: Reference to VAE encoder.

    Example:
        >>> model = FullLatentForecastingModel(vae, forecaster)
        >>> z_forecast = model(x_t, x_t1, context_next, context_t)
    """

    def __init__(self, vae: VAEModel, forecaster: LatentForecaster):
        """
        Initializes the full latent forecasting model.

        Args:
            vae: Pre-trained VAEModel instance.
            forecaster: LatentForecaster instance.
        """
        super().__init__()

        self.vae = vae
        self.forecaster = forecaster
        self.encoder = vae.encoder

    def forward(
        self,
        input_current: torch.Tensor,
        input_prev: torch.Tensor,
        context_next: torch.Tensor,
        context_current: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forecasts future latent state from raw inputs.

        Args:
            input_current: Input features at t-1, shape (batch_size, input_dim).
            input_prev: Input features at t-2, shape (batch_size, input_dim).
            context_next: Context at t (target time), shape (batch_size, context_dim).
            context_current: Context at t-1, shape (batch_size, context_dim).

        Returns:
            Forecasted latent at t, shape (batch_size, latent_dim).
        """
        # Encode inputs to latent distributions
        mean_current, log_var_current = self.encoder(input_current)
        mean_prev, log_var_prev = self.encoder(input_prev)

        # Sample from latent distributions
        z_current = reparameterize(mean_current, torch.exp(0.5 * log_var_current))
        z_prev = reparameterize(mean_prev, torch.exp(0.5 * log_var_prev))

        # Forecast next latent
        return self.forecaster(z_current, z_prev, context_next, context_current)


# =============================================================================
# Loss Functions
# =============================================================================


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes VAE loss: reconstruction + KL divergence.

    The loss corresponds to the negative ELBO (Evidence Lower Bound):
        L = -E[log p(x|z)] + KL(q(z|x) || p(z))

    We use L1 (MAE) for reconstruction as it empirically outperforms MSE
    for heterogeneous socioeconomic data.

    Args:
        x: Ground truth inputs, shape (batch_size, input_dim).
        x_hat: Reconstructed inputs, shape (batch_size, input_dim).
        mean: Latent means, shape (batch_size, latent_dim).
        log_var: Latent log-variances, shape (batch_size, latent_dim).

    Returns:
        Tuple of (reconstruction_loss, kl_divergence), both scalar tensors.
        Total loss = weight_recon * recon_loss + weight_kl * kl_loss

    Note:
        Reconstruction errors are clamped to [0, 5] to prevent unstable
        gradients from outliers during early training.
    """
    # L1 reconstruction loss with clamping for stability
    recon_loss = torch.nn.functional.l1_loss(x_hat, x, reduction="none")
    recon_loss = torch.clamp(recon_loss, 0, 5.0)
    recon_loss = torch.nanmean(recon_loss)

    # KL divergence: KL(N(mean, var) || N(0, I))
    kl_loss = -0.5 * torch.nanmean(1 + log_var - mean.pow(2) - log_var.exp())

    return recon_loss, kl_loss


def uncertainty_aware_mse_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    log_uncertainty: torch.Tensor,
    mode: str = "regular",
) -> torch.Tensor:
    """
    Uncertainty-aware MSE loss for emission prediction.

    This loss function jointly optimizes prediction accuracy and uncertainty
    calibration. It follows the principled heteroscedastic uncertainty
    formulation where the model learns to predict both the mean and variance
    of the output distribution.

    Loss = 0.5 * exp(-log_σ²) * (y - ŷ)² + regularization(log_σ²)

    The regularization term prevents the model from assigning arbitrarily
    high uncertainty to avoid the prediction penalty.

    Args:
        target: Ground truth emissions, shape (batch_size, num_sectors).
        prediction: Predicted emissions, shape (batch_size, num_sectors).
        log_uncertainty: Log-variance uncertainty estimates, shape (batch_size, num_sectors).
        mode: Uncertainty regularization mode:
            - 'regular': 0.5 * log_σ² (standard formulation)
            - 'exponential': tanh(log_σ²) (bounded regularization)
            - 'factor': 0.005 * log_σ² (reduced regularization weight)

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If mode is not recognized.

    Note:
        Squared errors are clamped to [0, 2] to improve training stability
        on outliers.
    """
    # Ensure gradient tracking for uncertainty
    if log_uncertainty.requires_grad and not log_uncertainty.is_leaf:
        log_uncertainty.retain_grad()

    # Squared error with clamping
    squared_error = torch.square(target - prediction)
    squared_error = torch.clamp(squared_error, min=0.0, max=2.0)

    # Uncertainty-weighted loss
    loss = 0.5 * torch.exp(-log_uncertainty) * squared_error

    # Add regularization term based on mode
    if mode == "regular":
        loss = loss + 0.5 * log_uncertainty
    elif mode == "exponential":
        loss = loss + torch.tanh(log_uncertainty)
    elif mode == "factor":
        loss = loss + 0.005 * log_uncertainty
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Use 'regular', 'exponential', or 'factor'."
        )

    return loss.nanmean()


def uncertainty_aware_mae_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    log_uncertainty: torch.Tensor,
    mode: str = "regular",
) -> torch.Tensor:
    """
    Uncertainty-aware MAE loss for latent space forecasting.

    Similar to uncertainty_aware_mse_loss but uses absolute error instead
    of squared error. This is more robust to outliers in the latent space.

    Loss = 0.5 * exp(-log_σ²) * |y - ŷ| + regularization(log_σ²)

    Args:
        target: Target latent means, shape (batch_size, latent_dim).
        prediction: Predicted latents, shape (batch_size, latent_dim).
        log_uncertainty: Target latent log-variances (from encoder),
            shape (batch_size, latent_dim). Used to discount errors in
            high-variance latent dimensions.
        mode: Uncertainty regularization mode:
            - 'regular': 0.5 * log_σ²
            - 'exponential': exp(log_σ²)
            - 'factor': 10 * log_σ²
            - 'quadrexp': exp(log_σ²)²

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If mode is not recognized.
    """
    # Ensure gradient tracking
    if log_uncertainty.requires_grad and not log_uncertainty.is_leaf:
        log_uncertainty.retain_grad()

    # Absolute error with clamping
    abs_error = torch.abs(target - prediction)
    abs_error = torch.clamp(abs_error, min=0.0, max=2.0)

    # Uncertainty-weighted loss
    loss = 0.5 * torch.exp(-log_uncertainty) * abs_error

    # Add regularization
    if mode == "regular":
        loss = loss + 0.5 * log_uncertainty
    elif mode == "exponential":
        loss = loss + torch.exp(log_uncertainty)
    elif mode == "factor":
        loss = loss + 10 * log_uncertainty
    elif mode == "quadrexp":
        loss = loss + torch.exp(log_uncertainty) ** 2
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Use 'regular', 'exponential', 'factor', or 'quadrexp'."
        )

    return loss.nanmean()
