import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable


class LIVOperator(nn.Module):
    """
    Linear Input-Varying (LIV) operator.
    
    A LIV operator is a linear operator whose weights are generated on-the-fly
    from the input it is acting on, allowing for input-dependent computation.
    
    This implementation follows the equation: y = T(x) * x
    where T is an input-dependent weight matrix.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_generator_dim: int = None,
        bias: bool = True,
        activation: nn.Module = None,
        weight_init_std: float = 0.02,
        use_sequence_mean: bool = True,
        efficient_forward: bool = True,
    ):
        """
        Initialize the LIV operator.
        
        Args:
            input_dim: Dimension of the input
            output_dim: Dimension of the output
            weight_generator_dim: Dimension of the weight generator network
            bias: Whether to include a bias term
            activation: Activation function to use in the weight generator
            weight_init_std: Standard deviation for weight initialization
            use_sequence_mean: Whether to use the mean of the sequence for weight generation
            efficient_forward: Whether to use an efficient implementation of the forward pass
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator_dim = weight_generator_dim or max(input_dim // 2, 32)
        self.bias = bias
        self.activation = activation or nn.GELU()
        self.weight_init_std = weight_init_std
        self.use_sequence_mean = use_sequence_mean
        self.efficient_forward = efficient_forward
        
        # Weight generator network
        self.weight_generator = nn.Sequential(
            nn.Linear(input_dim, self.weight_generator_dim),
            self.activation,
            nn.Linear(self.weight_generator_dim, input_dim * output_dim),
        )
        
        if bias:
            self.bias_generator = nn.Sequential(
                nn.Linear(input_dim, self.weight_generator_dim),
                self.activation,
                nn.Linear(self.weight_generator_dim, output_dim),
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.weight_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _get_weight_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the representation used for weight generation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Representation tensor of shape (batch_size, input_dim)
        """
        if self.use_sequence_mean:
            # Use the mean of the sequence
            return x.mean(dim=1)  # (batch_size, input_dim)
        else:
            # Use the last token in the sequence
            return x[:, -1, :]  # (batch_size, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LIV operator.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get representation for weight generation
        x_repr = self._get_weight_representation(x)  # (batch_size, input_dim)
        
        # Generate weights
        weights = self.weight_generator(x_repr)  # (batch_size, input_dim * output_dim)
        weights = weights.view(batch_size, self.input_dim, self.output_dim)  # (batch_size, input_dim, output_dim)
        
        # Apply the input-varying linear transformation
        if self.efficient_forward:
            # Efficient implementation using batch matrix multiplication
            output = torch.bmm(x.view(-1, 1, self.input_dim), 
                              weights.repeat(seq_len, 1, 1, 1).view(-1, self.input_dim, self.output_dim))
            output = output.view(batch_size, seq_len, self.output_dim)
        else:
            # Original implementation (more readable but less efficient)
            outputs = []
            for i in range(batch_size):
                output = torch.matmul(x[i], weights[i])  # (seq_len, output_dim)
                outputs.append(output)
            output = torch.stack(outputs)  # (batch_size, seq_len, output_dim)
        
        # Add bias if needed
        if self.bias:
            bias = self.bias_generator(x_repr).unsqueeze(1)  # (batch_size, 1, output_dim)
            output = output + bias
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string representation of the module."""
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, " \
               f"weight_generator_dim={self.weight_generator_dim}, bias={self.bias}, " \
               f"use_sequence_mean={self.use_sequence_mean}"


class GatedLIVOperator(nn.Module):
    """
    Gated Linear Input-Varying (LIV) operator.
    
    This extends the basic LIV operator with a gating mechanism that controls
    how much of the input passes through the transformation.
    
    The gating mechanism follows the equation: y = T(x) * x * σ(G(x))
    where T is the input-dependent weight matrix, G is the gate projection,
    and σ is the sigmoid activation function.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_generator_dim: int = None,
        bias: bool = True,
        activation: nn.Module = None,
        gate_activation: Callable = F.sigmoid,
        weight_init_std: float = 0.02,
        use_sequence_mean: bool = True,
        efficient_forward: bool = True,
    ):
        """
        Initialize the Gated LIV operator.
        
        Args:
            input_dim: Dimension of the input
            output_dim: Dimension of the output
            weight_generator_dim: Dimension of the weight generator network
            bias: Whether to include a bias term
            activation: Activation function to use in the weight generator
            gate_activation: Activation function to use for the gate
            weight_init_std: Standard deviation for weight initialization
            use_sequence_mean: Whether to use the mean of the sequence for weight generation
            efficient_forward: Whether to use an efficient implementation of the forward pass
        """
        super().__init__()
        self.liv = LIVOperator(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_generator_dim=weight_generator_dim,
            bias=bias,
            activation=activation,
            weight_init_std=weight_init_std,
            use_sequence_mean=use_sequence_mean,
            efficient_forward=efficient_forward,
        )
        self.gate = nn.Linear(input_dim, output_dim)
        self.gate_activation = gate_activation
        self.weight_init_std = weight_init_std
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.weight_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Gated LIV operator.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Apply LIV transformation
        transformed = self.liv(x)
        
        # Apply gating mechanism
        gate_values = self.gate_activation(self.gate(x))
        
        # Element-wise multiplication with gate values
        output = transformed * gate_values
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string representation of the module."""
        return f"input_dim={self.liv.input_dim}, output_dim={self.liv.output_dim}, " \
               f"weight_generator_dim={self.liv.weight_generator_dim}, bias={self.liv.bias}, " \
               f"gate_activation={self.gate_activation.__name__}"


class DoubleGatedLIVConv(nn.Module):
    """
    Double-gated short-range LIV convolution block, as used in LFM2.
    
    This implements the block structure:
    
    def lfm2_conv(x):
      B, C, x = linear(x)  # input projection
      x = B*x              # gating (gate depends on input)
      x = conv(x)          # short conv
      x = C*x              # gating
      x = linear(x)
      return x
    
    This is a specialized form of the LIV operator where the weights are
    generated through a combination of gating and convolution operations.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        kernel_size: int = 4,
        dropout_rate: float = 0.1,
        groups: int = None,
        weight_init_std: float = 0.02,
        use_bias: bool = True,
        activation: Callable = F.sigmoid,
    ):
        """
        Initialize the double-gated LIV convolution block.
        
        Args:
            hidden_size: Size of the hidden dimension
            intermediate_size: Size of the intermediate dimension
            kernel_size: Size of the convolution kernel
            dropout_rate: Dropout rate
            groups: Number of groups for grouped convolution (default: hidden_size for depthwise)
            weight_init_std: Standard deviation for weight initialization
            use_bias: Whether to use bias in linear layers
            activation: Activation function to use for gating
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.kernel_size = kernel_size
        self.groups = groups or hidden_size  # Default to depthwise convolution
        self.weight_init_std = weight_init_std
        self.use_bias = use_bias
        self.activation = activation
        
        # Input projection
        self.input_projection = nn.Linear(hidden_size, intermediate_size * 2 + hidden_size, bias=use_bias)
        
        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=self.groups,
            bias=use_bias,
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.weight_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the double-gated LIV convolution block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Input projection
        projection = self.input_projection(x)
        
        # Split the projection into B, C, and x_proj parts
        # Ensure the split sizes match the actual dimensions
        split_sizes = [
            self.intermediate_size, 
            self.intermediate_size, 
            self.hidden_size
        ]
        
        # Check if the projection size matches the expected total size
        expected_size = sum(split_sizes)
        if projection.size(-1) != expected_size:
            # Adjust split sizes if there's a mismatch
            total_size = projection.size(-1)
            ratio = total_size / expected_size
            split_sizes = [int(size * ratio) for size in split_sizes]
            # Ensure the sum of split sizes equals the total size
            split_sizes[-1] = total_size - sum(split_sizes[:-1])
        
        B, C, x_proj = torch.split(projection, split_sizes, dim=-1)
        
        # First gating
        x_gated = x_proj * self.activation(B)
        
        # Apply convolution
        # We need to transpose for conv1d which expects [batch, channels, seq_len]
        batch_size, seq_len, hidden_size = x_gated.shape
        x_conv = x_gated.transpose(1, 2)  # [batch, hidden_size, seq_len]
        
        # Apply padding and convolution
        x_conv = self.conv(x_conv)
        
        # Trim to original sequence length (remove extra padding)
        x_conv = x_conv[:, :, :seq_len]
        
        # Transpose back to [batch, seq_len, hidden_size]
        x_conv = x_conv.transpose(1, 2)
        
        # Second gating
        x_gated2 = x_conv * self.activation(C)
        
        # Output projection
        output = self.output_projection(x_gated2)
        output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string representation of the module."""
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, " \
               f"kernel_size={self.kernel_size}, groups={self.groups}, " \
               f"use_bias={self.use_bias}"


class AdaptiveLIVOperator(nn.Module):
    """
    Adaptive Linear Input-Varying (LIV) operator.
    
    This is a more general implementation of the LIV operator that can adapt
    to different types of operations (linear, convolution, attention) based
    on the input.
    
    The operator follows the equation: y = α(x) * T₁(x) + β(x) * T₂(x) * x + γ(x)
    where α, β, γ are input-dependent coefficients, and T₁, T₂ are input-dependent
    transformations.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None,
        num_modes: int = 3,  # Number of operation modes to mix
        dropout_rate: float = 0.1,
        weight_init_std: float = 0.02,
        use_bias: bool = True,
    ):
        """
        Initialize the Adaptive LIV operator.
        
        Args:
            input_dim: Dimension of the input
            output_dim: Dimension of the output
            hidden_dim: Dimension of the hidden layers
            num_modes: Number of operation modes to mix
            dropout_rate: Dropout rate
            weight_init_std: Standard deviation for weight initialization
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(input_dim // 2, 32)
        self.num_modes = num_modes
        self.weight_init_std = weight_init_std
        self.use_bias = use_bias
        
        # Mode selection network
        self.mode_selector = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_modes, bias=use_bias),
            nn.Softmax(dim=-1),
        )
        
        # Transformation networks for each mode
        self.transformations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim, bias=use_bias),
                nn.GELU(),
                nn.Linear(self.hidden_dim, input_dim * output_dim, bias=use_bias),
            )
            for _ in range(num_modes)
        ])
        
        # Bias term
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.weight_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Adaptive LIV operator.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get representation for mode selection (use mean of sequence)
        x_mean = x.mean(dim=1)  # (batch_size, input_dim)
        
        # Select operation modes
        mode_weights = self.mode_selector(x_mean)  # (batch_size, num_modes)
        
        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.output_dim, device=x.device)
        
        # Apply each transformation with its corresponding weight
        for i in range(self.num_modes):
            # Generate transformation weights
            trans_weights = self.transformations[i](x_mean)  # (batch_size, input_dim * output_dim)
            trans_weights = trans_weights.view(batch_size, self.input_dim, self.output_dim)  # (batch_size, input_dim, output_dim)
            
            # Apply transformation
            mode_output = torch.bmm(x.view(batch_size * seq_len, 1, self.input_dim),
                                   trans_weights.repeat(seq_len, 1, 1, 1).view(batch_size * seq_len, self.input_dim, self.output_dim))
            mode_output = mode_output.view(batch_size, seq_len, self.output_dim)
            
            # Weight by mode selection
            mode_output = mode_output * mode_weights[:, i].view(batch_size, 1, 1)
            
            # Add to output
            output = output + mode_output
        
        # Add bias if needed
        if self.use_bias:
            output = output + self.bias
        
        # Apply dropout
        output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string representation of the module."""
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, " \
               f"hidden_dim={self.hidden_dim}, num_modes={self.num_modes}, " \
               f"use_bias={self.use_bias}"