import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable, List

from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.liv_operator import DoubleGatedLIVConv


class ConvBlock(nn.Module):
    """
    Basic convolution block for the Liquid Foundation Model.
    
    This block consists of:
    1. RMSNorm
    2. Double-gated LIV convolution
    3. Residual connection
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        kernel_size: int = 4,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        groups: int = None,
        use_bias: bool = True,
        activation: Callable = F.sigmoid,
    ):
        """
        Initialize the convolution block.
        
        Args:
            hidden_size: Size of the hidden dimension
            intermediate_size: Size of the intermediate dimension
            kernel_size: Size of the convolution kernel
            dropout_rate: Dropout rate
            layer_norm_epsilon: Epsilon for layer normalization
            groups: Number of groups for grouped convolution (default: hidden_size for depthwise)
            use_bias: Whether to use bias in linear layers
            activation: Activation function to use for gating
        """
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        self.conv = DoubleGatedLIVConv(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            groups=groups,
            use_bias=use_bias,
            activation=activation,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolution block.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Apply layer normalization
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Apply convolution
        hidden_states = self.conv(hidden_states)
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        return hidden_states


class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale convolution block for the Liquid Foundation Model.
    
    This block uses multiple convolution kernels with different sizes
    to capture patterns at different scales.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        use_bias: bool = True,
    ):
        """
        Initialize the multi-scale convolution block.
        
        Args:
            hidden_size: Size of the hidden dimension
            intermediate_size: Size of the intermediate dimension
            kernel_sizes: List of kernel sizes for different scales
            dropout_rate: Dropout rate
            layer_norm_epsilon: Epsilon for layer normalization
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        
        # Create convolution layers for each kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size // len(kernel_sizes),
                kernel_size=k,
                padding=k // 2,
                groups=hidden_size // len(kernel_sizes),  # Depthwise convolution
                bias=use_bias,
            )
            for k in kernel_sizes
        ])
        
        # Input and output projections
        self.input_projection = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-scale convolution block.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Apply layer normalization
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Apply input projection
        hidden_states = self.input_projection(hidden_states)
        
        # Transpose for convolution
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)  # [batch, hidden_size, seq_len]
        
        # Apply convolutions at different scales
        outputs = []
        for conv in self.convs:
            outputs.append(conv(hidden_states))
        
        # Concatenate outputs from different scales
        hidden_states = torch.cat(outputs, dim=1)
        
        # Transpose back
        hidden_states = hidden_states.transpose(1, 2)  # [batch, seq_len, hidden_size]
        
        # Apply output projection
        hidden_states = self.output_projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        return hidden_states


class DilatedConvBlock(nn.Module):
    """
    Dilated convolution block for the Liquid Foundation Model.
    
    This block uses dilated convolutions to increase the receptive field
    without increasing the number of parameters.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        kernel_size: int = 3,
        dilation_rates: List[int] = [1, 2, 4],
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        use_bias: bool = True,
    ):
        """
        Initialize the dilated convolution block.
        
        Args:
            hidden_size: Size of the hidden dimension
            intermediate_size: Size of the intermediate dimension
            kernel_size: Size of the convolution kernel
            dilation_rates: List of dilation rates
            dropout_rate: Dropout rate
            layer_norm_epsilon: Epsilon for layer normalization
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        
        # Create convolution layers for each dilation rate
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=(kernel_size // 2) * d,
                dilation=d,
                groups=hidden_size,  # Depthwise convolution
                bias=use_bias,
            )
            for d in dilation_rates
        ])
        
        # Input and output projections
        self.input_projection = nn.Linear(hidden_size, hidden_size * len(dilation_rates), bias=use_bias)
        self.output_projection = nn.Linear(hidden_size * len(dilation_rates), hidden_size, bias=use_bias)
        
        # Activation
        self.activation = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the dilated convolution block.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Apply layer normalization
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Apply input projection
        hidden_states = self.input_projection(hidden_states)
        
        # Split for different dilation rates
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, seq_len, len(self.convs), -1)
        hidden_states = hidden_states.permute(0, 2, 3, 1)  # [batch, num_convs, hidden_size, seq_len]
        hidden_states = hidden_states.reshape(batch_size * len(self.convs), -1, seq_len)
        
        # Apply dilated convolutions
        outputs = []
        for i, conv in enumerate(self.convs):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            outputs.append(conv(hidden_states[start_idx:end_idx]))
        
        # Concatenate outputs
        hidden_states = torch.cat(outputs, dim=1)
        
        # Transpose back
        hidden_states = hidden_states.transpose(1, 2)  # [batch, seq_len, hidden_size * num_convs]
        
        # Apply activation
        hidden_states = self.activation(hidden_states)
        
        # Apply output projection
        hidden_states = self.output_projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        return hidden_states


class GatedConvBlock(nn.Module):
    """
    Gated convolution block for the Liquid Foundation Model.
    
    This block uses a gating mechanism to control information flow
    through the convolution layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        kernel_size: int = 4,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        use_bias: bool = True,
    ):
        """
        Initialize the gated convolution block.
        
        Args:
            hidden_size: Size of the hidden dimension
            intermediate_size: Size of the intermediate dimension
            kernel_size: Size of the convolution kernel
            dropout_rate: Dropout rate
            layer_norm_epsilon: Epsilon for layer normalization
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        
        # Input projection
        self.input_projection = nn.Linear(hidden_size, intermediate_size * 2, bias=use_bias)
        
        # Convolution layers for value and gate
        self.value_conv = nn.Conv1d(
            in_channels=intermediate_size,
            out_channels=intermediate_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=intermediate_size,  # Depthwise convolution
            bias=use_bias,
        )
        
        self.gate_conv = nn.Conv1d(
            in_channels=intermediate_size,
            out_channels=intermediate_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=intermediate_size,  # Depthwise convolution
            bias=use_bias,
        )
        
        # Output projection
        self.output_projection = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the gated convolution block.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Apply layer normalization
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Apply input projection
        hidden_states = self.input_projection(hidden_states)
        value, gate = torch.chunk(hidden_states, 2, dim=-1)
        
        # Transpose for convolution
        batch_size, seq_len, dim = value.shape
        value = value.transpose(1, 2)  # [batch, dim, seq_len]
        gate = gate.transpose(1, 2)  # [batch, dim, seq_len]
        
        # Apply convolutions
        value = self.value_conv(value)
        gate = self.gate_conv(gate)
        
        # Apply gating
        gate = torch.sigmoid(gate)
        hidden_states = value * gate
        
        # Transpose back
        hidden_states = hidden_states.transpose(1, 2)  # [batch, seq_len, dim]
        
        # Apply output projection
        hidden_states = self.output_projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        return hidden_states


def create_conv_block(
    block_type: str,
    hidden_size: int,
    intermediate_size: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of convolution blocks.
    
    Args:
        block_type: Type of convolution block to create
        hidden_size: Size of the hidden dimension
        intermediate_size: Size of the intermediate dimension
        **kwargs: Additional arguments for the specific block
        
    Returns:
        Convolution block module
    """
    if block_type == "basic":
        return ConvBlock(hidden_size, intermediate_size, **kwargs)
    elif block_type == "multi_scale":
        return MultiScaleConvBlock(hidden_size, intermediate_size, **kwargs)
    elif block_type == "dilated":
        return DilatedConvBlock(hidden_size, intermediate_size, **kwargs)
    elif block_type == "gated":
        return GatedConvBlock(hidden_size, intermediate_size, **kwargs)
    else:
        raise ValueError(f"Unknown convolution block type: {block_type}")