import torch
import torch.nn as nn
import torch.nn.functional as F

from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.simple_liv_conv import SimpleLIVConv


class ConvBlock(nn.Module):
    """
    Convolution block for the Liquid Foundation Model.
    
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
    ):
        """
        Initialize the convolution block.
        
        Args:
            hidden_size: Size of the hidden dimension
            intermediate_size: Size of the intermediate dimension
            kernel_size: Size of the convolution kernel
            dropout_rate: Dropout rate
            layer_norm_epsilon: Epsilon for layer normalization
        """
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        self.conv = SimpleLIVConv(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
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