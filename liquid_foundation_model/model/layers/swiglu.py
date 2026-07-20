import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    SwiGLU is a variant of GLU (Gated Linear Unit) that uses SwiSH activation
    instead of sigmoid for the gate.
    """
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        """
        Initialize the SwiGLU layer.
        
        Args:
            in_features: Number of input features
            hidden_features: Number of hidden features
            out_features: Number of output features
            bias: Whether to include bias terms
        """
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU layer.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # SwiGLU activation
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = x1 * F.silu(x2)
        
        # Project back to output dimension
        output = self.w3(hidden)
        
        return output