import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm is a simpler alternative to LayerNorm that normalizes by the RMS
    of the inputs, without centering (subtracting the mean).
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm layer.
        
        Args:
            hidden_size: Size of the hidden dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Normalized tensor of the same shape
        """
        # Calculate RMS along the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_normalized = x / rms
        x_scaled = x_normalized * self.weight
        
        return x_scaled