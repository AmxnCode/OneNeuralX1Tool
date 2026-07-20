import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLIVConv(nn.Module):
    """
    A simplified version of the double-gated LIV convolution block for testing.
    
    This implementation avoids the complex tensor splitting and reshaping
    that was causing issues in the original implementation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        kernel_size: int = 4,
        dropout_rate: float = 0.1,
        use_bias: bool = True,
    ):
        """
        Initialize the simplified LIV convolution block.
        
        Args:
            hidden_size: Size of the hidden dimension
            intermediate_size: Size of the intermediate dimension
            kernel_size: Size of the convolution kernel
            dropout_rate: Dropout rate
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.kernel_size = kernel_size
        
        # Input projection
        self.input_projection = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # First gate
        self.gate1 = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size,  # Depthwise convolution
            bias=use_bias,
        )
        
        # Second gate
        self.gate2 = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the simplified LIV convolution block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Input projection
        x_proj = self.input_projection(x)
        
        # First gating
        gate1 = torch.sigmoid(self.gate1(x))
        x_gated = x_proj * gate1
        
        # Apply convolution
        batch_size, seq_len, hidden_size = x_gated.shape
        x_conv = x_gated.transpose(1, 2)  # [batch, hidden_size, seq_len]
        
        # Apply convolution
        x_conv = self.conv(x_conv)
        
        # Trim to original sequence length (remove extra padding)
        x_conv = x_conv[:, :, :seq_len]
        
        # Transpose back to [batch, seq_len, hidden_size]
        x_conv = x_conv.transpose(1, 2)
        
        # Second gating
        gate2 = torch.sigmoid(self.gate2(x))
        x_gated2 = x_conv * gate2
        
        # Output projection
        output = self.output_projection(x_gated2)
        output = self.dropout(output)
        
        return output