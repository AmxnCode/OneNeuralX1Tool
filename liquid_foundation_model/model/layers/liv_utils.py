import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable, List

from liquid_foundation_model.model.layers.liv_operator import (
    LIVOperator,
    GatedLIVOperator,
    DoubleGatedLIVConv,
    AdaptiveLIVOperator,
)


def create_liv_operator(
    operator_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of LIV operators.
    
    Args:
        operator_type: Type of LIV operator to create
        input_dim: Dimension of the input
        output_dim: Dimension of the output
        **kwargs: Additional arguments for the specific operator
        
    Returns:
        LIV operator module
    """
    if operator_type == "basic":
        return LIVOperator(input_dim, output_dim, **kwargs)
    elif operator_type == "gated":
        return GatedLIVOperator(input_dim, output_dim, **kwargs)
    elif operator_type == "double_gated_conv":
        return DoubleGatedLIVConv(input_dim, output_dim, **kwargs)
    elif operator_type == "adaptive":
        return AdaptiveLIVOperator(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown LIV operator type: {operator_type}")


class LIVResidualBlock(nn.Module):
    """
    Residual block with a LIV operator.
    
    This block applies a LIV operator followed by a residual connection.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        dropout_rate: float = 0.1,
        liv_operator_type: str = "gated",
        layer_norm: bool = True,
        **liv_kwargs
    ):
        """
        Initialize the LIV residual block.
        
        Args:
            input_dim: Dimension of the input
            hidden_dim: Dimension of the hidden layer (if None, same as input_dim)
            dropout_rate: Dropout rate
            liv_operator_type: Type of LIV operator to use
            layer_norm: Whether to apply layer normalization
            **liv_kwargs: Additional arguments for the LIV operator
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None
        
        # LIV operator
        self.liv = create_liv_operator(
            operator_type=liv_operator_type,
            input_dim=input_dim,
            output_dim=self.hidden_dim,
            **liv_kwargs
        )
        
        # Output projection (if hidden_dim != input_dim)
        self.output_proj = nn.Linear(self.hidden_dim, input_dim) if self.hidden_dim != input_dim else None
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LIV residual block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, input_dim)
        """
        # Apply layer normalization if needed
        if self.layer_norm is not None:
            x_norm = self.layer_norm(x)
        else:
            x_norm = x
        
        # Apply LIV operator
        hidden = self.liv(x_norm)
        
        # Apply output projection if needed
        if self.output_proj is not None:
            hidden = self.output_proj(hidden)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Add residual connection
        output = x + hidden
        
        return output


class LIVSequential(nn.Module):
    """
    Sequential container for LIV operators.
    
    This module allows for creating a sequence of LIV operators with
    different configurations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = None,
        liv_operator_types: List[str] = None,
        dropout_rate: float = 0.1,
        residual: bool = True,
        layer_norm: bool = True,
        **liv_kwargs
    ):
        """
        Initialize the LIV sequential container.
        
        Args:
            input_dim: Dimension of the input
            hidden_dims: List of hidden dimensions for each layer
            output_dim: Dimension of the output (if None, same as input_dim)
            liv_operator_types: List of LIV operator types for each layer
            dropout_rate: Dropout rate
            residual: Whether to use residual connections
            layer_norm: Whether to apply layer normalization
            **liv_kwargs: Additional arguments for the LIV operators
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim or input_dim
        self.residual = residual
        
        # Default operator type if not specified
        if liv_operator_types is None:
            liv_operator_types = ["gated"] * len(hidden_dims)
        
        # Ensure operator_types has the same length as hidden_dims
        if len(liv_operator_types) != len(hidden_dims):
            raise ValueError(
                f"liv_operator_types must have the same length as hidden_dims, "
                f"got {len(liv_operator_types)} and {len(hidden_dims)}"
            )
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        if residual:
            self.layers.append(
                LIVResidualBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dims[0],
                    dropout_rate=dropout_rate,
                    liv_operator_type=liv_operator_types[0],
                    layer_norm=layer_norm,
                    **liv_kwargs
                )
            )
        else:
            self.layers.append(
                create_liv_operator(
                    operator_type=liv_operator_types[0],
                    input_dim=input_dim,
                    output_dim=hidden_dims[0],
                    **liv_kwargs
                )
            )
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            if residual:
                self.layers.append(
                    LIVResidualBlock(
                        input_dim=hidden_dims[i-1],
                        hidden_dim=hidden_dims[i],
                        dropout_rate=dropout_rate,
                        liv_operator_type=liv_operator_types[i],
                        layer_norm=layer_norm,
                        **liv_kwargs
                    )
                )
            else:
                self.layers.append(
                    create_liv_operator(
                        operator_type=liv_operator_types[i],
                        input_dim=hidden_dims[i-1],
                        output_dim=hidden_dims[i],
                        **liv_kwargs
                    )
                )
        
        # Output layer (if output_dim != last hidden_dim)
        if self.output_dim != hidden_dims[-1]:
            self.output_proj = nn.Linear(hidden_dims[-1], self.output_dim)
        else:
            self.output_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LIV sequential container.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply output projection if needed
        if self.output_proj is not None:
            x = self.output_proj(x)
        
        return x