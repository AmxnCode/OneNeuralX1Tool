import torch
import torch.nn as nn

from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.grouped_query_attention import GroupedQueryAttention
from liquid_foundation_model.model.layers.swiglu import SwiGLU


class AttentionBlock(nn.Module):
    """
    Attention block for the Liquid Foundation Model.
    
    This block consists of:
    1. RMSNorm + Grouped Query Attention + Residual
    2. RMSNorm + SwiGLU + Residual
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
    ):
        """
        Initialize the attention block.
        
        Args:
            hidden_size: Size of the hidden dimension
            intermediate_size: Size of the intermediate dimension
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads
            dropout_rate: Dropout rate for SwiGLU
            attention_dropout_rate: Dropout rate for attention
            layer_norm_epsilon: Epsilon for layer normalization
        """
        super().__init__()
        # Attention sub-block
        self.norm1 = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_rate=attention_dropout_rate,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Feed-forward sub-block
        self.norm2 = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = SwiGLU(
            in_features=hidden_size,
            hidden_features=intermediate_size,
            out_features=hidden_size,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the attention block.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, 1, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Attention sub-block
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attention_output, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout1(attention_output) + residual
        
        # Feed-forward sub-block
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = self.dropout2(mlp_output) + residual
        
        return hidden_states