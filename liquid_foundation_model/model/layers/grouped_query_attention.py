import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation.
    
    GQA reduces the computational cost of attention by sharing key and value
    projections across multiple query heads.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize the GQA layer.
        
        Args:
            hidden_size: Size of the hidden dimension
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads (less than or equal to num_attention_heads)
            head_dim: Dimension of each attention head (if None, computed as hidden_size / num_attention_heads)
            dropout_rate: Dropout probability for attention weights
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by num_attention_heads {num_attention_heads}"
            )
        
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads {num_key_value_heads} cannot be larger than num_attention_heads {num_attention_heads}"
            )
        
        # Number of queries per key/value
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        """Reshape tensor for attention computation."""
        return tensor.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GQA layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, 1, seq_len)
            position_bias: Optional position bias
            output_attentions: Whether to return attention weights
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, hidden_size)
            attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project inputs to queries, keys, and values
        query_states = self.q_proj(hidden_states)  # (batch_size, seq_len, num_attention_heads * head_dim)
        key_states = self.k_proj(hidden_states)    # (batch_size, seq_len, num_key_value_heads * head_dim)
        value_states = self.v_proj(hidden_states)  # (batch_size, seq_len, num_key_value_heads * head_dim)
        
        # Reshape for attention computation
        query_states = self._shape(query_states, seq_len, batch_size)  # (batch_size, num_attention_heads, seq_len, head_dim)
        key_states = self._shape(key_states, seq_len, batch_size)      # (batch_size, num_key_value_heads, seq_len, head_dim)
        value_states = self._shape(value_states, seq_len, batch_size)  # (batch_size, num_key_value_heads, seq_len, head_dim)
        
        # Repeat keys and values for each query head
        if self.num_key_value_heads != self.num_attention_heads:
            key_states = key_states.repeat_interleave(self.num_queries_per_kv, dim=1)
            value_states = value_states.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))  # (batch_size, num_attention_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Add position bias if provided
        if position_bias is not None:
            attention_scores = attention_scores + position_bias
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value_states)  # (batch_size, num_attention_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous()         # (batch_size, seq_len, num_attention_heads, head_dim)
        context = context.reshape(batch_size, seq_len, -1)     # (batch_size, seq_len, num_attention_heads * head_dim)
        
        # Project back to hidden size
        output = self.o_proj(context)
        
        if output_attentions:
            return output, attention_probs
        else:
            return output, None