import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from liquid_foundation_model.model.layers.rope import RotaryPositionEmbedding


class CrossAttention(nn.Module):
    """
    Cross-Attention layer for decoder to attend to encoder outputs.
    
    In encoder-decoder architectures, cross-attention allows the decoder
    to attend to the encoder's representation of the input sequence.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        """
        Initialize Cross-Attention layer.
        
        Args:
            hidden_size: Size of the hidden dimension
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads (for GQA)
            head_dim: Dimension of each attention head
            dropout_rate: Dropout rate for attention
            max_seq_len: Maximum sequence length for RoPE
            rope_theta: Base frequency for RoPE
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads
        
        # Query projection (from decoder)
        self.q_proj = nn.Linear(hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        
        # Key/Value projections (from encoder)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE for query (key uses learned positions in cross-attention)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len, rope_theta)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of cross-attention.
        
        Args:
            hidden_states: Decoder input [batch_size, seq_len, hidden_size]
            encoder_hidden_states: Encoder output [batch_size, enc_seq_len, hidden_size]
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape
        enc_seq_len = encoder_hidden_states.shape[1]
        
        # Project inputs
        query_states = self.q_proj(hidden_states)  # [batch, seq_len, num_heads * head_dim]
        key_states = self.k_proj(encoder_hidden_states)  # [batch, enc_seq_len, num_kv_heads * head_dim]
        value_states = self.v_proj(encoder_hidden_states)  # [batch, enc_seq_len, num_kv_heads * head_dim]
        
        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, enc_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, enc_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to query
        query_states = self.rope(query_states, seq_len)
        
        # Repeat keys/values for GQA
        if self.num_key_value_heads != self.num_attention_heads:
            key_states = key_states.repeat_interleave(self.num_queries_per_kv, dim=1)
            value_states = value_states.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value_states)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)
        
        # Project output
        output = self.o_proj(context)
        
        if output_attentions:
            return output, attention_probs
        else:
            return output, None
