import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.rope import RotaryPositionEmbedding


class EncoderSelfAttention(nn.Module):
    """
    Self-attention with GQA and RoPE for encoder.
    
    Encoder uses bidirectional attention (no causal mask).
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
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len, rope_theta)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        query_states = self.rope(query_states, seq_len)
        key_states = self.rope(key_states, seq_len)
        
        # GQA repeat
        if self.num_key_value_heads != self.num_attention_heads:
            key_states = key_states.repeat_interleave(self.num_queries_per_kv, dim=1)
            value_states = value_states.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply mask (if provided)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply to values
        context = torch.matmul(attention_probs, value_states)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.o_proj(context)
        
        if output_attentions:
            return output, attention_probs
        return output, None


class EncoderBlock(nn.Module):
    """
    Single encoder block with self-attention and gated residual.
    
    Architecture (from Needle):
    - ZCRMSNorm
    - Self-Attention (GQA + RoPE)
    - Gated Residual
    
    No FFN (pure attention) like Needle.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        
        # Zero-centered RMSNorm (ZCRMSNorm)
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        
        # Self-attention
        self.self_attention = EncoderSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        
        # Gated residual
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with gated residual
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        attn_output, attn_weights = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        # Gated residual
        gate_input = torch.cat([residual, attn_output], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        output = residual + gate * attn_output
        output = self.dropout(output)
        
        return output, attn_weights
