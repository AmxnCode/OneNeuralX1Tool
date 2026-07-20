"""
Hybrid Attention Layer (from Ternary Bonsai 27B)

Mix of linear attention (O(n)) and full attention (O(n²)):
- 75% of layers use linear attention (fast, long context)
- 25% of layers use full attention (precise, standard)

This allows 4x longer context (8K → 32K) without memory explosion.

Reference: https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.rope import RotaryPositionEmbedding


class LinearAttention(nn.Module):
    """
    Linear attention: O(n) complexity, no KV cache needed.
    
    Uses kernel approximation: exp(Q·K^T) ≈ φ(Q)·φ(K)^T
    Where φ is a random feature map: φ(x) = elu(x) + 1
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Random feature map: φ(x) = elu(x) + 1
        This approximates exp(x) in kernel space.
        """
        return F.elu(x) + 1
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply feature map
        Q = self.feature_map(Q)  # [batch, seq, heads, head_dim]
        K = self.feature_map(K)
        
        # Transpose for batched matmul
        Q = Q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Linear attention: O(n) complexity
        # Compute KV^T first (head_dim × head_dim), then Q @ (K^T @ V)
        KV = torch.matmul(K.transpose(-1, -2), V)  # [batch, heads, head_dim, head_dim]
        QKV = torch.matmul(Q, KV)  # [batch, heads, seq, head_dim]
        
        # Normalize
        Z = 1.0 / (torch.matmul(Q, K.transpose(-1, -2).sum(dim=-1, keepdim=True)) + 1e-6)
        output = QKV * Z
        
        # Reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output, None


class FullAttention(nn.Module):
    """
    Full attention: O(n²) complexity, standard scaled dot-product.
    
    Used for precise attention when needed.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.num_queries_per_kv = self.num_heads // self.num_key_value_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
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
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        query_states = self.rope(query_states, seq_len)
        key_states = self.rope(key_states, seq_len)
        
        # GQA repeat
        if self.num_key_value_heads != self.num_heads:
            key_states = key_states.repeat_interleave(self.num_queries_per_kv, dim=1)
            value_states = value_states.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Attention output
        output = torch.matmul(attention_probs, value_states)
        
        # Reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output, attention_probs


class HybridAttention(nn.Module):
    """
    Hybrid attention that switches between linear and full attention.
    
    Default: 75% linear (fast) + 25% full (precise)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        use_linear: bool = True,  # True for linear, False for full
    ):
        super().__init__()
        
        if use_linear:
            self.attention = LinearAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout_rate=dropout_rate,
            )
        else:
            self.attention = FullAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
            )
        
        self.use_linear = use_linear
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.attention(hidden_states, attention_mask, output_attentions)


def create_hybrid_attention_layers(
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    num_layers: int,
    linear_ratio: float = 0.75,  # 75% linear, 25% full
    **kwargs,
) -> nn.ModuleList:
    """
    Create hybrid attention layers.
    
    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        num_key_value_heads: Number of KV heads (for GQA)
        num_layers: Total number of layers
        linear_ratio: Fraction of linear attention layers
        **kwargs: Additional arguments for attention
        
    Returns:
        ModuleList of hybrid attention layers
    """
    num_linear = int(num_layers * linear_ratio)
    num_full = num_layers - num_linear
    
    layers = []
    for i in range(num_layers):
        use_linear = i < num_linear
        layer = HybridAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            use_linear=use_linear,
            **kwargs,
        )
        layers.append(layer)
    
    return nn.ModuleList(layers)
