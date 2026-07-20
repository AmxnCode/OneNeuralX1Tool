import torch
import torch.nn as nn
from typing import Optional


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    RoPE encodes position information by rotating query and key vectors
    in attention, enabling better length generalization.
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            theta: Base frequency for rotation
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute rotation frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute rotation matrix for max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build rotation cache for given sequence length."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        
        # Store cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> tuple:
        """
        Apply rotary position embedding.
        
        Args:
            x: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
            seq_len: Sequence length (if None, use x.shape[2])
            
        Returns:
            Tuple of (rotated_x, rotated_x) for query and key
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        # Build cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        
        # Get cos and sin for this sequence
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        
        # Split x into pairs for rotation
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        
        # Apply rotation
        rotated_x1 = x1 * cos[..., :self.head_dim // 2] - x2 * sin[..., :self.head_dim // 2]
        rotated_x2 = x1 * sin[..., :self.head_dim // 2] + x2 * cos[..., :self.head_dim // 2]
        
        # Concatenate
        rotated_x = torch.cat([rotated_x1, rotated_x2], dim=-1)
        
        return rotated_x
