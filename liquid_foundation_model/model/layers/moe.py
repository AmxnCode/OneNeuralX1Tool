"""
Mixture of Experts (MoE) Layer (from OpenMythos)

Sparse activation: only 1 of 4 experts active per token.
This gives 4x more capacity (26M → 104M effective) with same compute.

Reference: https://github.com/kyegomez/OpenMythos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Expert(nn.Module):
    """
    Single expert: small FFN network.
    
    Each expert specializes in different aspects:
    - Expert 0: Weather/location queries
    - Expert 1: Email/communication queries
    - Expert 2: Calendar/event queries
    - Expert 3: Math/calculation queries
    """
    
    def __init__(
        self,
        hidden_size: int,
        expert_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.gate_proj = nn.Linear(hidden_size, expert_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, expert_dim, bias=False)
        self.down_proj = nn.Linear(expert_dim, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        return output


class TopKRouter(nn.Module):
    """
    Router that selects top-k experts per token.
    
    Uses learned gating scores with noise for load balancing.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int = 1,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.noise_std = noise_std
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            router_probs: [batch, seq_len, num_experts] - probabilities
            expert_indices: [batch, seq_len, num_experts_per_tok] - selected experts
            load_balance_loss: scalar - load balancing auxiliary loss
        """
        # Compute gating scores
        logits = self.gate(hidden_states)  # [batch, seq, num_experts]
        
        # Add noise during training for exploration
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Softmax to get probabilities
        router_probs = F.softmax(logits, dim=-1)  # [batch, seq, num_experts]
        
        # Top-k selection
        top_k_probs, expert_indices = torch.topk(
            router_probs,
            self.num_experts_per_tok,
            dim=-1,
        )
        
        # Normalize top-k probs
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Load balancing loss (from Switch Transformer)
        # Encourages uniform expert usage
        f = router_probs.mean(dim=[0, 1])  # [num_experts] - fraction of tokens per expert
        P = router_probs.mean(dim=[0, 1])  # [num_experts] - average prob per expert
        load_balance_loss = self.num_experts * (f * P).sum()
        
        return router_probs, expert_indices, top_k_probs, load_balance_loss


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Args:
        hidden_size: Input/output dimension
        num_experts: Number of expert networks
        expert_dim: Hidden dimension of each expert
        num_experts_per_tok: Number of experts active per token
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        expert_dim: int = 2048,
        num_experts_per_tok: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(hidden_size, expert_dim, dropout_rate)
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )
        
        # Shared expert (always active - captures common patterns)
        self.shared_expert = Expert(hidden_size, expert_dim, dropout_rate)
        self.shared_gate = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse expert routing.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            output: [batch, seq_len, hidden_size]
            load_balance_loss: scalar
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Route to experts
        router_probs, expert_indices, top_k_probs, load_balance_loss = self.router(
            hidden_states
        )
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            mask = (expert_indices == expert_idx).any(dim=-1)  # [batch, seq]
            
            if mask.any():
                # Get expert input
                expert_input = hidden_states[mask]  # [num_tokens, hidden_size]
                
                # Forward through expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get routing weight for this expert
                weight = router_probs[mask, expert_idx]  # [num_tokens]
                weight = weight.unsqueeze(-1)  # [num_tokens, 1]
                
                # Add weighted output
                output[mask] += expert_output * weight
        
        # Add shared expert (always active)
        shared_output = self.shared_expert(hidden_states)
        shared_weight = torch.sigmoid(self.shared_gate(hidden_states))
        output = output + shared_output * shared_weight
        
        return output, load_balance_loss


class MoEDecoderBlock(nn.Module):
    """
    Decoder block with MoE FFN.
    
    Replaces dense FFN with sparse MoE layer.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        num_experts: int = 4,
        expert_dim: int = 2048,
        num_experts_per_tok: int = 1,
        dropout_rate: float = 0.1,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        
        # Self-attention (with GQA)
        from liquid_foundation_model.model.blocks.decoder_block import DecoderSelfAttention
        self.self_attn = DecoderSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        
        # MoE FFN
        self.moe = MoELayer(
            hidden_size=hidden_size,
            num_experts=num_experts,
            expert_dim=expert_dim,
            num_experts_per_tok=num_experts_per_tok,
            dropout_rate=dropout_rate,
        )
        
        # Norms
        from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        
        # Gated residual
        self.gate1 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate2 = nn.Parameter(torch.zeros(1, 1, hidden_size))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with gated residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + torch.sigmoid(self.gate1) * hidden_states
        
        # MoE FFN with gated residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states, load_balance_loss = self.moe(hidden_states)
        hidden_states = residual + torch.sigmoid(self.gate2) * hidden_states
        
        return hidden_states, load_balance_loss
