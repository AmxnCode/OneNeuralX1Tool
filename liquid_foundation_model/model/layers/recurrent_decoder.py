"""
Recurrent Decoder with ACT Halting (from OpenMythos)

Key innovations:
1. Recurrent decoder: Same weights looped 2-4x for deeper reasoning
2. ACT (Adaptive Computation Time): Learned early exit per position
3. Loop-index embedding: Different computation per loop iteration

This allows 26M params to behave like 100M+ params.

Reference: https://github.com/kyegomez/OpenMythos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.blocks.decoder_block import DecoderBlock


class LoopIndexEmbedding(nn.Module):
    """
    Embed loop iteration index to differentiate computation across loops.
    
    Similar to RoPE but for loop iterations, not sequence positions.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_loops: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_loops, hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        loop_idx: int,
    ) -> torch.Tensor:
        """
        Add loop index embedding to hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            loop_idx: Current loop iteration (0, 1, 2, ...)
            
        Returns:
            hidden_states + loop_embedding
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create loop embedding
        loop_emb = self.embedding(
            torch.tensor([loop_idx], device=hidden_states.device)
        )  # [1, hidden_size]
        
        # Expand to match hidden states
        loop_emb = loop_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, hidden_size]
        loop_emb = loop_emb.expand(-1, seq_len, -1)  # [batch, seq_len, hidden_size]
        
        return hidden_states + loop_emb


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time (ACT) halting mechanism.
    
    Learns when to stop looping based on input complexity:
    - Simple queries: halt after 1 loop
    - Complex queries: loop 2-4 times
    
    Uses learned halting probability per position.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_loops: int = 4,
    ):
        super().__init__()
        self.max_loops = max_loops
        
        # Halting gate: predicts probability of stopping
        self.halt_gate = nn.Linear(hidden_size, 1, bias=False)
        
        # Cumulative halting state
        self.register_buffer('halting_state', torch.zeros(1))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        loop_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Decide whether to halt or continue looping.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            loop_idx: Current loop iteration
            
        Returns:
            hidden_states: Possibly scaled by halting probability
            halting_prob: [batch, seq_len, 1] - probability of halting
            should_halt: True if all positions have halted
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute halting probability
        halt_logit = self.halt_gate(hidden_states)  # [batch, seq, 1]
        halt_prob = torch.sigmoid(halt_logit)  # [batch, seq, 1]
        
        # Accumulate halting probability
        if loop_idx == 0:
            self.halting_state = torch.zeros(batch_size, seq_len, 1, device=hidden_states.device)
        
        self.halting_state = self.halting_state + halt_prob
        
        # Determine if we should halt
        # Halt if cumulative probability >= 1.0 or we've reached max loops
        should_halt = (self.halting_state >= 1.0).all() or (loop_idx >= self.max_loops - 1)
        
        # Scale hidden states by halting probability
        # This is the "weighted sum" of computation across loops
        if loop_idx == 0:
            # First loop: weight = halt_prob
            weighted_states = hidden_states * halt_prob
        else:
            # Subsequent loops: weight = halt_prob * (1 - cumulative_prev)
            remaining = 1.0 - self.halting_state + halt_prob
            weighted_states = hidden_states * halt_prob * remaining
        
        return weighted_states, halt_prob, should_halt
    
    def reset(self):
        """Reset halting state for new forward pass."""
        self.halting_state = torch.zeros(1)


class RecurrentDecoderLayer(nn.Module):
    """
    Single decoder layer for use in recurrent block.
    
    Same weights, different loop iteration.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        dropout_rate: float = 0.1,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        
        # Self-attention
        from liquid_foundation_model.model.blocks.decoder_block import DecoderSelfAttention
        self.self_attn = DecoderSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        
        # Cross-attention (for encoder-decoder)
        from liquid_foundation_model.model.layers.cross_attention import CrossAttention
        self.cross_attn = CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_rate=dropout_rate,
        )
        
        # FFN (simple, since we have MoE elsewhere)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
        )
        
        # Norms
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        
        # Gated residuals
        self.gate1 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate2 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate3 = nn.Parameter(torch.zeros(1, 1, hidden_size))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + torch.sigmoid(self.gate1) * hidden_states
        
        # Cross-attention
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states, _ = self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            cross_attention_mask,
        )
        hidden_states = residual + torch.sigmoid(self.gate2) * hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + torch.sigmoid(self.gate3) * hidden_states
        
        return hidden_states


class RecurrentDecoder(nn.Module):
    """
    Recurrent decoder with ACT halting.
    
    Architecture:
    - Initial decoder layer (run once)
    - Recurrent block (looped 2-4x with same weights)
    - Final decoder layer (run once)
    
    Key features:
    1. Weight sharing: Same weights reused across loops
    2. Loop-index embedding: Different computation per loop
    3. ACT halting: Early exit for simple queries
    4. Input injection: Encoder hidden states injected at each loop
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_key_value_heads: int = 4,
        max_loops: int = 4,
        dropout_rate: float = 0.1,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_loops = max_loops
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Initial decoder layer (run once)
        self.initial_layer = RecurrentDecoderLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        
        # Recurrent decoder layer (same weights, looped)
        self.recurrent_layer = RecurrentDecoderLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        
        # Loop index embedding
        self.loop_embedding = LoopIndexEmbedding(
            hidden_size=hidden_size,
            max_loops=max_loops * 2,  # Extra capacity
        )
        
        # ACT halting
        self.act_halting = ACTHalting(
            hidden_size=hidden_size,
            max_loops=max_loops,
        )
        
        # Input injection gate
        self.input_gate = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # Final layer
        self.final_layer = RecurrentDecoderLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        
        # Final norm
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        
        # Language model head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        max_loops: Optional[int] = None,
    ) -> dict:
        """
        Forward pass with recurrent decoding.
        
        Args:
            input_ids: [batch, dec_seq_len]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            attention_mask: Optional causal mask
            cross_attention_mask: Optional cross-attention mask
            labels: Optional labels for training
            output_hidden_states: Whether to return hidden states
            max_loops: Override max loops (for inference)
            
        Returns:
            dict with logits, loss, hidden_states, loop_info
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Initial layer
        hidden_states = self.initial_layer(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            cross_attention_mask,
        )
        
        # Reset ACT state
        self.act_halting.reset()
        
        # Recurrent loop
        all_hidden_states = [hidden_states] if output_hidden_states else []
        loop_probs = []
        num_loops_used = 0
        
        max_loops = max_loops or self.max_loops
        
        for loop_idx in range(max_loops):
            # Add loop index embedding
            hidden_states_with_loop = self.loop_embedding(hidden_states, loop_idx)
            
            # Input injection: concat with encoder hidden states
            # This keeps original input signal alive across loops
            encoder_pooled = encoder_hidden_states.mean(dim=1, keepdim=True)
            encoder_pooled = encoder_pooled.expand(-1, seq_len, -1)
            input_injected = torch.cat([hidden_states_with_loop, encoder_pooled], dim=-1)
            hidden_states_injected = self.input_gate(input_injected)
            
            # Recurrent layer (same weights!)
            hidden_states_new = self.recurrent_layer(
                hidden_states_injected,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
            )
            
            # ACT halting decision
            weighted_states, halt_prob, should_halt = self.act_halting(
                hidden_states_new,
                loop_idx,
            )
            
            # Update hidden states
            hidden_states = hidden_states + weighted_states
            loop_probs.append(halt_prob)
            num_loops_used += 1
            
            if output_hidden_states:
                all_hidden_states.append(hidden_states.clone())
            
            if should_halt:
                break
        
        # Final layer
        hidden_states = self.final_layer(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            cross_attention_mask,
        )
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
            )
        
        result = {
            "logits": logits,
            "loss": loss,
            "loop_probs": loop_probs,
            "num_loops_used": num_loops_used,
        }
        
        if output_hidden_states:
            result["hidden_states"] = all_hidden_states
        
        return result
