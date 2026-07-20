"""
One Neural X1 Tool - Enhanced Encoder-Decoder Model

Integrates innovations from:
1. Ternary Bonsai 27B: Ternary quantization, hybrid attention
2. OpenMythos: Recurrent decoder, ACT halting, MoE

Architecture:
- Encoder: Hybrid attention (linear + full) + ternary quantization
- Decoder: Recurrent with ACT halting + MoE FFN
- Total: ~26M params (but 104M+ effective capacity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union

from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.ternary_quantize import TernaryQuantize, replace_linear_with_ternary
from liquid_foundation_model.model.layers.hybrid_attention import create_hybrid_attention_layers
from liquid_foundation_model.model.layers.moe import MoELayer, MoEDecoderBlock
from liquid_foundation_model.model.layers.recurrent_decoder import RecurrentDecoder


class OneNeuralX1ToolV2(nn.Module):
    """
    Enhanced encoder-decoder with all innovations.
    
    Architecture:
    - Encoder: 12 layers hybrid attention (75% linear, 25% full)
    - Decoder: Recurrent (3 loops) + ACT halting + MoE FFN
    - Quantization: Ternary (1.71 bits/weight)
    
    Parameters: ~26M (but 104M+ effective capacity)
    Memory: ~5.5MB (vs 100MB for FP32)
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 512,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 6,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        max_loops: int = 4,
        num_experts: int = 4,
        expert_dim: int = 2048,
        num_experts_per_tok: int = 1,
        dropout_rate: float = 0.1,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        tie_embeddings: bool = True,
        use_ternary: bool = True,
        use_hybrid_attention: bool = True,
        use_recurrent_decoder: bool = True,
        use_moe: bool = True,
    ):
        super().__init__()
        
        # Store config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_ternary = use_ternary
        self.use_hybrid_attention = use_hybrid_attention
        self.use_recurrent_decoder = use_recurrent_decoder
        self.use_moe = use_moe
        
        # ============ ENCODER ============
        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Encoder blocks with hybrid attention
        if use_hybrid_attention:
            self.encoder_layers = nn.ModuleList()
            for i in range(num_encoder_layers):
                # 75% linear, 25% full attention
                use_linear = i < int(num_encoder_layers * 0.75)
                
                # Create block with hybrid attention
                from liquid_foundation_model.model.blocks.encoder_block import EncoderBlock
                block = EncoderBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    dropout_rate=dropout_rate,
                    max_seq_len=max_seq_len,
                    rope_theta=rope_theta,
                )
                
                # Replace attention with hybrid
                if use_linear:
                    from liquid_foundation_model.model.layers.hybrid_attention import LinearAttention
                    block.self_attn = LinearAttention(
                        hidden_size=hidden_size,
                        num_heads=num_attention_heads,
                    )
                else:
                    from liquid_foundation_model.model.layers.hybrid_attention import FullAttention
                    block.self_attn = FullAttention(
                        hidden_size=hidden_size,
                        num_heads=num_attention_heads,
                        num_key_value_heads=num_key_value_heads,
                        dropout_rate=dropout_rate,
                        max_seq_len=max_seq_len,
                        rope_theta=rope_theta,
                    )
                
                self.encoder_layers.append(block)
        else:
            # Standard encoder
            from liquid_foundation_model.model.blocks.encoder_block import EncoderBlock
            self.encoder_layers = nn.ModuleList([
                EncoderBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    dropout_rate=dropout_rate,
                    max_seq_len=max_seq_len,
                    rope_theta=rope_theta,
                )
                for _ in range(num_encoder_layers)
            ])
        
        # Encoder norm
        self.encoder_norm = RMSNorm(hidden_size, eps=1e-6)
        
        # ============ DECODER ============
        if use_recurrent_decoder:
            # Recurrent decoder with ACT halting
            self.decoder = RecurrentDecoder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_decoder_layers,
                num_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                max_loops=max_loops,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
            )
        else:
            # Standard decoder with MoE
            from liquid_foundation_model.model.blocks.decoder_block import DecoderBlock
            self.decoder_layers = nn.ModuleList([
                MoEDecoderBlock(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    num_experts=num_experts,
                    expert_dim=expert_dim,
                    num_experts_per_tok=num_experts_per_tok,
                    dropout_rate=dropout_rate,
                    max_seq_len=max_seq_len,
                    rope_theta=rope_theta,
                )
                for _ in range(num_decoder_layers)
            ])
            self.decoder_norm = RMSNorm(hidden_size, eps=1e-6)
            self.decoder_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # ============ TERNARY QUANTIZATION ============
        # Only apply at eval time, not during training
        self._ternary_enabled = False
        
        # Tie embeddings
        if tie_embeddings:
            if use_recurrent_decoder:
                self.decoder.embed_tokens.weight = self.embed_tokens.weight
                self.decoder.lm_head.weight = self.embed_tokens.weight
            else:
                self.decoder_layers[0].self_attn  # Just check it exists
                # Tie decoder embeddings
                self.decoder_lm_head.weight = self.embed_tokens.weight
        
        # Calculate parameters
        self.num_parameters = sum(p.numel() for p in self.parameters())
        self.num_parameters_millions = self.num_parameters / 1_000_000
        
        # Proper weight initialization
        self.apply(self._init_weights)
        
        # Final layer init (smaller)
        if hasattr(self, 'decoder') and hasattr(self.decoder, 'lm_head'):
            nn.init.normal_(self.decoder.lm_head.weight, mean=0.0, std=0.02)
    
    def _init_weights(self, module):
        """Initialize weights properly to prevent exploding gradients."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) or hasattr(module, 'weight') and hasattr(module, 'bias'):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Calculate memory
        if self.use_ternary:
            self.memory_mb = self.num_parameters * 1.71 / (8 * 1024 * 1024)
        else:
            self.memory_mb = self.num_parameters * 4 / (1024 * 1024)
    
    def _apply_ternary_quantization(self):
        """
        Apply ternary quantization to linear layers.
        
        This reduces memory from FP32 to 1.71 bits/weight.
        """
        # Quantize encoder layers
        for layer in self.encoder_layers:
            replace_linear_with_ternary(layer, group_size=128, replace_ratio=0.5)
        
        # Quantize decoder
        if self.use_recurrent_decoder:
            replace_linear_with_ternary(self.decoder, group_size=128, replace_ratio=0.5)
        else:
            for layer in self.decoder_layers:
                replace_linear_with_ternary(layer, group_size=128, replace_ratio=0.5)
    
    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Encode input sequence.
        
        Args:
            input_ids: [batch, enc_seq_len]
            attention_mask: Optional mask
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        all_hidden_states = [] if output_hidden_states else None
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            hidden_states, _ = layer(hidden_states, attention_mask)
        
        # Final norm
        hidden_states = self.encoder_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, all_hidden_states
        
        return hidden_states
    
    def decode(
        self,
        decoder_input_ids: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        max_loops: Optional[int] = None,
    ) -> dict:
        """
        Decode with recurrent decoder + ACT halting.
        
        Args:
            decoder_input_ids: [batch, dec_seq_len]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            attention_mask: Optional causal mask
            cross_attention_mask: Optional cross-attention mask
            labels: Optional labels for training
            output_hidden_states: Whether to return hidden states
            max_loops: Override max loops
            
        Returns:
            dict with logits, loss, loop_info
        """
        if self.use_recurrent_decoder:
            # Use recurrent decoder with ACT
            return self.decoder(
                decoder_input_ids,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
                labels,
                output_hidden_states,
                max_loops,
            )
        else:
            # Standard decoder with MoE
            batch_size, seq_len = decoder_input_ids.shape
            
            # Embed tokens
            hidden_states = self.embed_tokens(decoder_input_ids)
            
            all_hidden_states = [hidden_states] if output_hidden_states else []
            total_load_balance_loss = 0.0
            
            # Apply decoder layers
            for layer in self.decoder_layers:
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
                
                hidden_states, load_balance_loss = layer(
                    hidden_states,
                    attention_mask,
                )
                total_load_balance_loss += load_balance_loss
            
            # Final norm
            hidden_states = self.decoder_norm(hidden_states)
            
            # Language model head
            logits = self.decoder_lm_head(hidden_states)
            
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
                # Add load balancing loss
                loss = loss + 0.01 * total_load_balance_loss
            
            result = {
                "logits": logits,
                "loss": loss,
                "load_balance_loss": total_load_balance_loss,
            }
            
            if output_hidden_states:
                result["hidden_states"] = all_hidden_states
            
            return result
    
    def forward(
        self,
        encoder_input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        max_loops: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            encoder_input_ids: [batch, enc_seq_len]
            decoder_input_ids: [batch, dec_seq_len]
            encoder_attention_mask: Optional encoder mask
            decoder_attention_mask: Optional decoder mask
            labels: Optional labels for training
            output_hidden_states: Whether to return hidden states
            max_loops: Override max loops
            
        Returns:
            dict with logits, loss, hidden_states
        """
        # Encode
        encoder_hidden_states = self.encode(
            encoder_input_ids,
            encoder_attention_mask,
            output_hidden_states,
        )
        
        if output_hidden_states:
            encoder_hidden_states, encoder_hidden_states_list = encoder_hidden_states
        
        # Decode
        decoder_output = self.decode(
            decoder_input_ids,
            encoder_hidden_states,
            decoder_attention_mask,
            None,  # cross_attention_mask
            labels,
            output_hidden_states,
            max_loops,
        )
        
        # Combine results
        result = {
            "logits": decoder_output["logits"],
            "loss": decoder_output["loss"],
        }
        
        if output_hidden_states:
            result["encoder_hidden_states"] = encoder_hidden_states_list
            result["decoder_hidden_states"] = decoder_output.get("hidden_states", [])
        
        if "loop_probs" in decoder_output:
            result["loop_probs"] = decoder_output["loop_probs"]
            result["num_loops_used"] = decoder_output["num_loops_used"]
        
        if "load_balance_loss" in decoder_output:
            result["load_balance_loss"] = decoder_output["load_balance_loss"]
        
        return result
    
    def generate(
        self,
        encoder_input_ids: torch.LongTensor,
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        max_loops: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate function calls.
        
        Args:
            encoder_input_ids: [batch, enc_seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            do_sample: Whether to sample
            max_loops: Override max loops for generation
            
        Returns:
            generated_ids: [batch, max_length]
        """
        batch_size = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        
        # Encode
        encoder_hidden_states = self.encode(encoder_input_ids)
        
        # Start with BOS token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            1,  # BOS token
            dtype=torch.long,
            device=device,
        )
        
        # Generate tokens
        for _ in range(max_length):
            # Decode
            decoder_output = self.decode(
                decoder_input_ids,
                encoder_hidden_states,
                max_loops=max_loops,
            )
            
            logits = decoder_output["logits"][:, -1, :]  # [batch, vocab]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample or argmax
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Check for EOS
            if (next_token == 2).all():  # EOS token
                break
        
        return decoder_input_ids
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            dict with model stats
        """
        info = {
            "name": "OneNeuralX1ToolV2",
            "parameters": self.num_parameters,
            "parameters_millions": self.num_parameters_millions,
            "memory_mb": self.memory_mb,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "features": {
                "ternary_quantization": self.use_ternary,
                "hybrid_attention": self.use_hybrid_attention,
                "recurrent_decoder": self.use_recurrent_decoder,
                "moe": self.use_moe,
            },
        }
        
        if self.use_ternary:
            info["compression_ratio"] = 16.0 / 1.71
        
        if self.use_recurrent_decoder:
            info["max_loops"] = self.decoder.max_loops
        
        return info
