import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union

from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.blocks.encoder_block import EncoderBlock
from liquid_foundation_model.model.blocks.decoder_block import DecoderBlock


class OneNeuralX1ToolEncoder(nn.Module):
    """
    Encoder for One Neural X1 Tool.
    
    Architecture (from Needle):
    - 12 layers
    - GQA (8H/4KV)
    - RoPE
    - Gated residuals
    - No FFN (pure attention)
    
    Parameters: ~18M
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 512,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        dropout_rate: float = 0.1,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
            )
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(hidden_size, eps=1e-6)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
            all_hidden_states: Optional list of hidden states
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        all_hidden_states = [] if output_hidden_states else None
        
        # Apply encoder layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            hidden_states, _ = layer(hidden_states, attention_mask)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, all_hidden_states
        
        return hidden_states


class OneNeuralX1ToolDecoder(nn.Module):
    """
    Decoder for One Neural X1 Tool.
    
    Architecture (from Needle):
    - 8 layers
    - Self-attention (causal) + Cross-attention
    - RoPE
    - Gated residuals
    - No FFN (pure attention)
    
    Parameters: ~9M
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 512,
        num_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        dropout_rate: float = 0.1,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        
        # Token embeddings (tied with encoder)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
            )
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        
        # Language model head (tied with embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of decoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            encoder_hidden_states: Encoder output [batch_size, enc_seq_len, hidden_size]
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            all_hidden_states: Optional list of hidden states
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        all_hidden_states = [] if output_hidden_states else None
        
        # Apply decoder layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            hidden_states, _ = layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
            )
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return logits, all_hidden_states
        
        return logits


class OneNeuralX1Tool(nn.Module):
    """
    One Neural X1 Tool - Encoder-Decoder model for function calling.
    
    Architecture:
    - Encoder: 12 layers, GQA (8H/4KV), RoPE, gated residuals
    - Decoder: 8 layers, self-attn + cross-attn, gated residuals
    - d_model: 512
    - Vocab: 8,192 (SentencePiece BPE)
    - No FFN (pure attention)
    - Total: ~27M parameters
    
    Features:
    - Function calling
    - Tool use
    - Structured output (JSON)
    - Encoder-decoder for better structured generation
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 512,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        dropout_rate: float = 0.1,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        
        # Store config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder = OneNeuralX1ToolEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        
        # Decoder
        self.decoder = OneNeuralX1ToolDecoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_decoder_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        
        # Tie embeddings
        if tie_embeddings:
            self.decoder.embed_tokens.weight = self.encoder.embed_tokens.weight
            self.decoder.lm_head.weight = self.encoder.embed_tokens.weight
        
        # Calculate parameters
        self.num_parameters = sum(p.numel() for p in self.parameters())
        self.num_parameters_millions = self.num_parameters / 1_000_000
    
    def forward(
        self,
        encoder_input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of encoder-decoder model.
        
        Args:
            encoder_input_ids: Input tokens for encoder [batch_size, enc_seq_len]
            decoder_input_ids: Input tokens for decoder [batch_size, dec_seq_len]
            encoder_attention_mask: Optional mask for encoder
            decoder_attention_mask: Optional mask for decoder
            labels: Optional labels for training [batch_size, dec_seq_len]
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary with logits, loss, and optional hidden states
        """
        # Encode
        encoder_hidden_states = self.encoder(
            encoder_input_ids,
            encoder_attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        if output_hidden_states:
            encoder_hidden_states, encoder_hidden_states_list = encoder_hidden_states
        
        # Decode
        decoder_output = self.decoder(
            decoder_input_ids,
            encoder_hidden_states,
            decoder_attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        if output_hidden_states:
            logits, decoder_hidden_states_list = decoder_output
        else:
            logits = decoder_output
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )
        
        result = {
            "logits": logits,
            "loss": loss,
        }
        
        if output_hidden_states:
            result["encoder_hidden_states"] = encoder_hidden_states_list
            result["decoder_hidden_states"] = decoder_hidden_states_list
        
        return result
    
    def generate(
        self,
        encoder_input_ids: torch.LongTensor,
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        early_stopping: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.LongTensor:
        """
        Generate text using the encoder-decoder model.
        
        Args:
            encoder_input_ids: Input tokens for encoder [batch_size, enc_seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample or use greedy decoding
            num_beams: Number of beams for beam search
            early_stopping: Stop when all beams reach EOS
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            generated_ids: Generated token IDs [batch_size, seq_len]
        """
        batch_size = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        
        # Encode input
        encoder_hidden_states = self.encoder(encoder_input_ids)
        
        # Start with BOS token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # Generate tokens
        for _ in range(max_length):
            # Forward pass
            logits = self.decoder(
                decoder_input_ids,
                encoder_hidden_states,
            )
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < top_k_values[:, -1:]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            
            # Check for EOS
            if (next_token == eos_token_id).all():
                break
        
        return decoder_input_ids
    
    def get_encoder_output(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Get encoder output for given input.
        
        Args:
            input_ids: Input tokens [batch_size, seq_len]
            
        Returns:
            encoder_hidden_states: [batch_size, seq_len, hidden_size]
        """
        return self.encoder(input_ids)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "total": encoder_params + decoder_params,
            "encoder_millions": encoder_params / 1_000_000,
            "decoder_millions": decoder_params / 1_000_000,
            "total_millions": (encoder_params + decoder_params) / 1_000_000,
        }
