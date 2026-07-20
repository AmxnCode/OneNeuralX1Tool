"""
NeedleBonsai - 27M Parameter Hybrid Attention Model

Inspired by:
- Needle (Cactus-Compute): Pure attention encoder-decoder, ZCRMSNorm, gated residuals, no FFN
- Bonsai-27B (Prism-ML): Hybrid attention (75% linear / 25% full)

Architecture:
- Encoder: 12 layers, hybrid attention (9 linear + 3 full), ZCRMSNorm, gated residuals
- Decoder: 8 layers, self-attn + cross-attention, ZCRMSNorm, gated residuals
- No FFN/MLP anywhere - pure attention network
- Shared embeddings between encoder and decoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, '/tmp/opencode/OneNeuralX1Tool')

from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.rope import RotaryPositionEmbedding
from liquid_foundation_model.model.layers.hybrid_attention import LinearAttention, FullAttention


class ZCRMSNorm(nn.Module):
    """Zero-Centered RMSNorm from Needle."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * (1.0 + self.weight)


class GatedResidual(nn.Module):
    """Gated residual connection from Needle."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, residual], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        return gate * x + (1 - gate) * residual


class EncoderLayer(nn.Module):
    """Encoder layer: ZCRMSNorm -> Self-Attention (hybrid) -> Gated Residual."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_linear: bool,
        dropout_rate: float = 0.1,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.norm = ZCRMSNorm(hidden_size)
        if use_linear:
            self.self_attn = LinearAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout_rate=dropout_rate,
            )
        else:
            self.self_attn = FullAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
            )
        self.residual = GatedResidual(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.self_attn(x, mask)
        return self.residual(x, residual)


class DecoderLayer(nn.Module):
    """Decoder layer: ZCRMSNorm -> Masked Self-Attn -> ZCRMSNorm -> Cross-Attn -> Gated Residuals."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        dropout_rate: float = 0.1,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.self_attn_norm = ZCRMSNorm(hidden_size)
        self.self_attn = FullAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        self.self_attn_residual = GatedResidual(hidden_size)

        self.cross_attn_norm = ZCRMSNorm(hidden_size)
        self.cross_attn_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.cross_attn_k = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.cross_attn_v = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.cross_attn_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.cross_attn_residual = GatedResidual(hidden_size)

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_key_value_heads

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Masked self-attention
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, self_attn_mask)
        x = self.self_attn_residual(x, residual)

        # Cross-attention
        residual = x
        x_normed = self.cross_attn_norm(x)
        B, T, _ = x_normed.shape
        S = encoder_hidden.shape[1]

        q = self.cross_attn_q(x_normed).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_attn_k(encoder_hidden).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.cross_attn_v(encoder_hidden).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.num_key_value_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.cross_attn_o(out)

        return self.cross_attn_residual(out, residual)


class NeedleBonsai(nn.Module):
    """
    NeedleBonsai: 27M parameter encoder-decoder with pure attention.

    Architecture:
    - 12 encoder layers (9 linear + 3 full attention)
    - 8 decoder layers (self-attn + cross-attn)
    - No FFN/MLP anywhere
    - ZCRMSNorm, gated residuals, RoPE, GQA
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 384,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 8,
        num_heads: int = 8,
        num_key_value_heads: int = 4,
        max_seq_len: int = 1024,
        dropout_rate: float = 0.1,
        rope_theta: float = 10000.0,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        head_dim = hidden_size // num_heads

        # Shared embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Encoder: hybrid attention (75% linear, 25% full)
        self.encoder_layers = nn.ModuleList()
        for i in range(num_encoder_layers):
            use_linear = i < int(num_encoder_layers * 0.75)
            self.encoder_layers.append(EncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                use_linear=use_linear,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
            ))
        self.encoder_norm = ZCRMSNorm(hidden_size)

        # Decoder: self-attn + cross-attention (all full attention)
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.decoder_layers.append(DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
            ))
        self.decoder_norm = ZCRMSNorm(hidden_size)

        # LM head (tied with embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def encode(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.encoder_norm(x)

    def decode(
        self,
        decoder_ids: torch.Tensor,
        encoder_hidden: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(decoder_ids)
        for layer in self.decoder_layers:
            x = layer(x, encoder_hidden, self_attn_mask)
        return self.decoder_norm(x)

    def forward(
        self,
        encoder_input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = encoder_input_ids.shape[0]

        # Encode
        encoder_hidden = self.encode(encoder_input_ids)

        # Causal mask for decoder
        dec_len = decoder_input_ids.shape[1]
        causal_mask = self._make_causal_mask(dec_len, decoder_input_ids.device)

        # Decode
        dec_hidden = self.decode(decoder_input_ids, encoder_hidden, causal_mask)

        # LM head
        logits = self.lm_head(dec_hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids: torch.LongTensor,
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.LongTensor:
        B = encoder_input_ids.shape[0]
        device = encoder_input_ids.device

        encoder_hidden = self.encode(encoder_input_ids)

        decoder_ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_len = decoder_ids.shape[1]
            causal_mask = self._make_causal_mask(dec_len, device)
            dec_hidden = self.decode(decoder_ids, encoder_hidden, causal_mask)
            logits = self.lm_head(dec_hidden[:, -1, :]) / temperature

            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            decoder_ids = torch.cat([decoder_ids, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return decoder_ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_info(self) -> dict:
        params = self.num_parameters()
        return {
            "name": "NeedleBonsai",
            "parameters": params,
            "parameters_millions": params / 1e6,
            "memory_mb": params * 4 / (1024 * 1024),
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "architecture": "Encoder-Decoder, Pure Attention (No FFN)",
            "encoder_layers": len(self.encoder_layers),
            "decoder_layers": len(self.decoder_layers),
            "hybrid_attention": "75% linear + 25% full",
            "norm": "ZCRMSNorm",
            "residuals": "Gated",
        }


def create_model():
    """Create ~27M parameter NeedleBonsai model."""
    model = NeedleBonsai(
        vocab_size=8192,
        hidden_size=384,
        num_encoder_layers=12,
        num_decoder_layers=8,
        num_heads=6,
        num_key_value_heads=3,
        max_seq_len=1024,
        dropout_rate=0.1,
        tie_embeddings=True,
    )
    return model


if __name__ == "__main__":
    model = create_model()
    info = model.get_info()
    print("=" * 60)
    print("NeedleBonsai Model")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Test forward pass
    enc = torch.randint(0, 8192, (2, 128))
    dec = torch.randint(0, 8192, (2, 64))
    labels = torch.randint(0, 8192, (2, 64))
    out = model(enc, dec, labels=labels)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
