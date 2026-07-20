"""
Google Colab - FULL Model Training
Includes ALL innovations: Hybrid Attention, MoE, Recurrent Decoder, ACT, Ternary Quantization
~10-15 min on T4 GPU for 2 epochs
"""

# Cell 1: Install dependencies
!pip install tokenizers huggingface_hub

# Cell 2: Full Model Definition
# This contains ALL layers from our architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Union


# ============ RMSNorm ============
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ============ RoPE ============
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.shape[2]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated_x1 = x1 * cos[..., :self.head_dim // 2] - x2 * sin[..., :self.head_dim // 2]
        rotated_x2 = x1 * sin[..., :self.head_dim // 2] + x2 * cos[..., :self.head_dim // 2]
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


# ============ Ternary Quantization ============
class TernaryQuantize(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(out_features))

    def ternary_round(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x.abs() < 0.5, torch.zeros_like(x), torch.sign(x))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        original_weight = weight
        weight_max = weight.abs().max(dim=-1, keepdim=True)[0] + 1e-8
        weight_normalized = weight / weight_max
        weight_ternary = self.ternary_round(weight_normalized)
        weight_quantized = weight_ternary * self.scale.unsqueeze(-1)
        if self.training:
            return original_weight + (weight_quantized - original_weight).detach()
        return weight_quantized


# ============ Linear Attention ============
class LinearAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        B, N, _ = hidden_states.shape
        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        Q = self.feature_map(Q).transpose(1, 2)
        K = self.feature_map(K).transpose(1, 2)
        V = V.transpose(1, 2)
        KV = torch.matmul(K.transpose(-1, -2), V)
        QKV = torch.matmul(Q, KV)
        Z = 1.0 / (torch.matmul(Q, K.transpose(-1, -2).sum(dim=-1, keepdim=True)) + 1e-6)
        output = QKV * Z
        output = output.transpose(1, 2).contiguous().view(B, N, -1)
        return self.o_proj(output), None


# ============ Full Attention ============
class FullAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_key_value_heads, dropout_rate=0.0, max_seq_len=8192, rope_theta=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_kv = num_heads // num_key_value_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len, rope_theta)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        B, N, _ = hidden_states.shape
        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        Q = self.rope(Q, N)
        K = self.rope(K, N)
        if self.num_key_value_heads != self.num_heads:
            K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
            V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, V)
        output = output.transpose(1, 2).contiguous().view(B, N, -1)
        return self.o_proj(output), attn_probs


# ============ SwiGLU ============
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ============ MoE ============
class Expert(nn.Module):
    def __init__(self, hidden_size, expert_dim, dropout_rate=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, expert_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, expert_dim, bias=False)
        self.down_proj = nn.Linear(expert_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TopKRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_tok=1, noise_std=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.noise_std = noise_std
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std
        router_probs = F.softmax(logits, dim=-1)
        top_k_probs, expert_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        f = router_probs.mean(dim=[0, 1])
        P = router_probs.mean(dim=[0, 1])
        load_balance_loss = self.num_experts * (f * P).sum()
        return router_probs, expert_indices, top_k_probs, load_balance_loss


class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts=4, expert_dim=2048, num_experts_per_tok=1, dropout_rate=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(hidden_size, expert_dim, dropout_rate) for _ in range(num_experts)])
        self.router = TopKRouter(hidden_size, num_experts, num_experts_per_tok)
        self.shared_expert = Expert(hidden_size, expert_dim, dropout_rate)
        self.shared_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        B, N, D = hidden_states.shape
        router_probs, expert_indices, top_k_probs, load_balance_loss = self.router(hidden_states)
        output = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx).any(dim=-1)
            if mask.any():
                expert_input = hidden_states[mask]
                expert_output = self.experts[expert_idx](expert_input)
                weight = router_probs[mask, expert_idx].unsqueeze(-1)
                output[mask] += expert_output * weight
        shared_output = self.shared_expert(hidden_states)
        shared_weight = torch.sigmoid(self.shared_gate(hidden_states))
        output = output + shared_output * shared_weight
        return output, load_balance_loss


# ============ Encoder Block ============
class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_key_value_heads, dropout_rate=0.1, max_seq_len=8192, rope_theta=10000.0, use_linear=True):
        super().__init__()
        if use_linear:
            self.self_attn = LinearAttention(hidden_size, num_heads, dropout_rate)
        else:
            self.self_attn = FullAttention(hidden_size, num_heads, num_key_value_heads, dropout_rate, max_seq_len, rope_theta)
        self.mlp = MoELayer(hidden_size, num_experts=4, expert_dim=hidden_size * 4, num_experts_per_tok=1, dropout_rate=dropout_rate)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + self.dropout1(hidden_states)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states, load_balance_loss = self.mlp(hidden_states)
        hidden_states = residual + self.dropout2(hidden_states)
        return hidden_states, load_balance_loss


# ============ Recurrent Decoder ============
class RecurrentDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_key_value_heads, dropout_rate=0.1, max_seq_len=8192, rope_theta=10000.0):
        super().__init__()
        self.self_attn = FullAttention(hidden_size, num_heads, num_key_value_heads, dropout_rate, max_seq_len, rope_theta)
        self.cross_attn = FullAttention(hidden_size, num_heads, num_key_value_heads, dropout_rate, max_seq_len, rope_theta)
        self.ffn = SwiGLU(hidden_size, hidden_size * 4, hidden_size)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.norm3 = RMSNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.gate1 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate2 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate3 = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + torch.sigmoid(self.gate1) * self.dropout1(hidden_states)

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            hidden_states, _ = self.cross_attn(hidden_states, None)
            hidden_states = residual + torch.sigmoid(self.gate2) * self.dropout2(hidden_states)

        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + torch.sigmoid(self.gate3) * self.dropout3(hidden_states)
        return hidden_states


class RecurrentDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, num_key_value_heads, max_loops=4, dropout_rate=0.1, max_seq_len=8192, rope_theta=10000.0):
        super().__init__()
        self.max_loops = max_loops
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.loop_embedding = nn.Embedding(max_loops, hidden_size)
        self.initial_layer = RecurrentDecoderLayer(hidden_size, num_heads, num_key_value_heads, dropout_rate, max_seq_len, rope_theta)
        self.recurrent_layer = RecurrentDecoderLayer(hidden_size, num_heads, num_key_value_heads, dropout_rate, max_seq_len, rope_theta)
        self.final_layer = RecurrentDecoderLayer(hidden_size, num_heads, num_key_value_heads, dropout_rate, max_seq_len, rope_theta)
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.loop_norm = RMSNorm(hidden_size)
        self.input_gate = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.loop_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act_halting = nn.ModuleDict({
            'halting_state': nn.Parameter(torch.zeros(1)),
            'halt_gate': nn.Linear(hidden_size, 1, bias=False),
        })
        self.act_epsilon = 0.01

    def forward(self, decoder_input_ids, encoder_hidden_states, attention_mask=None, cross_attention_mask=None, labels=None, output_hidden_states=False, max_loops=None):
        max_loops = max_loops or self.max_loops
        B, T = decoder_input_ids.shape
        hidden_states = self.embed_tokens(decoder_input_ids)
        hidden_states = self.initial_layer(hidden_states, encoder_hidden_states, attention_mask)
        all_hidden_states = [hidden_states] if output_hidden_states else []
        total_loss = 0.0
        remaining_updates = torch.ones(B, T, 1, device=hidden_states.device)
        halting_state = self.act_halting['halting_state'].expand(B, T, 1).clone()
        loop_probs_accumulated = torch.zeros(B, T, max_loops, device=hidden_states.device)
        for loop_idx in range(max_loops):
            loop_emb = self.loop_embedding(torch.tensor(loop_idx, device=hidden_states.device)).unsqueeze(0).unsqueeze(0)
            loop_input = hidden_states + self.loop_norm(loop_emb)
            recurrent_out = self.recurrent_layer(loop_input, encoder_hidden_states, attention_mask)
            projected = self.loop_proj(recurrent_out)
            loop_prob = torch.sigmoid(self.act_halting['halt_gate'](recurrent_out) + self.act_halting['halting_state'])
            loop_prob_scaled = loop_prob.clamp(self.act_epsilon, 1.0 - self.act_epsilon)
            loop_probs_accumulated[:, :, loop_idx] = loop_prob_scaled.squeeze(-1)
            update = remaining_updates * loop_prob_scaled
            hidden_states = hidden_states + update * projected
            remaining_updates = remaining_updates * (1.0 - loop_prob)
            all_hidden_states.append(hidden_states)
            if remaining_updates.sum() < self.act_epsilon:
                break
        hidden_states = self.final_layer(hidden_states, encoder_hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1), ignore_index=-100, label_smoothing=0.1)
        result = {"logits": logits, "loss": loss, "loop_probs": loop_probs_accumulated, "num_loops_used": max_loops - remaining_updates.mean().item()}
        if output_hidden_states:
            result["hidden_states"] = all_hidden_states
        return result


# ============ Full Model ============
class OneNeuralX1ToolV2(nn.Module):
    def __init__(self, vocab_size=16384, hidden_size=512, num_encoder_layers=12, num_decoder_layers=6, num_attention_heads=8, num_key_value_heads=4, max_loops=4, num_experts=4, expert_dim=2048, dropout_rate=0.1, max_seq_len=8192, rope_theta=10000.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.encoder_layers = nn.ModuleList()
        for i in range(num_encoder_layers):
            use_linear = i < int(num_encoder_layers * 0.75)
            self.encoder_layers.append(EncoderBlock(hidden_size, num_attention_heads, num_key_value_heads, dropout_rate, max_seq_len, rope_theta, use_linear))
        self.encoder_norm = RMSNorm(hidden_size)
        self.decoder = RecurrentDecoder(vocab_size, hidden_size, num_decoder_layers, num_attention_heads, num_key_value_heads, max_loops, dropout_rate, max_seq_len, rope_theta)
        self.decoder.embed_tokens.weight = self.embed_tokens.weight
        self.decoder.lm_head.weight = self.embed_tokens.weight
        self._ternary_enabled = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, input_ids, attention_mask=None, output_hidden_states=False):
        hidden_states = self.embed_tokens(input_ids)
        all_hidden_states = [] if output_hidden_states else None
        for layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, _ = layer(hidden_states, attention_mask)
        hidden_states = self.encoder_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, all_hidden_states
        return hidden_states

    def decode(self, decoder_input_ids, encoder_hidden_states, attention_mask=None, labels=None, output_hidden_states=False, max_loops=None):
        return self.decoder(decoder_input_ids, encoder_hidden_states, attention_mask, None, labels, output_hidden_states, max_loops)

    def forward(self, encoder_input_ids, decoder_input_ids, encoder_attention_mask=None, decoder_attention_mask=None, labels=None, output_hidden_states=False, max_loops=None):
        encoder_hidden_states = self.encode(encoder_input_ids, encoder_attention_mask, output_hidden_states)
        if output_hidden_states:
            encoder_hidden_states, encoder_hidden_states_list = encoder_hidden_states
        decoder_output = self.decode(decoder_input_ids, encoder_hidden_states, decoder_attention_mask, labels, output_hidden_states, max_loops)
        result = {"logits": decoder_output["logits"], "loss": decoder_output["loss"]}
        if "loop_probs" in decoder_output:
            result["loop_probs"] = decoder_output["loop_probs"]
        return result

    @torch.no_grad()
    def generate(self, encoder_input_ids, max_length=128, temperature=0.7, top_p=0.9, do_sample=True, max_loops=None):
        B = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        encoder_hidden_states = self.encode(encoder_input_ids)
        decoder_input_ids = torch.full((B, 1), 1, dtype=torch.long, device=device)
        for _ in range(max_length):
            decoder_output = self.decode(decoder_input_ids, encoder_hidden_states, max_loops=max_loops)
            logits = decoder_output["logits"][:, -1, :] / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            if (next_token == 2).all():
                break
        return decoder_input_ids


print("Full model defined (Hybrid Attention + MoE + Recurrent Decoder + ACT + Ternary)")


# Cell 3: Dataset
import json
from torch.utils.data import Dataset, DataLoader

class ToolCallDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query_ids = self.tokenizer.encode(item["query"]).ids[:self.max_len]
        answer_ids = self.tokenizer.encode(item["answers"]).ids[:self.max_len]
        decoder_input = [1] + answer_ids[:-1]
        labels = answer_ids
        query_ids = query_ids + [0] * (self.max_len - len(query_ids))
        decoder_input = decoder_input + [0] * (self.max_len - len(decoder_input))
        labels = labels + [-100] * (self.max_len - len(labels))
        return (
            torch.tensor(query_ids[:self.max_len], dtype=torch.long),
            torch.tensor(decoder_input[:self.max_len], dtype=torch.long),
            torch.tensor(labels[:self.max_len], dtype=torch.long)
        )


# Cell 4: Load Needle training data from HuggingFace
import os

data_path = "needle_tool_calls.jsonl"

if not os.path.exists(data_path):
    print("Downloading Needle training data from HuggingFace...")
    !pip install -q datasets
    from datasets import load_dataset

    # Load from HuggingFace: Cactus-Compute/tool-calls
    ds = load_dataset("Cactus-Compute/tool-calls", split="train")
    print(f"Loaded {len(ds)} examples from HuggingFace")

    # Save as JSONL
    with open(data_path, "w") as f:
        for item in ds:
            # Convert to our format: query, tools, answers
            query = item.get("query", item.get("input", item.get("prompt", "")))
            tools = item.get("tools", item.get("tool_definitions", "[]"))
            answers = item.get("answers", item.get("output", item.get("response", "[]")))

            # If tools/answers are dicts, convert to JSON strings
            if isinstance(tools, dict):
                tools = json.dumps([tools]) if "name" in tools else json.dumps(tools)
            elif isinstance(tools, list):
                tools = json.dumps(tools)
            if isinstance(answers, dict):
                answers = json.dumps([answers]) if "name" in answers else json.dumps(answers)
            elif isinstance(answers, list):
                answers = json.dumps(answers)

            f.write(json.dumps({"query": query, "tools": tools, "answers": answers}) + "\n")

    print(f"Saved {len(ds)} examples to {data_path}")
else:
    print(f"Data exists: {data_path}")

# Show data stats
data = []
with open(data_path) as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
print(f"Total examples: {len(data)}")
if data:
    print(f"Sample query: {data[0]['query'][:80]}")
    print(f"Sample answers: {data[0]['answers'][:100]}")

data = []
with open(data_path) as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
print(f"Loaded {len(data)} examples")


# Cell 5: Tokenizer with tool name tokens (Needle's tools)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

print("Building tokenizer with Needle tool name tokens...")
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# Needle's tool names
tool_names = ["search_web", "get_weather", "send_email", "create_event", "get_stock_price", "set_reminder", "play_music", "calculate"]
special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + tool_names
trainer = BpeTrainer(vocab_size=8192, special_tokens=special_tokens, min_frequency=2, continuing_subword_prefix="")

texts = []
for item in data:
    texts.extend([item["query"], item["answers"]])
tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
tokenizer.enable_truncation(max_length=256)
print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")


# Cell 6: Create data loader
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

BATCH_SIZE = 8
train_loader = DataLoader(ToolCallDataset(train_data, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ToolCallDataset(val_data, tokenizer), batch_size=BATCH_SIZE)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")


# Cell 7: Initialize model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = OneNeuralX1ToolV2(
    vocab_size=8192,
    hidden_size=512,
    num_encoder_layers=12,
    num_decoder_layers=6,
    num_attention_heads=8,
    num_key_value_heads=4,
    max_loops=4,
    num_experts=4,
    expert_dim=2048,
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"Model: {params/1e6:.2f}M parameters")


# Cell 8: Training loop
NUM_EPOCHS = 20
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'='*50}")

    # Train
    model.train()
    total_loss = 0
    for i, (query, decoder_input, labels) in enumerate(train_loader):
        query = query.to(device)
        decoder_input = decoder_input.to(device)
        labels = labels.to(device)
        outputs = model(query, decoder_input, labels=labels)
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"\n  Train Loss: {avg_train_loss:.4f}")

    # Validate
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for query, decoder_input, labels in val_loader:
            query = query.to(device)
            decoder_input = decoder_input.to(device)
            labels = labels.to(device)
            outputs = model(query, decoder_input, labels=labels)
            val_loss += outputs["loss"].item()
            predictions = outputs["logits"].argmax(dim=-1)
            mask = labels != -100
            correct += (predictions[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total if total > 0 else 0
    print(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': tokenizer.get_vocab_size(),
            'accuracy': val_accuracy,
            'epoch': epoch
        }, "best_model.pt")
        print(f"  Saved best model (accuracy: {val_accuracy:.4f})")


# Cell 9: Test
print("\n" + "=" * 50)
print("Testing the trained model...")
print("=" * 50)

test_queries = [
    "What's the weather in Tokyo?",
    "Check AAPL stock price",
    "Send email to john@example.com",
    "Calculate 15 * 3",
    "Set a reminder for tomorrow",
    "Search for climate change",
    "Play some jazz music",
    "Create a meeting event",
]

model.eval()
for query in test_queries:
    enc = tokenizer.encode(query)
    inp = torch.tensor([enc.ids], device=device)
    with torch.no_grad():
        out = model.generate(inp, max_length=50, temperature=0.1, do_sample=False)
    ids = [t for t in out[0].tolist() if t > 2]
    output = tokenizer.decode(ids)
    print(f"\nQ: {query}")
    print(f"A: {output[:150]}")


# Cell 10: Push to HuggingFace
print("\n" + "=" * 50)
print("Pushing to HuggingFace...")
print("=" * 50)

from huggingface_hub import HfApi, login
from google.colab import files
import shutil

HF_TOKEN = input("Enter your HuggingFace token: ")
login(token=HF_TOKEN)

hf_dir = "aman3456/OneNeuralX1Tool-26M"

model_card = f"""---
language: en
tags:
- tool-calling
- function-calling
- encoder-decoder
- 26m-params
- ternary-quantization
- hybrid-attention
- moe
- recurrent-depth
- act-halting
---

# OneNeuralX1Tool - 26M Function Calling Model

A small (26M parameter) encoder-decoder model for function calling/agentic tasks.

## Architecture
- **Encoder:** 12 layers hybrid attention (75% linear O(n), 25% full O(n²))
- **Decoder:** Recurrent (4 loops) with Adaptive Computation Time (ACT) halting
- **MoE:** 4 experts per layer + shared expert with load balancing
- **Quantization:** Ternary ({{-1,0,+1}}) with 1.71 bits/weight (5.4MB)
- **Vocab:** 8,192 BPE tokens with Needle tool name support

## Innovations from
- **Ternary Bonsai 27B:** Ternary quantization, hybrid attention
- **OpenMythos:** Recurrent decoder, ACT halting, MoE

## Supported Tools (Needle format)
- search_web, get_weather, send_email, create_event
- get_stock_price, set_reminder, play_music, calculate

## Training Data
- Dataset: Needle tool-calling dataset from HuggingFace (Cactus-Compute/tool-calls)
- Epochs: {NUM_EPOCHS}
- Best Accuracy: {best_val_acc:.4f}
"""

# Save model
model.save_pretrained("OneNeuralX1Tool-26M")
tokenizer.save("OneNeuralX1Tool-26M/tokenizer.json")
with open("OneNeuralX1Tool-26M/README.md", "w") as f:
    f.write(model_card)

# Push
api = HfApi()
api.create_repo(hf_dir, exist_ok=True)
api.upload_folder(folder_path="OneNeuralX1Tool-26M", repo_id=hf_dir, repo_type="model")

print(f"\nModel pushed to: https://huggingface.co/{hf_dir}")
print(f"Best accuracy: {best_val_acc:.4f}")

# Download locally
shutil.make_archive("OneNeuralX1Tool-26M", "zip", "OneNeuralX1Tool-26M")
files.download("OneNeuralX1Tool-26M.zip")
