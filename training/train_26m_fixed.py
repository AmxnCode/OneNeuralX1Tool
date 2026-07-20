"""
Fixed Training Script - With proper tokenization of tool names.
Adds tool names as special tokens so they stay as single tokens.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import random
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

import sys
sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')

from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2

# Tool names that must be single tokens
TOOL_NAMES = [
    "get_weather", "send_email", "search_flights", "create_event",
    "get_stock_price", "calculate", "set_reminder", "get_directions",
    "search_knowledge_base", "book_flight"
]


def build_tokenizer(data_path, vocab_size=16384):
    """Build BPE tokenizer with tool names as special tokens."""
    print("   Building BPE tokenizer with tool name tokens...")
    
    # Special tokens
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + TOOL_NAMES
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=1,
    )
    
    # Read all text
    texts = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["query"])
            texts.append(ex["answers"])
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Verify tool names are single tokens
    for name in TOOL_NAMES:
        ids = tokenizer.encode(name).ids
        # Remove special tokens from count
        real_ids = [i for i in ids if i >= len(special_tokens)]
        assert len(real_ids) <= 1, f"Tool name '{name}' split into {len(real_ids)} tokens: {ids}"
    
    print(f"   Vocab size: {tokenizer.get_vocab_size()}")
    print(f"   Tool names verified as single tokens")
    
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
    tokenizer.enable_truncation(max_length=256)
    
    return tokenizer


class MultiToolDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        with open(data_path) as f:
            self.examples = [json.loads(line) for line in f if line.strip()]
        
        print(f"   Loaded {len(self.examples)} examples")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        query_enc = self.tokenizer.encode(ex["query"])
        answer_enc = self.tokenizer.encode(ex["answers"])
        
        query_ids = query_enc.ids[:self.max_len] + [0] * max(0, self.max_len - len(query_enc.ids))
        answer_ids = [1] + answer_enc.ids[:self.max_len-1] + [0] * max(0, self.max_len - len(answer_enc.ids) - 1)
        labels = answer_enc.ids[:self.max_len] + [-100] * max(0, self.max_len - len(answer_enc.ids))
        
        return {
            "encoder_input_ids": torch.tensor(query_ids[:self.max_len]),
            "decoder_input_ids": torch.tensor(answer_ids[:self.max_len]),
            "labels": torch.tensor(labels[:self.max_len]),
        }


def train():
    print("=" * 60)
    print("Training 26M Model (Fixed Tokenizer)")
    print("=" * 60)
    
    config = {
        "vocab_size": 16384,
        "hidden_size": 420,
        "num_encoder_layers": 10,
        "num_decoder_layers": 5,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "max_loops": 3,
        "num_experts": 4,
        "expert_dim": 1280,
        "batch_size": 4,
        "lr": 5e-5,
        "epochs": 20,
        "max_len": 256,
        "warmup_epochs": 2,
    }
    
    device = torch.device("cpu")
    data_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/data/multi_tool_chains.jsonl"
    
    # Clean old checkpoints
    import glob, os
    for old in glob.glob("/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/x1_26m_*.pt"):
        os.remove(old)
        print(f"   Deleted: {old}")
    
    # Build tokenizer
    print("\n1. Building tokenizer...")
    tokenizer = build_tokenizer(data_path, config["vocab_size"])
    
    # Update vocab size
    config["vocab_size"] = tokenizer.get_vocab_size()
    
    # Model
    print("\n2. Creating 26M model...")
    model = OneNeuralX1ToolV2(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        max_loops=config["max_loops"],
        num_experts=config["num_experts"],
        expert_dim=config["expert_dim"],
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Dataset
    print("\n3. Loading data...")
    dataset = MultiToolDataset(data_path, tokenizer, max_len=config["max_len"])
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=0)
    
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    
    # Training
    print(f"\n4. Training for {config['epochs']} epochs...")
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
        # LR schedule
        if epoch < config["warmup_epochs"]:
            lr_scale = (epoch + 1) / config["warmup_epochs"]
        else:
            progress = (epoch - config["warmup_epochs"]) / (config["epochs"] - config["warmup_epochs"])
            lr_scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = config["lr"] * lr_scale
        
        # Train
        model.train()
        train_loss = 0
        
        for i, batch in enumerate(train_loader):
            enc = batch["encoder_input_ids"].to(device)
            dec = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            out = model(enc, dec, labels=labels)
            loss = out["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 50 == 0:
                print(f"  Batch {i}/{len(train_loader)}: loss={loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                enc = batch["encoder_input_ids"].to(device)
                dec = batch["decoder_input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                out = model(enc, dec, labels=labels)
                val_loss += out["loss"].item()
                
                preds = out["logits"].argmax(dim=-1)
                mask = labels != -100
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0
        
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "accuracy": val_acc})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/x1_26m_best.pt"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            for old in glob.glob("/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/x1_26m_*.pt"):
                if old != save_path:
                    os.remove(old)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved (val_loss={best_val_loss:.4f})")
    
    # Save final
    final_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/x1_26m_final.pt"
    torch.save(model.state_dict(), final_path)
    
    # Save tokenizer
    tok_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/tokenizer.json"
    tokenizer.save(tok_path)
    
    with open("/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/x1_26m_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final accuracy: {val_acc:.4f}")
    print(f"  Tokenizer saved: {tok_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()
