"""
Quick Training Script - CPU Mode
Small model, few epochs, quick results.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import random

import sys
sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')

from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2


class SimpleDataset(Dataset):
    def __init__(self, data_path, max_len=128):
        with open(data_path) as f:
            self.examples = [json.loads(line) for line in f if line.strip()]
        self.max_len = max_len
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Simple char-level encoding
        query = ex["query"][:self.max_len]
        answer = ex["answers"][:self.max_len]
        
        # Pad
        query_ids = [ord(c) % 8192 for c in query] + [0] * (self.max_len - len(query))
        answer_ids = [1] + [ord(c) % 8192 for c in answer] + [0] * (self.max_len - len(answer) - 1)
        labels = [ord(c) % 8192 for c in answer] + [-100] * (self.max_len - len(answer))
        
        return {
            "encoder_input_ids": torch.tensor(query_ids[:self.max_len]),
            "decoder_input_ids": torch.tensor(answer_ids[:self.max_len]),
            "labels": torch.tensor(labels[:self.max_len]),
        }


def train():
    print("=" * 60)
    print("Quick Training - CPU Mode")
    print("=" * 60)
    
    # Config - SMALL for quick test
    config = {
        "vocab_size": 8192,
        "hidden_size": 256,
        "num_encoder_layers": 4,
        "num_decoder_layers": 2,
        "max_loops": 2,
        "num_experts": 4,
        "expert_dim": 512,
        "batch_size": 4,
        "lr": 1e-4,
        "epochs": 3,
        "max_len": 128,
        "max_samples": 500,
    }
    
    device = torch.device("cpu")
    
    # Model
    print("\n1. Creating model...")
    model = OneNeuralX1ToolV2(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        max_loops=config["max_loops"],
        num_experts=config["num_experts"],
        expert_dim=config["expert_dim"],
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Dataset
    print("\n2. Loading data...")
    data_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/data/multi_tool_chains.jsonl"
    dataset = SimpleDataset(data_path, max_len=config["max_len"])
    
    # Subsample for quick training
    if len(dataset) > config["max_samples"]:
        indices = random.sample(range(len(dataset)), config["max_samples"])
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])
    
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    # Training
    print(f"\n3. Training for {config['epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
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
            
            if i % 20 == 0:
                print(f"  Batch {i}: loss={loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                enc = batch["encoder_input_ids"].to(device)
                dec = batch["decoder_input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                out = model(enc, dec, labels=labels)
                val_loss += out["loss"].item()
        
        val_loss /= len(val_loader)
        
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/quick_best.pt"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model")
    
    # Save final
    final_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/quick_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Saved final model")
    
    # Quick test
    print("\n4. Quick inference test...")
    model.eval()
    test_query = "What's the weather in Paris?"
    test_ids = torch.tensor([[ord(c) % 8192 for c in test_query] + [0] * (config["max_len"] - len(test_query))])
    
    with torch.no_grad():
        gen = model.generate(test_ids, max_length=32, temperature=0.1, do_sample=False)
    
    print(f"   Query: {test_query}")
    print(f"   Generated: {gen[0].tolist()}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
