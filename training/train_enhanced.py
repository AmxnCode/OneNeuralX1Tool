"""
Training Script for Enhanced X1 Tool Model

Features:
- Distillation from Phi-3-mini teacher
- Multi-tool chain training
- ACT halting loss
- MoE load balancing loss
- Mixed precision training
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional
import time

import sys
sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')

from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2


class MultiToolChainDataset(Dataset):
    """Dataset for multi-tool chain training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_enc_len: int = 1024,
        max_dec_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        
        # Load data
        with open(data_path) as f:
            self.examples = [json.loads(line) for line in f if line.strip()]
        
        print(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Tokenize query (encoder input)
        query_tokens = self.tokenizer.encode(
            ex["query"],
            max_length=self.max_enc_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).squeeze()
        
        # Tokenize answer (decoder target)
        answer_text = ex["answers"]
        answer_tokens = self.tokenizer.encode(
            answer_text,
            max_length=self.max_dec_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).squeeze()
        
        # Decoder input (shift right)
        decoder_input = torch.cat([
            torch.tensor([1]),  # BOS
            answer_tokens[:-1]
        ])
        
        return {
            "encoder_input_ids": query_tokens,
            "decoder_input_ids": decoder_input,
            "labels": answer_tokens,
        }


class DistillationLoss(nn.Module):
    """
    Combined loss for training:
    1. Cross-entropy loss (student vs labels)
    2. Distillation loss (student vs teacher logits)
    3. ACT halting loss (encourage early exit)
    4. MoE load balancing loss (uniform expert usage)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,  # Weight for distillation
        beta: float = 0.1,   # Weight for ACT loss
        gamma: float = 0.01, # Weight for MoE loss
        temperature: float = 3.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        loop_probs: Optional[List[torch.Tensor]] = None,
        load_balance_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            student_logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
            teacher_logits: [batch, seq_len, vocab_size] (optional)
            loop_probs: List of [batch, seq_len, 1] (optional)
            load_balance_loss: scalar (optional)
            
        Returns:
            Dict with total_loss, ce_loss, distill_loss, etc.
        """
        # 1. Cross-entropy loss
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        total_loss = ce_loss
        
        # 2. Distillation loss (if teacher provided)
        distill_loss = torch.tensor(0.0, device=student_logits.device)
        if teacher_logits is not None:
            # Soft targets
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            distill_loss = F.kl_div(
                student_soft,
                teacher_soft,
                reduction='batchmean',
            ) * (self.temperature ** 2)
            
            total_loss = total_loss * (1 - self.alpha) + distill_loss * self.alpha
        
        # 3. ACT halting loss (encourage fewer loops)
        act_loss = torch.tensor(0.0, device=student_logits.device)
        if loop_probs:
            # Sum of halting probabilities should be close to 1
            total_halt = sum(loop_probs)
            act_loss = F.mse_loss(total_halt, torch.ones_like(total_halt))
            total_loss = total_loss + self.beta * act_loss
        
        # 4. MoE load balancing loss
        if load_balance_loss is not None:
            total_loss = total_loss + self.gamma * load_balance_loss
        
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "distill_loss": distill_loss,
            "act_loss": act_loss,
            "moe_loss": load_balance_loss or torch.tensor(0.0),
        }


class SimpleTokenizer:
    """Simple tokenizer for training."""
    
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
    
    def encode(
        self,
        text: str,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """Encode text to token IDs."""
        # Simple character-level encoding for testing
        tokens = [ord(c) % self.vocab_size for c in text[:max_length]]
        
        # Add BOS/EOS
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        
        # Truncate
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Pad
        if padding == "max_length":
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens


def train_epoch(
    model: OneNeuralX1ToolV2,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: DistillationLoss,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce = 0
    total_steps = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        encoder_input = batch["encoder_input_ids"].to(device)
        decoder_input = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        output = model(
            encoder_input_ids=encoder_input,
            decoder_input_ids=decoder_input,
            labels=labels,
        )
        
        # Compute loss
        losses = loss_fn(
            student_logits=output["logits"],
            labels=labels,
            loop_probs=output.get("loop_probs"),
            load_balance_loss=output.get("load_balance_loss"),
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses["total_loss"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Accumulate metrics
        total_loss += losses["total_loss"].item()
        total_ce += losses["ce_loss"].item()
        total_steps += 1
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"loss={losses['total_loss'].item():.4f}, "
                  f"ce={losses['ce_loss'].item():.4f}")
    
    return {
        "loss": total_loss / total_steps,
        "ce_loss": total_ce / total_steps,
    }


def evaluate(
    model: OneNeuralX1ToolV2,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            encoder_input = batch["encoder_input_ids"].to(device)
            decoder_input = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            output = model(
                encoder_input_ids=encoder_input,
                decoder_input_ids=decoder_input,
                labels=labels,
            )
            
            # Compute loss
            shift_logits = output["logits"][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            total_loss += loss.item()
            
            # Compute accuracy
            preds = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            total_correct += (preds[mask] == shift_labels[mask]).sum().item()
            total_tokens += mask.sum().item()
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": total_correct / total_tokens if total_tokens > 0 else 0,
    }


def main():
    """Main training function."""
    print("=" * 60)
    print("Training Enhanced X1 Tool Model")
    print("=" * 60)
    
    # Configuration
    config = {
        "vocab_size": 8192,
        "hidden_size": 512,
        "num_encoder_layers": 12,
        "num_decoder_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_loops": 4,
        "num_experts": 4,
        "expert_dim": 2048,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "num_epochs": 5,
        "warmup_steps": 100,
        "max_enc_len": 512,
        "max_dec_len": 256,
    }
    
    # Device - use CPU to avoid MPS graph cache filling disk
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("\n1. Creating model...")
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
    
    info = model.get_model_info()
    print(f"   Parameters: {info['parameters']:,}")
    print(f"   Memory: {info['memory_mb']:.2f} MB")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(config["vocab_size"])
    
    # Load dataset
    print("\n2. Loading dataset...")
    data_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/data/multi_tool_chains.jsonl"
    
    if not Path(data_path).exists():
        print("   Dataset not found. Generating...")
        from generate_multi_tool_data import MultiToolChainGenerator
        generator = MultiToolChainGenerator()
        examples = generator.generate_dataset(2000)
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        generator.save_dataset(examples, data_path)
    
    dataset = MultiToolChainDataset(
        data_path,
        tokenizer,
        max_enc_len=config["max_enc_len"],
        max_dec_len=config["max_dec_len"],
    )
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Optimizer and scheduler
    print("\n3. Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )
    
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=config["warmup_steps"] / total_steps,
    )
    
    # Loss function
    loss_fn = DistillationLoss(
        alpha=0.5,
        beta=0.1,
        gamma=0.01,
        temperature=3.0,
    )
    
    # Training loop
    print(f"\n4. Training for {config['num_epochs']} epochs...")
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, epoch
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        print(f"  Val loss: {val_metrics['loss']:.4f}, accuracy: {val_metrics['accuracy']:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/enhanced_x1_best.pt"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
    
    # Save final model
    final_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/enhanced_x1_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Saved final model to {final_path}")
    
    # Save history
    history_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
