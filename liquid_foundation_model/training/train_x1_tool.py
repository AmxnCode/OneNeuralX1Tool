"""
One Neural X1 Tool - Function Calling Training Script

This script trains the encoder-decoder model for function calling / tool use.

Usage:
    # Pre-training (200B tokens)
    python train_x1_tool.py --phase pretrain --data the-stack,github --tokens 200B
    
    # Distillation from Claude/GPT-4
    python train_x1_tool.py --phase distill --teacher claude-3-5-haiku --data distill_data.jsonl
    
    # SFT on function calling
    python train_x1_tool.py --phase sft --data toolbench,gorilla
    
    # Fine-tune on custom tools
    python train_x1_tool.py --phase finetune --data my_tools.jsonl
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import logging

from liquid_foundation_model.model.configuration.config import X1ToolConfig, X1ToolTrainingConfig
from liquid_foundation_model.model.encoder_decoder_model import OneNeuralX1Tool

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FunctionCallDataset(Dataset):
    """
    Dataset for function calling training.
    
    Expected format (JSONL):
    {
        "query": "What's the weather in San Francisco?",
        "tools": "[{\"name\": \"get_weather\", \"parameters\": {\"location\": \"string\"}}]",
        "response": "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"San Francisco\"}}]"
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        phase: str = "sft",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.phase = phase
        
        # Load data
        self.data = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                for line in f:
                    self.data.append(json.loads(line))
        elif data_path.endswith(".json"):
            with open(data_path, "r") as f:
                self.data = json.load(f)
        else:
            # Try loading from HuggingFace datasets
            try:
                dataset = load_dataset(data_path, split="train")
                self.data = list(dataset)
            except:
                raise ValueError(f"Cannot load data from {data_path}")
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format for encoder-decoder
        query = item.get("query", item.get("input", ""))
        tools = item.get("tools", "")
        response = item.get("response", item.get("output", ""))
        
        # Create encoder input (query + tools)
        encoder_text = f"{query}"
        if tools:
            encoder_text += f"\n\nAvailable tools:\n{tools}"
        
        # Create decoder input (response)
        decoder_text = response
        
        # Tokenize
        encoder_tokens = self.tokenizer(
            encoder_text,
            truncation=True,
            max_length=self.max_length // 2,  # Encoder gets half the length
            padding="max_length",
            return_tensors="pt",
        )
        
        decoder_tokens = self.tokenizer(
            decoder_text,
            truncation=True,
            max_length=self.max_length // 2,  # Decoder gets half the length
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "encoder_input_ids": encoder_tokens["input_ids"].squeeze(),
            "encoder_attention_mask": encoder_tokens["attention_mask"].squeeze(),
            "decoder_input_ids": decoder_tokens["input_ids"].squeeze(),
            "decoder_attention_mask": decoder_tokens["attention_mask"].squeeze(),
            "labels": decoder_tokens["input_ids"].squeeze(),
        }


class DistillationDataset(Dataset):
    """
    Dataset for distillation training.
    
    Expected format (JSONL):
    {
        "query": "What's the weather?",
        "tools": "[...]",
        "teacher_response": "[{\"name\": \"get_weather\", ...}]"
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.data = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            with open(data_path, "r") as f:
                self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} distillation examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format for encoder-decoder
        query = item.get("query", "")
        tools = item.get("tools", "")
        teacher_response = item.get("teacher_response", "")
        
        # Create encoder input
        encoder_text = f"{query}"
        if tools:
            encoder_text += f"\n\nAvailable tools:\n{tools}"
        
        # Tokenize
        encoder_tokens = self.tokenizer(
            encoder_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt",
        )
        
        decoder_tokens = self.tokenizer(
            teacher_response,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "encoder_input_ids": encoder_tokens["input_ids"].squeeze(),
            "encoder_attention_mask": encoder_tokens["attention_mask"].squeeze(),
            "decoder_input_ids": decoder_tokens["input_ids"].squeeze(),
            "decoder_attention_mask": decoder_tokens["attention_mask"].squeeze(),
            "labels": decoder_tokens["input_ids"].squeeze(),
        }


def create_model(config: X1ToolConfig) -> OneNeuralX1Tool:
    """Create model from config."""
    model = OneNeuralX1Tool(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        dropout_rate=config.dropout_rate,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        tie_embeddings=config.tie_embeddings,
    )
    
    # Log model info
    param_counts = model.count_parameters()
    logger.info(f"Created model with {param_counts['total_millions']:.2f}M parameters")
    logger.info(f"  Encoder: {param_counts['encoder_millions']:.2f}M")
    logger.info(f"  Decoder: {param_counts['decoder_millions']:.2f}M")
    
    return model


def train_epoch(
    model: OneNeuralX1Tool,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    config: X1ToolTrainingConfig,
    epoch: int,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Move to device
        encoder_input_ids = batch["encoder_input_ids"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        with autocast(enabled=config.bf16):
            outputs = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(
    model: OneNeuralX1Tool,
    dataloader: DataLoader,
    device: torch.device,
):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            encoder_input_ids = batch["encoder_input_ids"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def generate_function_call(
    model: OneNeuralX1Tool,
    tokenizer,
    query: str,
    tools: str,
    device: torch.device,
    max_length: int = 128,
    temperature: float = 0.7,
):
    """Generate function call from query and tools."""
    model.eval()
    
    # Format encoder input
    encoder_text = f"{query}\n\nAvailable tools:\n{tools}"
    
    # Tokenize
    encoder_tokens = tokenizer(
        encoder_text,
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    ).to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            encoder_input_ids=encoder_tokens["input_ids"],
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or 0,
            bos_token_id=tokenizer.bos_token_id or 1,
            eos_token_id=tokenizer.eos_token_id or 2,
        )
    
    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Train One Neural X1 Tool")
    parser.add_argument("--phase", type=str, required=True, choices=["pretrain", "distill", "sft", "finetune"],
                        help="Training phase")
    parser.add_argument("--data", type=str, required=True, help="Data path or HuggingFace dataset name")
    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "medium"],
                        help="Model size")
    parser.add_argument("--teacher", type=str, default="claude-3-5-haiku",
                        help="Teacher model for distillation")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/x1-tool",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create config
    if args.model_size == "tiny":
        config = X1ToolConfig.tiny()
    elif args.model_size == "small":
        config = X1ToolConfig.small()
    else:
        config = X1ToolConfig.medium()
    
    # Create training config
    if args.phase == "pretrain":
        train_config = X1ToolTrainingConfig.pretrain()
    elif args.phase == "distill":
        train_config = X1ToolTrainingConfig.distill()
    else:
        train_config = X1ToolTrainingConfig.sft()
    
    # Override with args
    train_config.batch_size = args.batch_size
    train_config.learning_rate = args.learning_rate
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Load tokenizer (using a small BPE tokenizer)
    # For now, we'll use a simple tokenizer
    # TODO: Train a custom SentencePiece tokenizer with 8192 vocab
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.vocab_size = config.vocab_size
    
    # Create dataset
    if args.phase == "distill":
        dataset = DistillationDataset(args.data, tokenizer, max_length=config.max_seq_len)
    else:
        dataset = FunctionCallDataset(args.data, tokenizer, max_length=config.max_length, phase=args.phase)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    
    # Create scheduler
    total_steps = len(dataloader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Create scaler for mixed precision
    scaler = GradScaler(enabled=train_config.fp16)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info(f"Total steps: {total_steps}")
    
    best_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, dataloader, optimizer, scheduler, scaler, device, train_config, epoch
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        eval_loss = evaluate(model, dataloader, device)
        logger.info(f"Eval loss: {eval_loss:.4f}")
        
        # Save best model
        if eval_loss < best_loss:
            best_loss = eval_loss
            save_path = os.path.join(args.output_dir, "best_model.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "loss": eval_loss,
                "config": config.to_dict(),
            }, save_path)
            logger.info(f"Saved best model to {save_path}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch + 1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "loss": eval_loss,
                "config": config.to_dict(),
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
    
    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
