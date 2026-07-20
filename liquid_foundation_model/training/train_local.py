"""
Train X1 Tool entirely on your local device.

No API needed. No cloud needed. Just your computer.

Usage:
    # Step 1: Generate training data
    python generate_local_data.py --num-samples 10000 --output data/tool_calling.jsonl
    
    # Step 2: Train with local teacher (optional)
    python local_distill.py --teacher qwen2.5-0.5b --data data/tool_calling.jsonl
    
    # Step 3: Train the model
    python train_local.py --data data/tool_calling.jsonl --epochs 10
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liquid_foundation_model.model.configuration.config import X1ToolConfig
from liquid_foundation_model.model.encoder_decoder_model import OneNeuralX1Tool

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FunctionCallDataset(Dataset):
    """Dataset for function calling data."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create encoder input (query + tools)
        encoder_text = item.get("query", "")
        tools = item.get("tools", "")
        if tools:
            encoder_text += f"\n\nTools:\n{tools}"
        
        # Create decoder input (response)
        decoder_text = item.get("response", item.get("teacher_response", ""))
        
        # Simple tokenizer (character-level for now)
        # TODO: Use proper SentencePiece tokenizer
        encoder_tokens = self._tokenize(encoder_text)
        decoder_tokens = self._tokenize(decoder_text)
        
        # Pad to max_length
        encoder_tokens = self._pad(encoder_tokens, self.max_length)
        decoder_tokens = self._pad(decoder_tokens, self.max_length)
        
        # Create labels (shift decoder tokens)
        labels = decoder_tokens[1:] + [0]  # Add pad token
        
        return {
            "encoder_input_ids": torch.tensor(encoder_tokens, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization (character-level)."""
        # This is a placeholder - use proper tokenizer in production
        return [ord(c) % 8192 for c in text[:self.max_length]]
    
    def _pad(self, tokens: List[int], max_length: int) -> List[int]:
        """Pad tokens to max_length."""
        if len(tokens) >= max_length:
            return tokens[:max_length]
        return tokens + [0] * (max_length - len(tokens))


class SimpleTokenizer:
    """Simple tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [ord(c) % self.vocab_size for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return "".join([chr(t) for t in tokens if 32 <= t < 127])


def train_epoch(
    model: OneNeuralX1Tool,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Move to device
        encoder_input_ids = batch["encoder_input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        
        loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
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
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            
            loss = outputs["loss"]
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def test_generation(
    model: OneNeuralX1Tool,
    tokenizer: SimpleTokenizer,
    query: str,
    tools: str,
    device: torch.device,
):
    """Test model generation."""
    model.eval()
    
    # Encode query
    encoder_text = f"{query}\n\nTools:\n{tools}"
    encoder_tokens = tokenizer.encode(encoder_text)
    encoder_input_ids = torch.tensor([encoder_tokens], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            encoder_input_ids=encoder_input_ids,
            max_length=64,
            temperature=0.7,
            do_sample=True,
        )
    
    # Decode
    output_text = tokenizer.decode(output_ids[0].tolist())
    
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Train X1 Tool locally")
    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small"],
                        help="Model size")
    parser.add_argument("--output", type=str, default="./checkpoints/x1-tool-local",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create config
    if args.model_size == "tiny":
        config = X1ToolConfig.tiny()
    else:
        config = X1ToolConfig.small()
    
    # Create model
    model = OneNeuralX1Tool(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        dropout_rate=config.dropout_rate,
        max_seq_len=config.max_seq_len,
    )
    
    # Move to device
    model = model.to(device)
    
    # Log model info
    param_counts = model.count_parameters()
    logger.info(f"Created model with {param_counts['total_millions']:.2f}M parameters")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(config.vocab_size)
    
    # Create dataset
    dataset = FunctionCallDataset(args.data, tokenizer, max_length=args.max_length)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        eval_loss = evaluate(model, dataloader, device)
        logger.info(f"Eval loss: {eval_loss:.4f}")
        
        # Save best model
        if eval_loss < best_loss:
            best_loss = eval_loss
            save_path = os.path.join(args.output, "best_model.pt")
            os.makedirs(args.output, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": eval_loss,
                "config": config.to_dict(),
            }, save_path)
            logger.info(f"Saved best model to {save_path}")
        
        # Test generation
        if (epoch + 1) % 5 == 0:
            logger.info("\nTesting generation...")
            test_query = "What's the weather in New York?"
            test_tools = '[{"name": "get_weather", "parameters": {"location": "string"}}]'
            
            response = test_generation(model, tokenizer, test_query, test_tools, device)
            logger.info(f"Query: {test_query}")
            logger.info(f"Response: {response}")
    
    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
