import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import logging

from liquid_foundation_model.model.configuration.config import LFMConfig, TrainingConfig
from liquid_foundation_model.model.liquid_foundation_model import LiquidFoundationModelForCausalLM

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # Create labels (shift input_ids right for causal LM)
        labels = input_ids.clone()
        # Set padding tokens to -100 so they're ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def train(args):
    # Load configuration
    config = LFMConfig.from_pretrained(args.model_size)
    
    # Configure HRM if enabled
    if args.enable_hrm:
        config.enable_hrm = True
        config.hrm_reasoning_steps = args.hrm_reasoning_steps
        logger.info(f"HRM enabled with {args.hrm_reasoning_steps} reasoning steps")
    else:
        config.enable_hrm = False
        logger.info("HRM disabled")
    
    training_config = TrainingConfig()
    
    # Update training config from args
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    training_config.max_steps = args.max_steps
    training_config.warmup_steps = args.warmup_steps
    
    # Create model
    logger.info(f"Creating model: {args.model_size}")
    model = LiquidFoundationModelForCausalLM(config)
    
    # Load tokenizer
    if args.tokenizer_path:
        logger.info(f"Loading tokenizer from {args.tokenizer_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            logger.info(f"Loaded tokenizer with vocabulary size: {len(tokenizer)}")
            logger.info(f"Special tokens: {tokenizer.special_tokens_map}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")
            logger.info("Falling back to GPT-2 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # Add LFM2 special tokens to the tokenizer
            special_tokens = {
                "bos_token": "<|startoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|pad|>",
            }
            tokenizer.add_special_tokens(special_tokens)
            
            # Add additional special tokens for chat and tool use
            additional_special_tokens = [
                "<|im_start|>",
                "<|tool_list_start|>", "<|tool_list_end|>",
                "<|tool_call_start|>", "<|tool_call_end|>",
                "<|tool_response_start|>", "<|tool_response_end|>"
            ]
            tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    else:
        logger.info("No tokenizer path provided, using GPT-2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Add LFM2 special tokens to the tokenizer
        special_tokens = {
            "bos_token": "<|startoftext|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|pad|>",
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # Add additional special tokens for chat and tool use
        additional_special_tokens = [
            "<|im_start|>",
            "<|tool_list_start|>", "<|tool_list_end|>",
            "<|tool_call_start|>", "<|tool_call_end|>",
            "<|tool_response_start|>", "<|tool_response_end|>"
        ]
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    
    # Resize token embeddings to match tokenizer
    logger.info(f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"]["text"]
        val_texts = dataset["validation"]["text"]
    else:
        # Add support for other datasets here
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Filter out empty texts
    train_texts = [text for text in train_texts if text.strip()]
    val_texts = [text for text in val_texts if text.strip()]
    
    logger.info(f"Training on {len(train_texts)} examples, validating on {len(val_texts)} examples")
    
    # Create datasets
    logger.info("Creating datasets")
    train_dataset = TextDataset(train_texts, tokenizer, max_length=args.max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=args.max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    total_steps = min(training_config.max_steps, len(train_loader) * args.num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Enable mixed precision training if available
    if training_config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("Using bfloat16 mixed precision training")
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        amp_dtype = torch.bfloat16
    elif training_config.fp16 and torch.cuda.is_available():
        logger.info("Using float16 mixed precision training")
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        amp_dtype = torch.float16
    else:
        logger.info("Mixed precision training not available, using float32")
        scaler = None
        amp_dtype = torch.float32
    
    # Training loop
    logger.info("Starting training")
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs[0]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs[0]
            
            # Backward pass with mixed precision
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            global_step += 1
            
            # Save checkpoint every 20 steps
            if global_step % 20 == 0:
                logger.info(f"Saving checkpoint at step {global_step}")
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                # Save training args
                with open(os.path.join(checkpoint_path, "training_args.txt"), "w") as f:
                    f.write(str(args))
                    
                # Save optimizer and scheduler states
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'global_step': global_step,
                    'epoch': epoch,
                }, os.path.join(checkpoint_path, "optimizer.pt"))
            
            if global_step >= training_config.max_steps:
                break
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs[0]
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"Validation loss improved to {best_val_loss:.4f}. Saving model.")
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save model and tokenizer
            model_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(model_path, exist_ok=True)
            
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Save training args
            with open(os.path.join(model_path, "training_args.txt"), "w") as f:
                f.write(str(args))
        
        if global_step >= training_config.max_steps:
            logger.info(f"Reached max steps ({training_config.max_steps}). Stopping training.")
            break
    
    # Save final model
    logger.info("Training complete. Saving final model.")
    final_model_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_model_path, exist_ok=True)
    
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train the Liquid Foundation Model")
    parser.add_argument("--model-size", type=str, default="350M", help="Model size (350M, 700M, 1.2B)")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to train on")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory for model checkpoints")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to the official LFM2 tokenizer")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum number of training steps")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--enable-hrm", action="store_true", help="Enable HRM reasoning")
    parser.add_argument("--hrm-reasoning-steps", type=int, default=3, help="Number of HRM reasoning steps")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()