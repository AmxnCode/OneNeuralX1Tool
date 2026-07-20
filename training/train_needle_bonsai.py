"""
Training Script for NeedleBonsai 27M Model

Trains on function-calling data in Needle format (query, tools, answers).
Uses BPE tokenizer trained on the data.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import sys

sys.path.insert(0, '/OneNeuralX1Tool')

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from liquid_foundation_model.model.needle_bonsai import create_model


class FnCallingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        print(f"  Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        query = ex.get("query", "")
        tools = ex.get("tools", "[]")
        answers = ex.get("answers", "")

        encoder_text = f"Query: {query}\nTools: {tools}"
        decoder_text = f"{answers}"

        enc_tokens = self.tokenizer.encode(encoder_text)
        dec_tokens = self.tokenizer.encode(decoder_text)

        enc_ids = enc_tokens.ids[:self.max_len]
        dec_ids = [1] + dec_tokens.ids[:self.max_len - 1]
        labels = dec_ids[1:] + [-100]

        # Pad
        enc_ids = enc_ids + [0] * max(0, self.max_len - len(enc_ids))
        dec_ids = dec_ids + [0] * max(0, self.max_len - len(dec_ids))
        labels = labels + [-100] * max(0, self.max_len - len(labels))

        return {
            "encoder_input_ids": torch.tensor(enc_ids[:self.max_len], dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_ids[:self.max_len], dtype=torch.long),
            "labels": torch.tensor(labels[:self.max_len], dtype=torch.long),
        }


def build_tokenizer(data_path, vocab_size=8192):
    print("  Building BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        min_frequency=2,
    )

    texts = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                texts.append(ex.get("query", ""))
                texts.append(ex.get("answers", ""))
                texts.append(ex.get("tools", ""))

    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
    tokenizer.enable_truncation(max_length=256)

    print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer


def train():
    print("=" * 60)
    print("Training NeedleBonsai 27M Model")
    print("=" * 60)

    config = {
        "vocab_size": 8192,
        "batch_size": 8,
        "lr": 5e-4,
        "epochs": 50,
        "max_len": 256,
        "warmup_epochs": 3,
        "grad_accum_steps": 4,
        "patience": 10,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    data_path = "/OneNeuralX1Tool/data/needle_format.jsonl"
    checkpoint_dir = Path("/OneNeuralX1Tool/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Build tokenizer
    print("\n1. Building tokenizer...")
    tokenizer = build_tokenizer(data_path, config["vocab_size"])

    # Save tokenizer
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"  Saved tokenizer to {tokenizer_path}")

    # Create model
    print("\n2. Creating model...")
    model = create_model().to(device)
    info = model.get_info()
    print(f"  Parameters: {info['parameters']:,} ({info['parameters_millions']:.2f}M)")

    # Dataset
    print("\n3. Loading dataset...")
    dataset = FnCallingDataset(data_path, tokenizer, max_len=config["max_len"])

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=0)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Optimizer with higher weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.1)

    # Training
    print(f"\n4. Training for {config['epochs']} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 40)

        # LR schedule: cosine with warmup
        if epoch < config["warmup_epochs"]:
            lr_scale = (epoch + 1) / config["warmup_epochs"]
        else:
            progress = (epoch - config["warmup_epochs"]) / (config["epochs"] - config["warmup_epochs"])
            lr_scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        for pg in optimizer.param_groups:
            pg['lr'] = config["lr"] * lr_scale

        # Train
        model.train()
        train_loss = 0
        train_batches = 0
        t0 = time.time()

        for i, batch in enumerate(train_loader):
            enc = batch["encoder_input_ids"].to(device)
            dec = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(enc, dec, labels=labels)
            loss = out["loss"] / config["grad_accum_steps"]
            loss.backward()

            if (i + 1) % config["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += out["loss"].item()
            train_batches += 1

            if i % 50 == 0:
                print(f"  Batch {i}/{len(train_loader)}: loss={out['loss'].item():.4f}")

        train_loss /= train_batches
        dt = time.time() - t0

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

        print(f"  Train loss: {train_loss:.4f} ({dt:.1f}s)")
        print(f"  Val loss:   {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": config["lr"] * lr_scale,
        })

        # Save best with early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / "needle_bonsai_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        # Save history after each epoch
        with open(checkpoint_dir / "history.json", 'w') as f:
            json.dump(history, f, indent=2)

    # Save final
    final_path = checkpoint_dir / "needle_bonsai_final.pt"
    torch.save(model.state_dict(), final_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Parameters: {info['parameters']:,} ({info['parameters_millions']:.2f}M)")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final accuracy: {val_acc:.4f}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    train()
