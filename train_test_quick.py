"""
1-Epoch Quick Train + Test
Fast feedback loop.
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import sys

sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')
from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2


class Dataset(Dataset):
    def __init__(self, path, tok, max_len=256):
        with open(path) as f:
            self.data = [json.loads(l) for l in f if l.strip()]
        self.tok, self.max_len = tok, max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        q = self.tok.encode(ex["query"]).ids[:self.max_len]
        a = self.tok.encode(ex["answers"]).ids[:self.max_len]
        q = q + [0]*(self.max_len - len(q))
        inp = [1] + a[:-1] + [0]*(self.max_len - len(a))
        lbl = a + [-100]*(self.max_len - len(a))
        return (torch.tensor(q[:self.max_len]), torch.tensor(inp[:self.max_len]), torch.tensor(lbl[:self.max_len]))


def main():
    print("=" * 50)
    print("  1-Epoch Train + Test")
    print("=" * 50)

    device = torch.device("cpu")
    data_path = "data/multi_tool_chains.jsonl"

    # Build tokenizer
    print("\n1. Tokenizer...")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=8192, special_tokens=["[PAD]","[BOS]","[EOS]","[UNK]"], min_frequency=2)
    texts = []
    with open(data_path) as f:
        for l in f:
            ex = json.loads(l)
            texts.extend([ex["query"], ex["answers"]])
    tok.train_from_iterator(texts, trainer=tr)
    tok.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
    tok.enable_truncation(max_length=256)
    print(f"   Vocab: {tok.get_vocab_size()}")

    # Model
    print("\n2. Model...")
    model = OneNeuralX1ToolV2(
        vocab_size=8192, hidden_size=420, num_encoder_layers=10,
        num_decoder_layers=5, num_attention_heads=6, num_key_value_heads=3,
        max_loops=3, num_experts=4, expert_dim=1280,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"   {params/1e6:.2f}M params")

    # Data
    print("\n3. Data...")
    ds = Dataset(data_path, tok)
    train_ds, val_ds = torch.utils.data.random_split(ds, [int(0.9*len(ds)), len(ds)-int(0.9*len(ds))])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=0)
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Train 1 epoch
    print("\n4. Training 1 epoch...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    total_loss = 0

    for i, (q, inp, lbl) in enumerate(train_loader):
        q, inp, lbl = q.to(device), inp.to(device), lbl.to(device)
        out = model(q, inp, labels=lbl)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if i % 50 == 0:
            print(f"   Batch {i}: loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"   Train loss: {avg_loss:.4f}")

    # Validate
    model.eval()
    val_loss = 0
    correct = total = 0
    with torch.no_grad():
        for q, inp, lbl in val_loader:
            q, inp, lbl = q.to(device), inp.to(device), lbl.to(device)
            out = model(q, inp, labels=lbl)
            val_loss += out["loss"].item()
            preds = out["logits"].argmax(-1)
            mask = lbl != -100
            correct += (preds[mask] == lbl[mask]).sum().item()
            total += mask.sum().item()
    print(f"   Val loss: {val_loss/len(val_loader):.4f}, Acc: {correct/total:.4f}")

    # Save
    torch.save(model.state_dict(), "checkpoints/x1_26m_final.pt")
    print("   Saved!")

    # Test
    print("\n5. Testing...")
    queries = [
        "What is the weather in Paris?",
        "Get Apple stock price",
        "Send email to john@example.com",
        "Calculate 15 * 3",
        "Set a reminder for tomorrow",
    ]

    for q in queries:
        enc = tok.encode(q)
        inp = torch.tensor([enc.ids])
        with torch.no_grad():
            out = model.generate(inp, max_length=50, temperature=0.1, do_sample=False)
        ids = [t for t in out[0].tolist() if t > 2]
        text = tok.decode(ids)
        print(f"\n   Q: {q}")
        print(f"   A: {text[:100]}")

    print("\n" + "=" * 50)
    print("  Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
