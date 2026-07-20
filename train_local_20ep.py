"""
Local Training - 20 epochs with Needle data
Runs on CPU (avoids MPS disk fill issue)
"""

import os, json, sys, torch, random, time
sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

NEEDLE_DATA = "/Users/amanpreetsingh/projects/experiment/needle/data/needle_tools.jsonl"
CHECKPOINT_DIR = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 50)
print("  20-Epoch Training (Needle Data, CPU)")
print("=" * 50)

# Load Needle data
print("\n1. Loading Needle data...")
data = []
with open(NEEDLE_DATA) as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
print(f"   Loaded {len(data)} examples")

# Build tokenizer
print("\n2. Building tokenizer...")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tool_names = ["search_web", "get_weather", "send_email", "create_event",
              "get_stock_price", "set_reminder", "play_music", "calculate"]
# JSON formatting tokens so they don't get split
json_tokens = ['{"name":', '{"arguments":', '"}', '},', '"', ":", "{", "}", "[", "]"]
special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + tool_names + json_tokens
tr = trainers.BpeTrainer(vocab_size=8192, special_tokens=special_tokens,
                         min_frequency=2, continuing_subword_prefix="")
texts = []
for item in data:
    texts.extend([item["query"], item["answers"]])
tokenizer.train_from_iterator(texts, trainer=tr)
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
tokenizer.enable_truncation(max_length=256)
print(f"   Vocab: {tokenizer.get_vocab_size()}")

# Load model
print("\n3. Loading model...")
from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2
model = OneNeuralX1ToolV2(
    vocab_size=8192, hidden_size=512,
    num_encoder_layers=12, num_decoder_layers=6,
    num_attention_heads=8, num_key_value_heads=4,
    max_loops=4, num_experts=4, expert_dim=2048,
).to("cpu")
params = sum(p.numel() for p in model.parameters())
print(f"   {params/1e6:.2f}M params")

# Dataset
class DS(Dataset):
    def __init__(self, d):
        self.d = d
    def __len__(self):
        return len(self.d)
    def __getitem__(self, i):
        ex = self.d[i]
        q = tokenizer.encode(ex["query"]).ids[:256]
        a = tokenizer.encode(ex["answers"]).ids[:256]
        q = q + [0] * (256 - len(q))
        inp = [1] + a[:-1] + [0] * (256 - len(a))
        lbl = a + [-100] * (256 - len(a))
        return (torch.tensor(q[:256]), torch.tensor(inp[:256]), torch.tensor(lbl[:256]))

train_size = int(0.9 * len(data))
train_ds = DS(data[:train_size])
val_ds = DS(data[train_size:])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)
print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")

# Train
print("\n4. Training 20 epochs...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
best_acc = 0
t0 = time.time()

for epoch in range(20):
    model.train()
    total_loss = 0
    for q, inp, lbl in train_loader:
        out = model(q, inp, labels=lbl)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for q, inp, lbl in val_loader:
            out = model(q, inp, labels=lbl)
            val_loss += out["loss"].item()
            preds = out["logits"].argmax(-1)
            mask = lbl != -100
            correct += (preds[mask] == lbl[mask]).sum().item()
            total += mask.sum().item()

    val_acc = correct / total if total > 0 else 0
    elapsed = (time.time() - t0) / 60
    print(f"   Epoch {epoch+1:2d}/20 | Loss: {avg_loss:.4f} | ValLoss: {val_loss/len(val_loader):.4f} | ValAcc: {val_acc:.4f} | {elapsed:.1f}min")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "x1_26m_final.pt"))
        print(f"     -> Saved best (acc: {val_acc:.4f})")

elapsed = (time.time() - t0) / 60
print(f"\n5. Done! Best acc: {best_acc:.4f} | Total: {elapsed:.1f}min")

# Test
print("\n6. Testing...")
queries = [
    "What's the weather in Tokyo?",
    "Check AAPL stock price",
    "Send email to john@example.com",
    "Calculate 15 * 3",
    "Set a reminder for tomorrow",
]
model.eval()
for q in queries:
    enc = tokenizer.encode(q)
    inp = torch.tensor([enc.ids])
    with torch.no_grad():
        out = model.generate(inp, max_length=50, temperature=0.1, do_sample=False)
    ids = [t for t in out[0].tolist() if t > 2]
    text = tokenizer.decode(ids)
    print(f"   Q: {q}")
    print(f"   A: {text[:120]}")
    print()

print("=" * 50)
print("  Training Complete!")
print("=" * 50)
