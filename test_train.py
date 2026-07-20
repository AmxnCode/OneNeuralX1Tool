"""
Fast test - 1000 examples, 3 epochs, small batches
"""
import json, sys, time
sys.path.insert(0, '/OneNeuralX1Tool')

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from liquid_foundation_model.model.needle_bonsai import create_model


class FnCallingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128, max_examples=1000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(data_path) as f:
            for line in f:
                if line.strip() and len(self.examples) < max_examples:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc_text = f"Query: {ex.get('query','')}\nTools: {ex.get('tools','[]')}"
        dec_text = ex.get('answers', '')

        enc_ids = self.tokenizer.encode(enc_text).ids[:self.max_len]
        dec_ids = [1] + self.tokenizer.encode(dec_text).ids[:self.max_len - 1]
        labels = dec_ids[1:] + [-100]

        enc_ids += [0] * max(0, self.max_len - len(enc_ids))
        dec_ids += [0] * max(0, self.max_len - len(dec_ids))
        labels += [-100] * max(0, self.max_len - len(labels))

        return {
            "encoder_input_ids": torch.tensor(enc_ids[:self.max_len], dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_ids[:self.max_len], dtype=torch.long),
            "labels": torch.tensor(labels[:self.max_len], dtype=torch.long),
        }


# 1. Build tokenizer (save it!)
print("1. Tokenizer...")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=["[PAD]","[BOS]","[EOS]","[UNK]"], min_frequency=2)
texts = []
with open('/OneNeuralX1Tool/data/needle_format.jsonl') as f:
    for line in f:
        if line.strip():
            ex = json.loads(line)
            texts.append(ex.get("query",""))
            texts.append(ex.get("answers",""))
tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
tokenizer.enable_truncation(max_length=128)

ckpt_dir = Path('/OneNeuralX1Tool/checkpoints')
tokenizer.save(str(ckpt_dir / "tokenizer.json"))
print(f"   Saved (vocab={tokenizer.get_vocab_size()})")

# 2. Model
print("2. Model...")
model = create_model()
print(f"   {model.get_info()['parameters_millions']:.2f}M params")

# 3. Data (only 1000 examples, max_len=128)
print("3. Data (1000 examples)...")
dataset = FnCallingDataset('/OneNeuralX1Tool/data/needle_format.jsonl', tokenizer, max_len=128, max_examples=1000)
split = int(0.9 * len(dataset))
train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)
print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")

# 4. Train 3 epochs
print("4. Training 3 epochs...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

for epoch in range(3):
    t0 = time.time()
    model.train()
    tloss = tn = 0
    for batch in train_loader:
        out = model(batch["encoder_input_ids"], batch["decoder_input_ids"], labels=batch["labels"])
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tloss += loss.item(); tn += 1

    model.eval()
    vloss = correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch["encoder_input_ids"], batch["decoder_input_ids"], labels=batch["labels"])
            vloss += out["loss"].item()
            preds = out["logits"].argmax(dim=-1)
            mask = batch["labels"] != -100
            correct += (preds[mask] == batch["labels"][mask]).sum().item()
            total += mask.sum().item()

    dt = time.time() - t0
    print(f"   Epoch {epoch+1}: train={tloss/tn:.4f} val={vloss/len(val_loader):.4f} acc={correct/total:.4f} ({dt:.0f}s)")

# 5. Save
torch.save(model.state_dict(), ckpt_dir / "needle_bonsai_best.pt")
print("5. Saved model")

# 6. TEST
print("\n6. TESTING GENERATION...")
model.eval()
tests = [
    ("What's the weather in Paris?", '[{"type":"function","function":{"name":"get_weather","description":"Get weather.","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]'),
    ("Turn on bedroom lights", '[{"type":"function","function":{"name":"control_lights","description":"Control lights.","parameters":{"type":"object","properties":{"room":{"type":"string"},"action":{"type":"string"}},"required":["room","action"]}}}]'),
    ("Set a timer for 5 minutes", '[{"type":"function","function":{"name":"set_timer","description":"Set timer.","parameters":{"type":"object","properties":{"duration":{"type":"string"}},"required":["duration"]}}}]'),
]

for q, tools in tests:
    enc_text = f"Query: {q}\nTools: {tools}"
    enc_ids = torch.tensor([tokenizer.encode(enc_text).ids[:128]], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(enc_ids, max_length=80, temperature=0.7, top_p=0.9, top_k=50)
    resp = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    ok = "<tool_call>" in resp or '"name"' in resp
    print(f"\n   Q: {q}")
    print(f"   A: {resp[:300]}")
    print(f"   -> {'OK' if ok else 'NEEDS MORE TRAINING'}")

print("\n" + "="*50)
print("DONE")
print("="*50)
