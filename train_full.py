"""
Full Training - NeedleBonsai 27M
Run this on your GPU machine.
"""
import json, sys, time
sys.path.insert(0, '/OneNeuralX1Tool')

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from liquid_foundation_model.model.needle_bonsai import create_model


class FnCallingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc_ids = self.tokenizer.encode(
            f"Query: {ex.get('query','')}\nTools: {ex.get('tools','[]')}"
        ).ids[:self.max_len]
        dec_ids = [1] + self.tokenizer.encode(ex.get('answers','')).ids[:self.max_len-1]
        labels = dec_ids[1:] + [-100]
        enc_ids += [0]*max(0, self.max_len-len(enc_ids))
        dec_ids += [0]*max(0, self.max_len-len(dec_ids))
        labels += [-100]*max(0, self.max_len-len(labels))
        return {
            "encoder_input_ids": torch.tensor(enc_ids[:self.max_len], dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_ids[:self.max_len], dtype=torch.long),
            "labels": torch.tensor(labels[:self.max_len], dtype=torch.long),
        }


def main():
    print("="*60)
    print("NeedleBonsai Full Training")
    print("="*60)

    ckpt_dir = Path("/OneNeuralX1Tool/checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build tokenizer (SAVE IT)
    print("\n1. Building tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=["[PAD]","[BOS]","[EOS]","[UNK]"], min_frequency=2)
    texts = []
    with open("/OneNeuralX1Tool/data/needle_format.jsonl") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                texts.append(ex.get("query",""))
                texts.append(ex.get("answers",""))
                texts.append(ex.get("tools",""))
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
    tokenizer.enable_truncation(max_length=128)
    tokenizer.save(str(ckpt_dir / "tokenizer.json"))
    print(f"   Saved tokenizer (vocab={tokenizer.get_vocab_size()})")

    # 2. Model
    print("\n2. Creating model...")
    model = create_model().to(device)
    print(f"   {model.get_info()['parameters_millions']:.2f}M params")

    # 3. Data
    print("\n3. Loading data...")
    dataset = FnCallingDataset("/OneNeuralX1Tool/data/needle_format.jsonl", tokenizer)
    split = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=2, pin_memory=True)
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 4. Train
    print("\n4. Training 30 epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    best_val = float('inf')
    patience = 0

    for epoch in range(30):
        t0 = time.time()

        # Cosine LR
        progress = epoch / 30
        lr = 5e-4 * 0.5 * (1 + __import__('math').cos(progress * 3.14159))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        tloss = tn = 0
        for batch in train_loader:
            out = model(batch["encoder_input_ids"].to(device), batch["decoder_input_ids"].to(device), labels=batch["labels"].to(device))
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
                out = model(batch["encoder_input_ids"].to(device), batch["decoder_input_ids"].to(device), labels=batch["labels"].to(device))
                vloss += out["loss"].item()
                preds = out["logits"].argmax(dim=-1)
                mask = batch["labels"] != -100
                correct += (preds[mask] == batch["labels"][mask]).sum().item()
                total += mask.sum().item()

        vl = vloss/len(val_loader)
        dt = time.time()-t0
        print(f"Epoch {epoch+1:2d}/30: train={tloss/tn:.4f} val={vl:.4f} acc={correct/total:.4f} lr={lr:.6f} ({dt:.0f}s)")

        if vl < best_val:
            best_val = vl
            patience = 0
            torch.save(model.state_dict(), ckpt_dir / "needle_bonsai_best.pt")
            print(f"  -> Saved best (val_loss={best_val:.4f})")
        else:
            patience += 1
            if patience >= 7:
                print(f"  -> Early stop at epoch {epoch+1}")
                break

    # 5. Test
    print("\n" + "="*60)
    print("GENERATION TESTS")
    print("="*60)
    model.load_state_dict(torch.load(ckpt_dir / "needle_bonsai_best.pt", weights_only=True))
    model.eval()

    tests = [
        ("What is the weather in Paris?", '[{"type":"function","function":{"name":"get_weather","description":"Get weather.","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]'),
        ("Turn on bedroom lights", '[{"type":"function","function":{"name":"control_lights","description":"Control lights.","parameters":{"type":"object","properties":{"room":{"type":"string"},"action":{"type":"string"}},"required":["room","action"]}}}]'),
        ("Set a timer for 10 minutes", '[{"type":"function","function":{"name":"set_timer","description":"Set timer.","parameters":{"type":"object","properties":{"duration":{"type":"string"}},"required":["duration"]}}}]'),
    ]

    for q, tools in tests:
        enc_text = f"Query: {q}\nTools: {tools}"
        enc_ids = torch.tensor([tokenizer.encode(enc_text).ids[:128]], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model.generate(enc_ids, max_length=80, temperature=0.7, top_p=0.9, top_k=50)
        resp = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"\nQ: {q}")
        print(f"A: {resp[:300]}")

    print("\n" + "="*60)
    print("DONE")
    print(f"Checkpoints: {ckpt_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
