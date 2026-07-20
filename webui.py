"""
OneNeuralX1Tool - Training & Chat Web UI
Run: python webui.py
"""

import gradio as gr
import torch
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============ Global state ============
MODEL = None
TOKENIZER = None
TRAINING = False
TRAIN_LOG = []
BEST_ACC = 0.0
CURRENT_DATA_PATH = None
TRAIN_HISTORY = {"epoch": [], "loss": [], "acc": [], "lr": [], "time": []}

NEEDLE_DATA = "/Users/amanpreetsingh/projects/experiment/needle/data/needle_tools.jsonl"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ============ Model loading ============
def load_model():
    global MODEL, TOKENIZER
    from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    ckpt_path = os.path.join(CHECKPOINT_DIR, "x1_26m_final.pt")
    if not os.path.exists(ckpt_path):
        print("No checkpoint found, using fresh model")
        ckpt_path = None

    MODEL = OneNeuralX1ToolV2(
        vocab_size=8192, hidden_size=512,
        num_encoder_layers=12, num_decoder_layers=6,
        num_attention_heads=8, num_key_value_heads=4,
        max_loops=4, num_experts=4, expert_dim=2048,
    )

    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        sd.pop("decoder.act_halting.halting_state", None)
        missing, _ = MODEL.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint (missing: {len(missing)})")

    MODEL.eval()

    data_path = CURRENT_DATA_PATH or NEEDLE_DATA
    if os.path.exists(data_path):
        data = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        TOKENIZER = Tokenizer(models.BPE())
        TOKENIZER.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tool_names = ["search_web", "get_weather", "send_email", "create_event",
                      "get_stock_price", "set_reminder", "play_music", "calculate"]
        json_tokens = ['{"name":', '{"arguments":', '"}', '},', '"', ":", "{", "}", "[", "]"]
        special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + tool_names + json_tokens
        tr = trainers.BpeTrainer(vocab_size=8192, special_tokens=special_tokens,
                                 min_frequency=2, continuing_subword_prefix="")
        texts = []
        for item in data:
            texts.extend([item["query"], item["answers"]])
        TOKENIZER.train_from_iterator(texts, trainer=tr)
        TOKENIZER.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
        TOKENIZER.enable_truncation(max_length=256)
        print(f"Tokenizer vocab: {TOKENIZER.get_vocab_size()}")


# ============ Chat ============
def chat(message, history):
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        return "Model not loaded yet. Click 'Load Model' first."

    enc = TOKENIZER.encode(message)
    inp = torch.tensor([enc.ids])

    with torch.no_grad():
        out = MODEL.generate(inp, max_length=80, temperature=0.3, do_sample=False)

    ids = [t for t in out[0].tolist() if t > 2]
    response = TOKENIZER.decode(ids) if ids else "(no output)"
    return response


# ============ Dataset management ============
def list_datasets():
    datasets = []
    if os.path.exists(NEEDLE_DATA):
        count = sum(1 for _ in open(NEEDLE_DATA))
        datasets.append((f"Needle Tool Calls ({count} examples)", NEEDLE_DATA))
    for f in sorted(Path(DATA_DIR).glob("*.jsonl")):
        count = sum(1 for _ in open(f))
        datasets.append((f"{f.stem} ({count} examples)", str(f)))
    for f in sorted(Path(".").glob("*.jsonl")):
        count = sum(1 for _ in open(f))
        datasets.append((f"{f.name} ({count} examples)", str(f)))
    return datasets


def refresh_datasets():
    ds_list = list_datasets()
    if not ds_list:
        return gr.update(choices=["No datasets found"], value="No datasets found")
    choices = [d[0] for d in ds_list]
    default = choices[0]
    for c in choices:
        if "combined" in c.lower() or "generate" in c.lower():
            default = c
            break
    return gr.update(choices=choices, value=default)


def upload_dataset(file):
    if file is None:
        return "No file uploaded", refresh_datasets()
    dest = os.path.join(DATA_DIR, os.path.basename(file.name))
    import shutil
    shutil.copy(file.name, dest)
    count = sum(1 for _ in open(dest))
    return f"Uploaded {count} examples to {dest}", refresh_datasets()


def generate_sample_data(num_samples=500):
    import random
    random.seed(42)

    TOOLS = {
        "get_weather": {"description": "Get current weather", "parameters": {"location": "City name"}},
        "send_email": {"description": "Send an email", "parameters": {"to": "email", "subject": "Subject", "body": "Body"}},
        "create_event": {"description": "Create calendar event", "parameters": {"title": "Title", "date": "YYYY-MM-DD", "time": "HH:MM"}},
        "get_stock_price": {"description": "Get stock price", "parameters": {"symbol": "Ticker"}},
        "set_reminder": {"description": "Set a reminder", "parameters": {"message": "Message", "datetime": "datetime"}},
        "calculate": {"description": "Calculate math", "parameters": {"expression": "Expression"}},
        "search_web": {"description": "Search the web", "parameters": {"query": "Search query"}},
        "play_music": {"description": "Play music", "parameters": {"query": "Song or artist"}},
    }

    locations = ["Tokyo", "Paris", "New York", "London", "Sydney", "Berlin", "Toronto", "Mumbai", "Dubai", "Singapore"]
    stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
    emails = ["user@example.com", "john@work.com", "sarah@company.com"]
    songs = ["jazz", "rock", "classical", "hip hop", "lo-fi", "acoustic", "pop", "blues"]
    topics = ["climate change", "machine learning", "best restaurants", "python tutorial", "news today"]
    reminders = ["buy groceries", "call mom", "check email", "submit report", "book flight"]
    exprs = ["2+2", "15*3", "100/4", "50-25", "2**10", "7*8", "200/5"]

    def gen_weather():
        loc = random.choice(locations)
        q = random.choice([f"What's the weather in {loc}?", f"How's the weather in {loc}?",
                           f"Check weather for {loc}", f"Is it raining in {loc}?"])
        return q, {"location": loc}

    def gen_email():
        to = random.choice(emails)
        return random.choice([f"Send email to {to}", f"Email {to}", f"Write to {to}"]), {"to": to, "subject": "Hello", "body": "How are you?"}

    def gen_event():
        title = random.choice(["Meeting", "Lunch", "Workout", "Call", "Review", "Demo"])
        return random.choice([f"Create a {title.lower()} event", f"Schedule {title.lower()}", f"Add {title.lower()} to calendar"]), {"title": title, "date": "2024-03-20", "time": "14:00"}

    def gen_stock():
        stock = random.choice(stocks)
        return random.choice([f"Check {stock} stock price", f"What's {stock} trading at?", f"Get {stock} price"]), {"symbol": stock}

    def gen_reminder():
        msg = random.choice(reminders)
        return random.choice([f"Remind me to {msg}", f"Set reminder: {msg}", f"Don't forget {msg}"]), {"message": msg, "datetime": "2024-03-21 09:00"}

    def gen_calc():
        expr = random.choice(exprs)
        return random.choice([f"Calculate {expr}", f"What is {expr}?", f"Compute {expr}"]), {"expression": expr}

    def gen_search():
        topic = random.choice(topics)
        return random.choice([f"Search for {topic}", f"Look up {topic}", f"Find info on {topic}"]), {"query": topic}

    def gen_music():
        song = random.choice(songs)
        return random.choice([f"Play {song} music", f"Put on some {song}", f"Listen to {song}"]), {"query": song}

    GENFNS = [gen_weather, gen_email, gen_event, gen_stock, gen_reminder, gen_calc, gen_search, gen_music]
    TOOL_NAMES = list(TOOLS.keys())
    all_tools_list = [{"name": n, "description": t["description"], "parameters": t["parameters"]} for n, t in TOOLS.items()]

    examples = []
    for _ in range(int(num_samples)):
        idx = random.randint(0, len(GENFNS) - 1)
        query, args = GENFNS[idx]()
        examples.append({
            "query": query,
            "tools": json.dumps(all_tools_list),
            "answers": json.dumps([{"name": TOOL_NAMES[idx], "arguments": args}])
        })

    out_path = os.path.join(DATA_DIR, "generated_data.jsonl")
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return f"Generated {len(examples)} examples -> {out_path}", refresh_datasets()


# ============ Plot ============
def make_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if not TRAIN_HISTORY["epoch"]:
        ax1.text(0.5, 0.5, "No training data yet", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No training data yet", ha="center", va="center", transform=ax2.transAxes)
        plt.tight_layout()
        return fig

    epochs = TRAIN_HISTORY["epoch"]
    loss = TRAIN_HISTORY["loss"]
    acc = TRAIN_HISTORY["acc"]

    ax1.plot(epochs, loss, "b-o", linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f9fa")

    ax2.plot(epochs, acc, "g-o", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f8f9fa")
    ax2.set_ylim(0, 1.05)

    if acc:
        best_epoch = epochs[acc.index(max(acc))]
        best_acc = max(acc)
        ax2.axhline(y=best_acc, color="r", linestyle="--", alpha=0.5)
        ax2.annotate(f"Best: {best_acc:.4f}\n(Epoch {best_epoch})", xy=(best_epoch, best_acc),
                     xytext=(best_epoch + 1, best_acc - 0.1),
                     arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")

    plt.tight_layout()
    plt.close(fig)
    return fig


# ============ Training ============
def start_training(dataset_name, epochs, lr, batch_size, max_samples, progress=gr.Progress()):
    global MODEL, TOKENIZER, TRAINING, TRAIN_LOG, BEST_ACC, CURRENT_DATA_PATH, TRAIN_HISTORY

    if TRAINING:
        return "Training already in progress!", "\n".join(TRAIN_LOG), make_plot()

    TRAINING = True
    TRAIN_LOG = []
    BEST_ACC = 0.0
    TRAIN_HISTORY = {"epoch": [], "loss": [], "acc": [], "lr": [], "time": []}

    ds_list = list_datasets()
    data_path = None
    for name, path in ds_list:
        if name == dataset_name:
            data_path = path
            break

    if not data_path or not os.path.exists(data_path):
        TRAINING = False
        return f"Dataset not found: {dataset_name}", "\n".join(TRAIN_LOG), make_plot()

    CURRENT_DATA_PATH = data_path
    TRAIN_LOG.append(f"Loading data from: {data_path}")

    data = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    TRAIN_LOG.append(f"Loaded {len(data)} examples")
    max_samples = int(max_samples)
    if max_samples > 0 and len(data) > max_samples:
        import random
        random.seed(42)
        data = random.sample(data, max_samples)
        TRAIN_LOG.append(f"Using {max_samples} samples (random subset)")
    print(f"TRAIN: {len(data)} examples, {epochs} epochs, lr={lr}, bs={batch_size}")

    if len(data) < 10:
        TRAINING = False
        return "Need at least 10 examples!", "\n".join(TRAIN_LOG), make_plot()

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tool_names = ["search_web", "get_weather", "send_email", "create_event",
                  "get_stock_price", "set_reminder", "play_music", "calculate"]
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
    vocab_size = tokenizer.get_vocab_size()
    TRAIN_LOG.append(f"Tokenizer vocab: {vocab_size}")

    from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2

    model = OneNeuralX1ToolV2(
        vocab_size=8192, hidden_size=512,
        num_encoder_layers=12, num_decoder_layers=6,
        num_attention_heads=8, num_key_value_heads=4,
        max_loops=4, num_experts=4, expert_dim=2048,
    ).to("cpu")
    TRAIN_LOG.append("Training from scratch (new tokenizer, no old checkpoint)")

    params = sum(p.numel() for p in model.parameters())
    TRAIN_LOG.append(f"Model: {params/1e6:.2f}M params")

    from torch.utils.data import Dataset, DataLoader

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

    split = int(0.9 * len(data))
    train_loader = DataLoader(DS(data[:split]), batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(DS(data[split:]), batch_size=int(batch_size))
    TRAIN_LOG.append(f"Train: {split}, Val: {len(data) - split}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    best_acc = 0.0
    t0 = time.time()
    epochs = int(epochs)

    TRAIN_LOG.append(f"Starting {epochs} epochs training...")
    yield "Training in progress...", "\n".join(TRAIN_LOG), make_plot()

    for epoch in range(epochs):
        if not TRAINING:
            TRAIN_LOG.append("Training stopped by user")
            break

        print(f"Epoch {epoch+1}/{epochs}...", flush=True)
        model.train()
        total_loss = 0
        batch_count = 0
        for q_batch, inp_batch, lbl_batch in train_loader:
            out = model(q_batch, inp_batch, labels=lbl_batch)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for q_batch, inp_batch, lbl_batch in val_loader:
                out = model(q_batch, inp_batch, labels=lbl_batch)
                preds = out["logits"].argmax(-1)
                mask = lbl_batch != -100
                correct += (preds[mask] == lbl_batch[mask]).sum().item()
                total += mask.sum().item()

        val_acc = correct / total if total > 0 else 0
        elapsed = (time.time() - t0) / 60
        current_lr = optimizer.param_groups[0]["lr"]

        TRAIN_HISTORY["epoch"].append(epoch + 1)
        TRAIN_HISTORY["loss"].append(avg_loss)
        TRAIN_HISTORY["acc"].append(val_acc)
        TRAIN_HISTORY["lr"].append(current_lr)
        TRAIN_HISTORY["time"].append(elapsed)

        line = f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.2e} | {elapsed:.1f}min"
        TRAIN_LOG.append(line)
        print(line, flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "x1_26m_final.pt"))
            TRAIN_LOG.append(f"  -> Saved best (acc: {val_acc:.4f})")

        progress((epoch + 1) / epochs)
        yield f"Epoch {epoch+1}/{epochs} | Acc: {val_acc:.4f} | Best: {best_acc:.4f}", "\n".join(TRAIN_LOG), make_plot()

    elapsed = (time.time() - t0) / 60
    TRAIN_LOG.append(f"\nDone! Best acc: {best_acc:.4f} | Total: {elapsed:.1f}min")
    print(f"\nDone! Best acc: {best_acc:.4f} | Total: {elapsed:.1f}min", flush=True)
    TRAINING = False

    load_model()

    return f"Training complete! Best accuracy: {best_acc:.4f}", "\n".join(TRAIN_LOG), make_plot()


def stop_training():
    global TRAINING
    TRAINING = False
    return "Stopping training..."


# ============ Build UI ============
def build_ui():
    with gr.Blocks(title="OneNeuralX1Tool - Train & Chat", theme=gr.themes.Soft()) as app:
        gr.Markdown("# OneNeuralX1Tool - Function Calling Model")
        gr.Markdown("Small model that knows WHERE to find knowledge and HOW to apply it.")

        with gr.Tabs():
            # ===== Chat Tab =====
            with gr.Tab("Chat"):
                gr.Markdown("Chat with the trained model")
                gr.ChatInterface(
                    fn=chat,
                    examples=[
                        "What's the weather in Tokyo?",
                        "Check AAPL stock price",
                        "Send email to john@example.com",
                        "Calculate 15 * 3",
                        "Set a reminder for tomorrow",
                        "Search for machine learning",
                        "Play some jazz music",
                        "Create a meeting event",
                    ],
                    cache_examples=False,
                )

            # ===== Training Tab =====
            with gr.Tab("Train"):
                gr.Markdown("## Training Controls")

                with gr.Row():
                    with gr.Column(scale=2):
                        dataset_dropdown = gr.Dropdown(
                            label="Training Dataset",
                            choices=[d[0] for d in list_datasets()],
                            value=list_datasets()[0][0] if list_datasets() else None,
                        )
                        with gr.Row():
                            refresh_btn = gr.Button("Refresh", size="sm")
                            upload_file = gr.File(label="Upload JSONL", file_types=[".jsonl", ".json"])
                            upload_btn = gr.Button("Upload", size="sm")
                        upload_status = gr.Textbox(label="", interactive=False, visible=False)

                    with gr.Column(scale=1):
                        epochs_input = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Epochs")
                        lr_input = gr.Textbox(label="Learning Rate", value="5e-5")
                        batch_input = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Batch Size")
                        max_samples_input = gr.Slider(minimum=0, maximum=10000, value=1000, step=100, label="Max Samples (0=all)")

                with gr.Row():
                    train_btn = gr.Button("Start Training", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop Training", variant="stop", size="lg")

                train_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    train_log = gr.Textbox(label="Training Log", lines=15, interactive=False, scale=1)
                    train_plot = gr.Plot(label="Loss & Accuracy", scale=1)

                # Generate data
                with gr.Accordion("Generate Sample Data", open=False):
                    with gr.Row():
                        num_samples = gr.Slider(minimum=100, maximum=10000, value=5000, step=100, label="Samples")
                        gen_btn = gr.Button("Generate", size="sm")
                        gen_status = gr.Textbox(label="", interactive=False, visible=False)

                # Wire events
                refresh_btn.click(fn=refresh_datasets, outputs=dataset_dropdown)
                upload_btn.click(fn=upload_dataset, inputs=upload_file, outputs=[upload_status, dataset_dropdown])
                gen_btn.click(fn=generate_sample_data, inputs=num_samples, outputs=[gen_status, dataset_dropdown])
                train_btn.click(
                    fn=start_training,
                    inputs=[dataset_dropdown, epochs_input, lr_input, batch_input, max_samples_input],
                    outputs=[train_status, train_log, train_plot],
                )
                stop_btn.click(fn=stop_training, outputs=train_status)

            # ===== Model Info Tab =====
            with gr.Tab("Model Info"):
                gr.Markdown("""
                ## Architecture
                - **Encoder:** 12 layers hybrid attention (75% linear O(n), 25% full O(n2))
                - **Decoder:** Recurrent (4 loops) with ACT halting
                - **MoE:** 4 experts + shared expert
                - **Vocab:** 8,192 BPE

                ## Innovations from
                - **Ternary Bonsai 27B:** Hybrid attention
                - **OpenMythos:** Recurrent decoder, ACT halting, MoE

                ## Supported Tools
                search_web, get_weather, send_email, create_event,
                get_stock_price, set_reminder, play_music, calculate

                ## Philosophy
                Like humans: we don't store all knowledge -- we know WHERE to find it and HOW to apply it.
                This small model runs on phones and delegates to APIs for actual knowledge.
                """)

        return app


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("Starting Web UI...")
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
