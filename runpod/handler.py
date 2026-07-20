import runpod
import torch
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2

MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global MODEL, TOKENIZER

    model_path = os.environ.get("MODEL_PATH", "/model/x1_26m_final.pt")
    data_path = os.environ.get("DATA_PATH", "/model/combined_training.jsonl")

    data = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    TOOL_NAMES = ['search_web', 'get_weather', 'send_email', 'create_event',
                  'get_stock_price', 'set_reminder', 'play_music', 'calculate']
    JSON_TOKENS = ['{"name":', '{"arguments":', '"}', '},', '"', ':', '{', '}', '[', ']']
    SPECIAL_TOKENS = ['[PAD]', '[BOS]', '[EOS]', '[UNK]'] + TOOL_NAMES + JSON_TOKENS

    TOKENIZER = Tokenizer(models.BPE())
    TOKENIZER.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=8192, special_tokens=SPECIAL_TOKENS,
        min_frequency=2, continuing_subword_prefix=""
    )
    texts = []
    for item in data:
        texts.extend([item['query'], item['answers']])
    TOKENIZER.train_from_iterator(texts, trainer=trainer)
    TOKENIZER.enable_padding(pad_id=0, pad_token='[PAD]', length=256)
    TOKENIZER.enable_truncation(max_length=256)

    MODEL = OneNeuralX1ToolV2(
        vocab_size=8192, hidden_size=512, num_encoder_layers=12,
        num_decoder_layers=6, num_attention_heads=8, num_key_value_heads=4,
        max_loops=4, num_experts=4, expert_dim=2048
    )

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        MODEL.load_state_dict(ckpt, strict=False)

    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"Model loaded on {DEVICE}")

def run_inference(query, max_length=100, temperature=0.3):
    enc = TOKENIZER.encode(query)
    inp = torch.tensor([enc.ids]).to(DEVICE)

    with torch.no_grad():
        out = MODEL.generate(inp, max_length=max_length, temperature=temperature, do_sample=False)

    ids = [t for t in out[0].tolist() if t > 3]
    text = TOKENIZER.decode(ids)
    return text

def handler(event):
    job_input = event.get("input", {})
    query = job_input.get("query", "")
    max_length = job_input.get("max_length", 100)
    temperature = job_input.get("temperature", 0.3)

    if not query:
        return {"error": "No query provided"}

    try:
        result = run_inference(query, max_length, temperature)
        return {"output": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
