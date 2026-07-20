# One Neural X1 Tool - Local Training Guide

Train a function calling model entirely on your device. No API needed.

## Quick Start

```bash
# Step 1: Install dependencies
pip install torch transformers datasets tqdm

# Step 2: Generate training data (synthetic)
cd liquid_foundation_model/training
python generate_local_data.py --num-samples 10000 --output data/tool_calling.jsonl

# Step 3: Train the model
python train_local.py --data data/tool_calling.jsonl --epochs 10

# Step 4: Test the model
python -c "
from encoder_decoder_model import OneNeuralX1Tool
from config import X1ToolConfig
import torch

# Load model
config = X1ToolConfig.tiny()
model = OneNeuralX1Tool(**config.to_dict())
model.load_state_dict(torch.load('checkpoints/x1-tool-local/best_model.pt')['model'])
"
```

## What You Get

- **Tiny model**: ~9M parameters (runs on any device)
- **Small model**: ~45M parameters (needs ~1GB RAM)
- **Training time**: ~30 min on CPU, ~5 min on GPU

## Architecture

```
Encoder (12 layers)
├── GQA (8H/4KV)
├── RoPE
└── Gated residuals

Decoder (8 layers)
├── Self-attention (causal)
├── Cross-attention (to encoder)
└── Gated residuals

Vocab: 8,192
hidden_size: 512
No FFN (pure attention)
```

## Training Options

### Option 1: Synthetic Data (Fastest)
```bash
python generate_local_data.py --num-samples 50000 --output data/synthetic.jsonl
python train_local.py --data data/synthetic.jsonl --epochs 20
```

### Option 2: Real Datasets
```bash
# Download Gorilla dataset
python generate_local_data.py --source gorilla --output data/gorilla.jsonl

# Download ToolBench
python generate_local_data.py --source toolbench --output data/toolbench.jsonl

# Train on real data
python train_local.py --data data/gorilla.jsonl --epochs 10
```

### Option 3: Local Distillation (Best Quality)
```bash
# Generate data with small teacher model
python local_distill.py --teacher qwen2.5-0.5b --data data/tool_calling.jsonl

# Train with distilled data
python train_local.py --data data/distilled_responses.jsonl --epochs 10
```

## Available Teacher Models (Free)

| Model | Size | RAM | Quality |
|-------|------|-----|---------|
| smollm2-360m | 360M | 0.5GB | Basic |
| qwen2.5-0.5b | 0.5B | 1GB | Good |
| qwen2.5-1.5b | 1.5B | 2GB | Better |
| gemma-2b | 2B | 3GB | Good |
| phi-3-mini | 3.8B | 4GB | Best |

## Testing

```bash
# Run architecture test
python tests/test_x1_tool.py

# Test training
python train_local.py --data data/tool_calling.jsonl --epochs 1
```

## What's Next?

After local training:

1. **Evaluate** on ToolBench/API-Bank benchmarks
2. **Quantize** to INT4 for mobile deployment
3. **Deploy** with ExecuTorch (mobile) or llama.cpp (desktop)

## Files

```
liquid_foundation_model/
├── model/
│   ├── encoder_decoder_model.py  # Main model
│   ├── blocks/
│   │   ├── encoder_block.py
│   │   └── decoder_block.py
│   └── layers/
│       ├── rope.py
│       └── cross_attention.py
└── training/
    ├── generate_local_data.py    # Data generation
    ├── local_distill.py          # Local distillation
    ├── train_local.py            # Local training
    └── train_x1_tool.py          # Full training pipeline
```
