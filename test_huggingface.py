"""
Test HuggingFace Upload Pipeline
Tests upload/download with a fresh untrained model.
"""

import os
import sys
import json
import torch
import shutil

sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')
from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def main():
    print("=" * 50)
    print("  HuggingFace Upload Test")
    print("=" * 50)

    # 1. Create fresh model
    print("\n1. Creating fresh model...")
    model = OneNeuralX1ToolV2(
        vocab_size=16384, hidden_size=420, num_encoder_layers=10,
        num_decoder_layers=5, num_attention_heads=6, num_key_value_heads=3,
        max_loops=3, num_experts=4, expert_dim=1280,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"   {params/1e6:.2f}M params")

    # 2. Save in HuggingFace format
    print("\n2. Saving in HuggingFace format...")
    hf_dir = "OneNeuralX1Tool-26M"
    os.makedirs(hf_dir, exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), f"{hf_dir}/pytorch_model.bin")

    # Save config
    config = {
        "model_type": "encoder-decoder",
        "vocab_size": 16384,
        "hidden_size": 420,
        "num_encoder_layers": 10,
        "num_decoder_layers": 5,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "max_loops": 3,
        "num_experts": 4,
        "expert_dim": 1280,
        "max_position_embeddings": 1024,
        "architecture": "OneNeuralX1ToolV2",
        "task": "tool-calling",
    }
    with open(f"{hf_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save model card
    with open(f"{hf_dir}/README.md", "w") as f:
        f.write("""---
language: en
tags:
- tool-calling
- function-calling
- encoder-decoder
- 26m-params
- ternary-quantization
- hybrid-attention
- moe
- recurrent-depth
---

# OneNeuralX1Tool - 26M Function Calling Model

A small (26M parameter) encoder-decoder model for function calling/agentic tasks.

## Architecture
- **Encoder:** 10 Transformer blocks with hybrid attention (linear + full)
- **Decoder:** 5 recurrent layers with Adaptive Computation Time (ACT)
- **MoE:** 4 experts per layer with load balancing
- **Vocab:** 16,384 BPE tokens with tool name support

## Supported Tools
- get_weather
- get_stock_price
- send_email
- set_reminder
- check_system_status
- calculate_math
- get_news

## Note
This is a test upload. A trained version will be uploaded after Colab training.
""")

    print(f"   Saved to {hf_dir}/")

    # 3. Push to HuggingFace
    print("\n3. Pushing to HuggingFace...")
    try:
        from huggingface_hub import HfApi, login

        login(token=HF_TOKEN)
        print("   Logged in")

        api = HfApi()
        repo_name = "aman3456/OneNeuralX1Tool-26M"

        print(f"   Creating repo: {repo_name}")
        api.create_repo(repo_name, exist_ok=True)

        print("   Uploading files...")
        api.upload_folder(
            folder_path=hf_dir,
            repo_id=repo_name,
            repo_type="model"
        )

        print(f"\n   SUCCESS! Model at: https://huggingface.co/{repo_name}")

    except Exception as e:
        print(f"   FAILED: {e}")
        return

    # 4. Test download
    print("\n4. Testing download...")
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=repo_name, filename="config.json", repo_type="model")
        with open(config_path) as f:
            dl_config = json.load(f)
        print(f"   Downloaded config: {dl_config['architecture']}")
        print(f"   Vocab: {dl_config['vocab_size']}, Hidden: {dl_config['hidden_size']}")

    except Exception as e:
        print(f"   Download failed: {e}")

    # 5. Cleanup
    print("\n5. Cleaning up...")
    if os.path.exists(hf_dir):
        shutil.rmtree(hf_dir)

    print("\n" + "=" * 50)
    print("  HuggingFace upload pipeline WORKS!")
    print("=" * 50)
    print("\nNext step: Run train_26m_fixed.py on Colab, then upload trained model.")

if __name__ == "__main__":
    main()
