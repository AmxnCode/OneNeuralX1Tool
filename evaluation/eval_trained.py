"""
Evaluation - 26M Trained Model
Tests tool calling accuracy with BPE tokenizer.
"""

import json
import torch
import time
from pathlib import Path
from tokenizers import Tokenizer

import sys
sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')

from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2


# Test cases with expected JSON output patterns
TEST_CASES = [
    # Single tool
    {"query": "What's the weather in Paris?", "expected": "get_weather", "args": ["Paris"]},
    {"query": "Get Apple stock price", "expected": "get_stock_price", "args": ["AAPL"]},
    {"query": "Calculate 15 * 3", "expected": "calculate", "args": ["15"]},
    {"query": "Set a reminder for tomorrow at 9am", "expected": "set_reminder", "args": []},
    {"query": "Send email to john@example.com", "expected": "send_email", "args": ["john"]},
    {"query": "Search for restaurants in Tokyo", "expected": "search_knowledge_base", "args": ["Tokyo"]},
    {"query": "Get directions to the airport", "expected": "get_directions", "args": ["airport"]},
    
    # Two-tool chains
    {"query": "Check weather in Tokyo and create an event", "expected": "get_weather", "args": ["Tokyo"]},
    {"query": "Find flights to Paris and email me", "expected": "search_flights", "args": ["Paris"]},
    {"query": "Search for coffee shops and get directions", "expected": "search_knowledge_base", "args": []},
    {"query": "Check stock price and set reminder", "expected": "get_stock_price", "args": []},
    
    # Three-tool chains
    {"query": "Check weather, find flights, add to calendar", "expected": "get_weather", "args": []},
    {"query": "Find restaurant, get directions, email them", "expected": "search_knowledge_base", "args": []},
]


def load_model(checkpoint_path):
    """Load trained model."""
    model = OneNeuralX1ToolV2(
        vocab_size=8192,
        hidden_size=420,
        num_encoder_layers=10,
        num_decoder_layers=5,
        num_attention_heads=6,
        num_key_value_heads=3,
        max_loops=3,
        num_experts=4,
        expert_dim=1280,
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Remove buffer mismatch
    for key in list(state_dict.keys()):
        if "halting_state" in key:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def load_tokenizer(data_path):
    """Load BPE tokenizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        min_frequency=2,
    )
    
    texts = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["query"])
            texts.append(ex["answers"])
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
    tokenizer.enable_truncation(max_length=256)
    
    return tokenizer


def generate_and_decode(model, tokenizer, query, max_length=64):
    """Generate response and decode."""
    # Tokenize query
    enc = tokenizer.encode(query)
    input_ids = torch.tensor([enc.ids])
    
    # Generate
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, temperature=0.1, do_sample=False)
    
    # Decode output
    output_ids = output[0].tolist()
    # Remove padding and special tokens
    output_ids = [t for t in output_ids if t > 2]  # Remove PAD, BOS, EOS
    
    try:
        decoded = tokenizer.decode(output_ids)
    except:
        decoded = str(output_ids[:50])
    
    return decoded, output_ids


def evaluate():
    print("=" * 60)
    print("Evaluating 26M Trained Model")
    print("=" * 60)
    
    # Load
    print("\n1. Loading model...")
    checkpoint_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/x1_26m_final.pt"
    model = load_model(checkpoint_path)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Load tokenizer
    print("\n2. Loading tokenizer...")
    data_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/data/multi_tool_chains.jsonl"
    tokenizer = load_tokenizer(data_path)
    print(f"   Vocab size: {tokenizer.get_vocab_size()}")
    
    # Evaluate
    print("\n3. Running tests...")
    results = []
    
    for i, test in enumerate(TEST_CASES):
        # Generate
        start = time.time()
        generated, token_ids = generate_and_decode(model, tokenizer, test["query"])
        latency = time.time() - start
        
        # Check if expected tool appears in output
        tool_found = test["expected"].lower() in generated.lower()
        
        # Check if any expected args appear
        args_found = sum(1 for arg in test["args"] if arg.lower() in generated.lower())
        args_score = args_found / len(test["args"]) if test["args"] else 1.0
        
        results.append({
            "query": test["query"],
            "expected_tool": test["expected"],
            "generated": generated[:100],
            "tool_correct": tool_found,
            "args_score": args_score,
            "latency": latency,
        })
        
        status = "✓" if tool_found else "✗"
        print(f"\n  {status} Query: {test['query']}")
        print(f"    Expected: {test['expected']}")
        print(f"    Generated: {generated[:80]}...")
        print(f"    Tool match: {tool_found}, Args: {args_score:.0%}, Latency: {latency:.3f}s")
    
    # Aggregate
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    tool_accuracy = sum(r["tool_correct"] for r in results) / len(results)
    avg_args = sum(r["args_score"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    print(f"\nTotal tests: {len(results)}")
    print(f"Tool selection accuracy: {tool_accuracy:.2%}")
    print(f"Argument accuracy: {avg_args:.2%}")
    print(f"Average latency: {avg_latency:.3f}s")
    print(f"Memory footprint: {params * 1.71 / (8*1024*1024):.2f} MB (ternary)")
    
    # Comparison
    print("\n" + "-" * 60)
    print("COMPARISON")
    print("-" * 60)
    print(f"{'Model':<25} {'Tool Acc':<12} {'Memory':<12}")
    print("-" * 49)
    print(f"{'Enhanced X1 (ours)':<25} {tool_accuracy:.2%}{'':<7} {params*1.71/(8*1024*1024):.2f} MB")
    print(f"{'Old 26M baseline':<25} {'72.00%':<12} {'100.0 MB':<12}")
    print(f"{'Claude Haiku':<25} {'95.00%':<12} {'2000+ MB':<12}")
    
    # Save
    output_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/eval_results.json"
    with open(output_path, 'w') as f:
        json.dump({"results": results, "summary": {
            "tool_accuracy": tool_accuracy,
            "args_accuracy": avg_args,
            "avg_latency": avg_latency,
            "memory_mb": params * 1.71 / (8*1024*1024),
        }}, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
