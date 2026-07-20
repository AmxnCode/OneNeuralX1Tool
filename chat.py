"""
Interactive Chat - 26M Enhanced Model
Run in terminal to test tool calling.
"""

import torch
import sys
import json
from tokenizers import Tokenizer

sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')
from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2


def load_model():
    model = OneNeuralX1ToolV2(
        vocab_size=8192, hidden_size=420, num_encoder_layers=10,
        num_decoder_layers=5, num_attention_heads=6, num_key_value_heads=3,
        max_loops=3, num_experts=4, expert_dim=1280,
    )
    state = torch.load('checkpoints/x1_26m_final.pt', map_location='cpu')
    for k in list(state.keys()):
        if 'halting_state' in k: del state[k]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_tokenizer():
    """Load tokenizer - rebuild from training data since old one wasn't saved."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=8192, special_tokens=["[PAD]","[BOS]","[EOS]","[UNK]"], min_frequency=2)
    
    texts = []
    with open('data/multi_tool_chains.jsonl') as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["query"])
            texts.append(ex["answers"])
    tok.train_from_iterator(texts, trainer=tr)
    
    tok.enable_padding(pad_id=0, pad_token='[PAD]', length=256)
    tok.enable_truncation(max_length=256)
    return tok


def main():
    print("=" * 60)
    print("  One Neural X1 Tool - Interactive Chat")
    print("  26M params | 5.42 MB | Recurrent + MoE + ACT")
    print("=" * 60)
    print("\nLoading model...")
    
    model = load_model()
    tokenizer = load_tokenizer()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Ready! ({params/1e6:.2f}M params)\n")
    print("Type a query or 'quit' to exit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query or query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        # Tokenize
        enc = tokenizer.encode(query)
        input_ids = torch.tensor([enc.ids])
        
        # Generate
        with torch.no_grad():
            output = model.generate(input_ids, max_length=64, temperature=0.1, do_sample=False)
        
        # Decode
        ids = [t for t in output[0].tolist() if t > 2]
        response = tokenizer.decode(ids)
        
        print(f"X1:  {response}\n")


if __name__ == "__main__":
    main()
