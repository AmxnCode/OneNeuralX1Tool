"""
Full 26M Parameter Enhanced X1 Tool Model
Same architecture as 7.3M, just scaled up to 26M.
"""

import torch
import sys
sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')

from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2


def create_26m_model():
    """Create 26M parameter model."""
    
    # Target 26M parameters
    model = OneNeuralX1ToolV2(
        vocab_size=8192,
        hidden_size=420,          # Tuned to hit 26M
        num_encoder_layers=10,    
        num_decoder_layers=5,     
        num_attention_heads=6,    
        num_key_value_heads=3,    
        max_loops=3,              
        num_experts=4,            
        expert_dim=1280,          
        dropout_rate=0.1,
        use_ternary=True,
        use_hybrid_attention=True,
        use_recurrent_decoder=True,
        use_moe=True,
    )
    
    return model


def main():
    print("=" * 60)
    print("Creating 26M Parameter Enhanced Model")
    print("=" * 60)
    
    model = create_26m_model()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,} ({params/1e6:.2f}M)")
    print(f"Memory (ternary): {params * 1.71 / (8 * 1024 * 1024):.2f} MB")
    
    # Test forward pass
    print("\nTesting forward pass...")
    enc = torch.randint(0, 8192, (2, 128))
    dec = torch.randint(0, 8192, (2, 64))
    labels = torch.randint(0, 8192, (2, 64))
    
    out = model(enc, dec, labels=labels)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    
    if 'num_loops_used' in out:
        print(f"Loops used: {out['num_loops_used']}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Name: {info['name']}")
    print(f"  Parameters: {info['parameters']:,}")
    print(f"  Memory: {info['memory_mb']:.2f} MB")
    print(f"  Features: {info['features']}")
    
    # Save config
    print("\n" + "=" * 60)
    print("Ready for training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
