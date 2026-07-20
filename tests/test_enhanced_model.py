"""
Test script for Enhanced X1 Tool Model
Verifies all innovations work together.
"""

import torch
import sys
sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')

from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2


def test_enhanced_model():
    """Test the enhanced model with all features."""
    
    print("=" * 60)
    print("Testing Enhanced X1 Tool Model")
    print("=" * 60)
    
    # Test 1: Model creation
    print("\n1. Creating model...")
    model = OneNeuralX1ToolV2(
        vocab_size=8192,
        hidden_size=512,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_loops=4,
        num_experts=4,
        expert_dim=2048,
        dropout_rate=0.1,
        use_ternary=True,
        use_hybrid_attention=True,
        use_recurrent_decoder=True,
        use_moe=True,
    )
    
    print(f"   ✓ Model created successfully")
    print(f"   Parameters: {model.num_parameters_millions:.2f}M")
    print(f"   Memory: {model.memory_mb:.2f} MB (ternary)")
    
    # Test 2: Forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    enc_seq_len = 128
    dec_seq_len = 64
    
    encoder_input_ids = torch.randint(0, 8192, (batch_size, enc_seq_len))
    decoder_input_ids = torch.randint(0, 8192, (batch_size, dec_seq_len))
    labels = torch.randint(0, 8192, (batch_size, dec_seq_len))
    
    with torch.no_grad():
        output = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
    
    print(f"   ✓ Forward pass successful")
    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Loss: {output['loss'].item():.4f}")
    
    # Test 3: Generation
    print("\n3. Testing generation...")
    with torch.no_grad():
        generated = model.generate(
            encoder_input_ids=encoder_input_ids[:1],  # Single example
            max_length=32,
            temperature=0.7,
            do_sample=False,  # Greedy for testing
        )
    
    print(f"   ✓ Generation successful")
    print(f"   Generated shape: {generated.shape}")
    print(f"   Generated tokens: {generated[0, :10].tolist()}...")
    
    # Test 4: Loop information
    print("\n4. Testing loop information...")
    if "num_loops_used" in output:
        print(f"   ✓ Loops used: {output['num_loops_used']}")
        print(f"   Loop probabilities: {len(output.get('loop_probs', []))} steps")
    
    # Test 5: Model info
    print("\n5. Model information...")
    info = model.get_model_info()
    print(f"   Name: {info['name']}")
    print(f"   Parameters: {info['parameters']:,}")
    print(f"   Memory: {info['memory_mb']:.2f} MB")
    print(f"   Features: {info['features']}")
    
    # Test 6: Compare with original
    print("\n6. Comparison with original model...")
    original_params = 26_000_000  # ~26M
    original_memory = 100.0  # ~100 MB (FP32)
    
    compression_ratio = original_memory / info['memory_mb']
    effective_capacity = info['parameters'] * 4  # MoE gives 4x capacity
    
    print(f"   Original: {original_params:,} params, {original_memory:.1f} MB")
    print(f"   Enhanced: {info['parameters']:,} params, {info['memory_mb']:.2f} MB")
    print(f"   Compression: {compression_ratio:.1f}x smaller")
    print(f"   Effective capacity: {effective_capacity:,} params (with MoE)")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    model = test_enhanced_model()
