"""
Test script for One Neural X1 Tool architecture.

Verifies that the encoder-decoder model builds correctly and can do forward/backward passes.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liquid_foundation_model.model.configuration.config import X1ToolConfig
from liquid_foundation_model.model.encoder_decoder_model import OneNeuralX1Tool


def test_model_creation():
    """Test model creation with different configs."""
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    # Test tiny config
    print("\n1. Testing tiny config (~10M params)...")
    config = X1ToolConfig.tiny()
    model = OneNeuralX1Tool(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
    )
    param_counts = model.count_parameters()
    print(f"   Created model with {param_counts['total_millions']:.2f}M parameters")
    print(f"   Encoder: {param_counts['encoder_millions']:.2f}M")
    print(f"   Decoder: {param_counts['decoder_millions']:.2f}M")
    
    # Test small config (target: ~27M)
    print("\n2. Testing small config (~27M params)...")
    config = X1ToolConfig.small()
    model = OneNeuralX1Tool(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
    )
    param_counts = model.count_parameters()
    print(f"   Created model with {param_counts['total_millions']:.2f}M parameters")
    print(f"   Encoder: {param_counts['encoder_millions']:.2f}M")
    print(f"   Decoder: {param_counts['decoder_millions']:.2f}M")
    
    return model


def test_forward_pass(model):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    # Create dummy inputs
    batch_size = 2
    enc_seq_len = 64
    dec_seq_len = 32
    
    encoder_input_ids = torch.randint(0, 8192, (batch_size, enc_seq_len))
    decoder_input_ids = torch.randint(0, 8192, (batch_size, dec_seq_len))
    labels = torch.randint(0, 8192, (batch_size, dec_seq_len))
    
    print(f"\nEncoder input shape: {encoder_input_ids.shape}")
    print(f"Decoder input shape: {decoder_input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Forward pass
    outputs = model(
        encoder_input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
    )
    
    print(f"\nLogits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    return outputs


def test_generation(model):
    """Test generation."""
    print("\n" + "=" * 60)
    print("Testing Generation")
    print("=" * 60)
    
    # Create dummy encoder input
    batch_size = 1
    enc_seq_len = 64
    
    encoder_input_ids = torch.randint(0, 8192, (batch_size, enc_seq_len))
    
    print(f"\nEncoder input shape: {encoder_input_ids.shape}")
    
    # Generate
    generated_ids = model.generate(
        encoder_input_ids=encoder_input_ids,
        max_length=32,
        temperature=0.7,
        do_sample=True,
    )
    
    print(f"Generated shape: {generated_ids.shape}")
    print(f"Generated tokens: {generated_ids[0][:10].tolist()}...")
    
    return generated_ids


def test_encoder_output(model):
    """Test encoder output extraction."""
    print("\n" + "=" * 60)
    print("Testing Encoder Output")
    print("=" * 60)
    
    # Create dummy input
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(0, 8192, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Get encoder output
    encoder_output = model.get_encoder_output(input_ids)
    
    print(f"Encoder output shape: {encoder_output.shape}")
    
    return encoder_output


def test_backward_pass(model):
    """Test backward pass."""
    print("\n" + "=" * 60)
    print("Testing Backward Pass")
    print("=" * 60)
    
    # Create dummy inputs
    batch_size = 2
    enc_seq_len = 64
    dec_seq_len = 32
    
    encoder_input_ids = torch.randint(0, 8192, (batch_size, enc_seq_len))
    decoder_input_ids = torch.randint(0, 8192, (batch_size, dec_seq_len))
    labels = torch.randint(0, 8192, (batch_size, dec_seq_len))
    
    # Forward pass
    outputs = model(
        encoder_input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
    )
    
    loss = outputs["loss"]
    print(f"\nLoss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    print(f"Number of parameters with gradients: {len(grad_norms)}")
    print(f"Average gradient norm: {sum(grad_norms) / len(grad_norms):.4f}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("One Neural X1 Tool - Architecture Test")
    print("=" * 60)
    
    # Test model creation
    model = test_model_creation()
    
    # Test forward pass
    test_forward_pass(model)
    
    # Test generation
    test_generation(model)
    
    # Test encoder output
    test_encoder_output(model)
    
    # Test backward pass
    test_backward_pass(model)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
