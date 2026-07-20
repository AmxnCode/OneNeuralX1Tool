#!/usr/bin/env python3
"""
Test script for HRM integration in LFM2 model.
"""

import torch
from liquid_foundation_model.model.configuration.config import LFMConfig
from liquid_foundation_model.model.liquid_foundation_model import LiquidFoundationModelForCausalLM

def test_hrm_integration():
    """Test the HRM integration with the LFM2 model."""
    
    # Create config with HRM enabled
    config = LFMConfig.from_pretrained("350M")
    config.enable_hrm = True
    config.hrm_reasoning_steps = 2  # Reduce for faster testing
    
    print("Creating model with HRM integration...")
    model = LiquidFoundationModelForCausalLM(config)
    
    # Print model info
    model_info = model.model.get_model_size_info()
    print(f"Model info: {model_info}")
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Test without HRM
    print("\nTesting without HRM...")
    with torch.no_grad():
        output_no_hrm = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_hrm=False
        )
    print(f"Output shape (no HRM): {output_no_hrm.shape}")
    
    # Test with HRM for general task
    print("\nTesting with HRM (general task)...")
    with torch.no_grad():
        output_general = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_hrm=True,
            task_type="general"
        )
    print(f"Output shape (general): {output_general.shape}")
    
    # Test with HRM for reasoning task
    print("\nTesting with HRM (reasoning task)...")
    with torch.no_grad():
        output_reasoning = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_hrm=True,
            task_type="reasoning"
        )
    print(f"Output shape (reasoning): {output_reasoning.shape}")
    
    # Test auto task detection
    print("\nTesting with HRM (auto task detection)...")
    with torch.no_grad():
        output_auto = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_type="auto"
        )
    print(f"Output shape (auto): {output_auto.shape}")
    
    print("\n✅ HRM integration test completed successfully!")
    
    # Compare outputs
    print(f"\nOutput differences:")
    print(f"No HRM vs General: {torch.mean(torch.abs(output_no_hrm - output_general)).item():.6f}")
    print(f"General vs Reasoning: {torch.mean(torch.abs(output_general - output_reasoning)).item():.6f}")
    print(f"Reasoning vs Auto: {torch.mean(torch.abs(output_reasoning - output_auto)).item():.6f}")

if __name__ == "__main__":
    test_hrm_integration()