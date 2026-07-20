import torch
import unittest
from liquid_foundation_model.model.configuration.config import LFMConfig
from liquid_foundation_model.model.liquid_foundation_model import (
    LiquidFoundationModel,
    LiquidFoundationModelForCausalLM,
)


class TestFullModel(unittest.TestCase):
    """Test cases for the full Liquid Foundation Model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a small model config for testing
        self.config = LFMConfig(
            model_size="small",
            num_layers=4,
            hidden_size=128,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            vocab_size=1000,
            conv_kernel_size=3,
            num_conv_blocks=2,
            num_attention_blocks=2,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
        )
        
        # Create a batch of input tokens
        self.batch_size = 2
        self.seq_length = 10
        self.input_ids = torch.randint(
            0, self.config.vocab_size, (self.batch_size, self.seq_length)
        )
        self.attention_mask = torch.ones_like(self.input_ids)
    
    def test_model_initialization(self):
        """Test model initialization for different sizes."""
        # Test small model
        small_config = LFMConfig.from_pretrained("small")
        small_model = LiquidFoundationModel(small_config)
        self.assertEqual(small_model.config.hidden_size, 512)
        
        # Test medium model
        medium_config = LFMConfig.from_pretrained("medium")
        medium_model = LiquidFoundationModel(medium_config)
        self.assertEqual(medium_model.config.hidden_size, 768)
        
        # Test large model
        large_config = LFMConfig.from_pretrained("large")
        large_model = LiquidFoundationModel(large_config)
        self.assertEqual(large_config.hidden_size, 1024)
    
    def test_model_forward(self):
        """Test forward pass of the base model."""
        model = LiquidFoundationModel(self.config)
        
        # Test without output_hidden_states
        outputs = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        )
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.config.hidden_size))
        
        # Test with output_hidden_states
        outputs, all_hidden_states = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            output_hidden_states=True,
        )
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.config.hidden_size))
        self.assertEqual(len(all_hidden_states), self.config.num_layers + 1)  # +1 for final output
        
        # Check that each hidden state has the correct shape
        for hidden_state in all_hidden_states:
            self.assertEqual(hidden_state.shape, (self.batch_size, self.seq_length, self.config.hidden_size))
    
    def test_causal_lm_forward(self):
        """Test forward pass of the causal language model."""
        model = LiquidFoundationModelForCausalLM(self.config)
        
        # Test without labels
        logits = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        )
        self.assertEqual(logits.shape, (self.batch_size, self.seq_length, self.config.vocab_size))
        
        # Test with labels
        labels = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        loss, logits = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=labels,
        )
        self.assertEqual(loss.shape, ())  # Scalar loss
        self.assertEqual(logits.shape, (self.batch_size, self.seq_length, self.config.vocab_size))
        
        # Test with output_hidden_states
        loss, logits, all_hidden_states = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        self.assertEqual(loss.shape, ())  # Scalar loss
        self.assertEqual(logits.shape, (self.batch_size, self.seq_length, self.config.vocab_size))
        self.assertEqual(len(all_hidden_states), self.config.num_layers + 1)  # +1 for final output
    
    def test_generation(self):
        """Test text generation."""
        model = LiquidFoundationModelForCausalLM(self.config)
        
        # Test greedy generation
        generated_ids = model.generate(
            input_ids=self.input_ids[:1, :5],  # Use only one example with shorter context
            max_length=15,
            do_sample=False,
        )
        self.assertEqual(generated_ids.shape[0], 1)  # Batch size
        self.assertEqual(generated_ids.shape[1], 15)  # Sequence length
        
        # Test sampling
        generated_ids = model.generate(
            input_ids=self.input_ids[:1, :5],
            max_length=15,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
        self.assertEqual(generated_ids.shape[0], 1)  # Batch size
        self.assertEqual(generated_ids.shape[1], 15)  # Sequence length
        
        # Test multiple sequences
        generated_ids = model.generate(
            input_ids=self.input_ids[:1, :5],
            max_length=15,
            num_return_sequences=3,
        )
        self.assertEqual(generated_ids.shape[0], 3)  # Batch size * num_return_sequences
        self.assertEqual(generated_ids.shape[1], 15)  # Sequence length
        
        # Test with return_dict_in_generate
        outputs = model.generate(
            input_ids=self.input_ids[:1, :5],
            max_length=15,
            do_sample=True,
            return_dict_in_generate=True,
        )
        self.assertTrue(isinstance(outputs, dict))
        self.assertEqual(outputs["sequences"].shape[0], 1)  # Batch size
        self.assertEqual(outputs["sequences"].shape[1], 15)  # Sequence length
    
    def test_block_integration(self):
        """Test integration of convolution and attention blocks."""
        model = LiquidFoundationModel(self.config)
        
        # Check that we have the correct number of blocks
        self.assertEqual(len(model.blocks), self.config.num_conv_blocks + self.config.num_attention_blocks)
        
        # Check block types
        from liquid_foundation_model.model.blocks.conv_block import ConvBlock
        from liquid_foundation_model.model.blocks.attention_block import AttentionBlock
        
        # First blocks should be convolution blocks
        for i in range(self.config.num_conv_blocks):
            self.assertIsInstance(model.blocks[i], ConvBlock)
        
        # Last blocks should be attention blocks
        for i in range(self.config.num_conv_blocks, len(model.blocks)):
            self.assertIsInstance(model.blocks[i], AttentionBlock)


if __name__ == "__main__":
    unittest.main()