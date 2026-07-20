import unittest
import torch
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from liquid_foundation_model.model.configuration.config import LFMConfig
from liquid_foundation_model.model.layers.liv_operator import LIVOperator, GatedLIVOperator, DoubleGatedLIVConv
from liquid_foundation_model.model.layers.grouped_query_attention import GroupedQueryAttention
from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.swiglu import SwiGLU
from liquid_foundation_model.model.blocks.conv_block import ConvBlock
from liquid_foundation_model.model.blocks.attention_block import AttentionBlock
from liquid_foundation_model.model.liquid_foundation_model import LiquidFoundationModel, LiquidFoundationModelForCausalLM


class TestModelComponents(unittest.TestCase):
    """Test cases for model components."""
    
    def test_liv_operator(self):
        """Test the LIV operator."""
        batch_size = 2
        seq_len = 10
        input_dim = 32
        output_dim = 64
        
        # Create a LIV operator
        liv = LIVOperator(input_dim=input_dim, output_dim=output_dim)
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = liv(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))
    
    def test_gated_liv_operator(self):
        """Test the Gated LIV operator."""
        batch_size = 2
        seq_len = 10
        input_dim = 32
        output_dim = 64
        
        # Create a Gated LIV operator
        gated_liv = GatedLIVOperator(input_dim=input_dim, output_dim=output_dim)
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = gated_liv(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))
    
    def test_double_gated_liv_conv(self):
        """Test the Double-gated LIV convolution."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        intermediate_size = 64
        
        # Create a Double-gated LIV convolution
        conv = DoubleGatedLIVConv(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = conv(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_grouped_query_attention(self):
        """Test the Grouped Query Attention."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        num_attention_heads = 4
        num_key_value_heads = 2
        
        # Create a Grouped Query Attention
        attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output, _ = attention(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_rmsnorm(self):
        """Test the RMSNorm."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        
        # Create a RMSNorm
        norm = RMSNorm(hidden_size=hidden_size)
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = norm(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_swiglu(self):
        """Test the SwiGLU."""
        batch_size = 2
        seq_len = 10
        in_features = 32
        hidden_features = 64
        out_features = 32
        
        # Create a SwiGLU
        swiglu = SwiGLU(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, in_features)
        
        # Forward pass
        output = swiglu(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, out_features))
    
    def test_conv_block(self):
        """Test the Convolution Block."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        intermediate_size = 64
        
        # Create a Convolution Block
        block = ConvBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_attention_block(self):
        """Test the Attention Block."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        intermediate_size = 64
        num_attention_heads = 4
        num_key_value_heads = 2
        
        # Create an Attention Block
        block = AttentionBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_liquid_foundation_model(self):
        """Test the Liquid Foundation Model."""
        batch_size = 2
        seq_len = 10
        
        # Create a small model configuration
        config = LFMConfig.from_pretrained("small")
        
        # Create the model
        model = LiquidFoundationModel(config)
        
        # Create a random input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, config.hidden_size))
    
    def test_liquid_foundation_model_for_causal_lm(self):
        """Test the Liquid Foundation Model for Causal LM."""
        batch_size = 2
        seq_len = 10
        
        # Create a small model configuration
        config = LFMConfig.from_pretrained("small")
        
        # Create the model
        model = LiquidFoundationModelForCausalLM(config)
        
        # Create a random input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, config.vocab_size))
        
        # Test generation
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=seq_len + 5,
        )
        
        # Check output shape
        self.assertEqual(generated_ids.shape[0], batch_size)
        self.assertTrue(generated_ids.shape[1] > seq_len)


if __name__ == "__main__":
    unittest.main()