import unittest
import torch
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from liquid_foundation_model.model.layers.grouped_query_attention import GroupedQueryAttention
from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.swiglu import SwiGLU
from liquid_foundation_model.model.blocks.attention_block import AttentionBlock


class TestAttentionBlocks(unittest.TestCase):
    """Test cases for attention block components."""
    
    def test_rmsnorm(self):
        """Test the RMSNorm layer."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        
        # Create a RMSNorm layer
        norm = RMSNorm(hidden_size=hidden_size)
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = norm(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
        # Check that the output is normalized
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))
    
    def test_swiglu(self):
        """Test the SwiGLU activation function."""
        batch_size = 2
        seq_len = 10
        in_features = 32
        hidden_features = 64
        out_features = 32
        
        # Create a SwiGLU layer
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
    
    def test_grouped_query_attention(self):
        """Test the Grouped Query Attention layer."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        num_attention_heads = 4
        num_key_value_heads = 2
        
        # Create a Grouped Query Attention layer
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
        
        # Test with attention mask
        attention_mask = torch.zeros(batch_size, 1, 1, seq_len)
        attention_mask[:, :, :, seq_len // 2:] = -10000.0  # Mask out the second half
        
        # Forward pass with mask
        output, _ = attention(x, attention_mask=attention_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
        # Test with output_attentions=True
        output, attention_weights = attention(x, output_attentions=True)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        self.assertEqual(attention_weights.shape, (batch_size, num_attention_heads, seq_len, seq_len))
    
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
        
        # Test with attention mask
        attention_mask = torch.zeros(batch_size, 1, 1, seq_len)
        attention_mask[:, :, :, seq_len // 2:] = -10000.0  # Mask out the second half
        
        # Forward pass with mask
        output = block(x, attention_mask=attention_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
        # Test with residual connection
        # The output should be different from the input
        self.assertFalse(torch.allclose(output, x))


if __name__ == "__main__":
    unittest.main()