import unittest
import torch
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from liquid_foundation_model.model.blocks.conv_blocks import (
    ConvBlock,
    MultiScaleConvBlock,
    DilatedConvBlock,
    GatedConvBlock,
    create_conv_block,
)


class TestConvBlocks(unittest.TestCase):
    """Test cases for convolution blocks."""
    
    def test_conv_block(self):
        """Test the basic convolution block."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        intermediate_size = 64
        
        # Create a convolution block
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
        
        # Test with different configurations
        block = ConvBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kernel_size=3,
            groups=4,
            use_bias=False,
            activation=torch.tanh,
        )
        output = block(x)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_multi_scale_conv_block(self):
        """Test the multi-scale convolution block."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        intermediate_size = 64
        
        # Create a multi-scale convolution block
        block = MultiScaleConvBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
        # Test with different configurations
        block = MultiScaleConvBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kernel_sizes=[2, 4, 6],
            use_bias=False,
        )
        output = block(x)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_dilated_conv_block(self):
        """Test the dilated convolution block."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        intermediate_size = 64
        
        # Create a dilated convolution block
        block = DilatedConvBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
        # Test with different configurations
        block = DilatedConvBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kernel_size=5,
            dilation_rates=[1, 3, 5],
            use_bias=False,
        )
        output = block(x)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_gated_conv_block(self):
        """Test the gated convolution block."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        intermediate_size = 64
        
        # Create a gated convolution block
        block = GatedConvBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
        # Test with different configurations
        block = GatedConvBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kernel_size=5,
            use_bias=False,
        )
        output = block(x)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_create_conv_block(self):
        """Test the create_conv_block factory function."""
        hidden_size = 32
        intermediate_size = 64
        
        # Test creating different types of convolution blocks
        blocks = [
            create_conv_block("basic", hidden_size, intermediate_size),
            create_conv_block("multi_scale", hidden_size, intermediate_size),
            create_conv_block("dilated", hidden_size, intermediate_size),
            create_conv_block("gated", hidden_size, intermediate_size),
        ]
        
        # Check block types
        self.assertIsInstance(blocks[0], ConvBlock)
        self.assertIsInstance(blocks[1], MultiScaleConvBlock)
        self.assertIsInstance(blocks[2], DilatedConvBlock)
        self.assertIsInstance(blocks[3], GatedConvBlock)
        
        # Test with invalid block type
        with self.assertRaises(ValueError):
            create_conv_block("invalid", hidden_size, intermediate_size)


if __name__ == "__main__":
    unittest.main()