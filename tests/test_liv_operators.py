import unittest
import torch
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from liquid_foundation_model.model.layers.liv_operator import (
    LIVOperator,
    GatedLIVOperator,
    DoubleGatedLIVConv,
    AdaptiveLIVOperator,
)
from liquid_foundation_model.model.layers.liv_utils import (
    create_liv_operator,
    LIVResidualBlock,
    LIVSequential,
)


class TestLIVOperators(unittest.TestCase):
    """Test cases for LIV operators."""
    
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
        
        # Test with different configurations
        liv = LIVOperator(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_generator_dim=16,
            bias=False,
            use_sequence_mean=False,
            efficient_forward=False,
        )
        output = liv(x)
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
        
        # Test with different configurations
        gated_liv = GatedLIVOperator(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_generator_dim=16,
            bias=False,
            gate_activation=torch.tanh,
            use_sequence_mean=False,
        )
        output = gated_liv(x)
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
        
        # Test with different configurations
        conv = DoubleGatedLIVConv(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kernel_size=3,
            groups=4,
            use_bias=False,
            activation=torch.tanh,
        )
        output = conv(x)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_adaptive_liv_operator(self):
        """Test the Adaptive LIV operator."""
        batch_size = 2
        seq_len = 10
        input_dim = 32
        output_dim = 64
        
        # Create an Adaptive LIV operator
        adaptive_liv = AdaptiveLIVOperator(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = adaptive_liv(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))
        
        # Test with different configurations
        adaptive_liv = AdaptiveLIVOperator(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=16,
            num_modes=2,
            use_bias=False,
        )
        output = adaptive_liv(x)
        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))
    
    def test_create_liv_operator(self):
        """Test the create_liv_operator factory function."""
        input_dim = 32
        output_dim = 64
        
        # Test creating different types of LIV operators
        operators = [
            create_liv_operator("basic", input_dim, output_dim),
            create_liv_operator("gated", input_dim, output_dim),
            create_liv_operator("double_gated_conv", input_dim, output_dim),
            create_liv_operator("adaptive", input_dim, output_dim),
        ]
        
        # Check operator types
        self.assertIsInstance(operators[0], LIVOperator)
        self.assertIsInstance(operators[1], GatedLIVOperator)
        self.assertIsInstance(operators[2], DoubleGatedLIVConv)
        self.assertIsInstance(operators[3], AdaptiveLIVOperator)
        
        # Test with invalid operator type
        with self.assertRaises(ValueError):
            create_liv_operator("invalid", input_dim, output_dim)
    
    def test_liv_residual_block(self):
        """Test the LIV residual block."""
        batch_size = 2
        seq_len = 10
        input_dim = 32
        
        # Create a LIV residual block
        block = LIVResidualBlock(input_dim=input_dim)
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, input_dim))
        
        # Test with different configurations
        block = LIVResidualBlock(
            input_dim=input_dim,
            hidden_dim=64,
            dropout_rate=0.2,
            liv_operator_type="adaptive",
            layer_norm=False,
        )
        output = block(x)
        self.assertEqual(output.shape, (batch_size, seq_len, input_dim))
    
    def test_liv_sequential(self):
        """Test the LIV sequential container."""
        batch_size = 2
        seq_len = 10
        input_dim = 32
        hidden_dims = [64, 128, 64]
        output_dim = 32
        
        # Create a LIV sequential container
        sequential = LIVSequential(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
        
        # Create a random input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = sequential(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))
        
        # Test with different configurations
        sequential = LIVSequential(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            liv_operator_types=["basic", "gated", "adaptive"],
            dropout_rate=0.2,
            residual=False,
            layer_norm=False,
        )
        output = sequential(x)
        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))
        
        # Test with invalid configuration
        with self.assertRaises(ValueError):
            LIVSequential(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                liv_operator_types=["basic", "gated"],  # Too few operator types
            )


if __name__ == "__main__":
    unittest.main()