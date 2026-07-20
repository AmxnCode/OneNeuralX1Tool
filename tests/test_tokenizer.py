"""Unit tests for the LFM tokenizer."""

import os
import tempfile
import unittest
from typing import List

import pytest
import torch

from liquid_foundation_model.training.data.tokenizer.tokenizer import LFMTokenizer


class TestLFMTokenizer(unittest.TestCase):
    """Test cases for the LFM tokenizer."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a common pretrained tokenizer for testing
        self.pretrained_tokenizer_name = "gpt2"
        
        # Create a temporary directory for saving tokenizers
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample texts for testing
        self.sample_texts = [
            "Hello, world!",
            "This is a test of the LFM tokenizer.",
            "It should handle various inputs correctly.",
            "Including special characters: !@#$%^&*()",
            "And numbers: 1234567890",
        ]
    
    def test_load_pretrained_tokenizer(self):
        """Test loading a pretrained tokenizer."""
        tokenizer = LFMTokenizer(from_pretrained=self.pretrained_tokenizer_name)
        
        # Check that the tokenizer was loaded correctly
        self.assertIsNotNone(tokenizer._tokenizer)
        self.assertGreater(len(tokenizer), 0)
        
        # Check special tokens
        self.assertIsNotNone(tokenizer.bos_token_id)
        self.assertIsNotNone(tokenizer.eos_token_id)
        self.assertIsNotNone(tokenizer.pad_token_id)
    
    def test_encode_decode_single(self):
        """Test encoding and decoding a single text."""
        tokenizer = LFMTokenizer(from_pretrained=self.pretrained_tokenizer_name)
        
        # Test with a single text
        text = "Hello, world!"
        token_ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(token_ids)
        
        # The decoded text might not match exactly due to tokenization artifacts
        # but it should contain the original text (ignoring case and some punctuation)
        self.assertIn("hello", decoded_text.lower())
        self.assertIn("world", decoded_text.lower())
    
    def test_encode_decode_batch(self):
        """Test encoding and decoding a batch of texts."""
        tokenizer = LFMTokenizer(from_pretrained=self.pretrained_tokenizer_name)
        
        # Test with a batch of texts
        texts = self.sample_texts
        token_ids = tokenizer.encode(texts, padding=True, truncation=True)
        decoded_texts = tokenizer.decode(token_ids)
        
        # Check that we get the same number of decoded texts
        self.assertEqual(len(texts), len(decoded_texts))
        
        # Check that each decoded text contains key parts of the original
        for i, (original, decoded) in enumerate(zip(texts, decoded_texts)):
            # Extract a key word from each original text
            key_word = original.split()[1].lower().strip(".,!?")
            self.assertIn(key_word, decoded.lower())
    
    def test_encode_with_tensors(self):
        """Test encoding with tensor output."""
        tokenizer = LFMTokenizer(from_pretrained=self.pretrained_tokenizer_name)
        
        # Test with a single text
        text = "Hello, world!"
        token_ids = tokenizer.encode(text, return_tensors="pt")
        
        # Check that we get a tensor
        self.assertIsInstance(token_ids, torch.Tensor)
        
        # Test with a batch of texts
        texts = self.sample_texts
        token_ids = tokenizer.encode(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Check that we get a tensor with the right shape
        self.assertIsInstance(token_ids, torch.Tensor)
        self.assertEqual(token_ids.shape[0], len(texts))
    
    def test_save_and_load(self):
        """Test saving and loading a tokenizer."""
        # Load a pretrained tokenizer
        tokenizer = LFMTokenizer(from_pretrained=self.pretrained_tokenizer_name)
        
        # Save it to a temporary directory
        save_path = os.path.join(self.temp_dir, "test_tokenizer")
        tokenizer.save_pretrained(save_path)
        
        # Load it back
        loaded_tokenizer = LFMTokenizer(from_pretrained=save_path)
        
        # Check that the vocabulary sizes match
        self.assertEqual(len(tokenizer), len(loaded_tokenizer))
        
        # Check that encoding produces the same results
        text = "Hello, world!"
        original_ids = tokenizer.encode(text)
        loaded_ids = loaded_tokenizer.encode(text)
        self.assertEqual(original_ids, loaded_ids)
    
    def test_adapt_tokenizer(self):
        """Test adapting a pretrained tokenizer."""
        # This is a simplified test since we can't easily create training files
        tokenizer = LFMTokenizer()
        
        # Adapt from a pretrained tokenizer
        save_path = os.path.join(self.temp_dir, "adapted_tokenizer")
        tokenizer.adapt_from_pretrained(self.pretrained_tokenizer_name, save_path)
        
        # Check that the tokenizer was saved
        self.assertTrue(os.path.exists(save_path))
        
        # Load it back
        loaded_tokenizer = LFMTokenizer(from_pretrained=save_path)
        
        # Check that it works
        text = "Hello, world!"
        token_ids = loaded_tokenizer.encode(text)
        decoded_text = loaded_tokenizer.decode(token_ids)
        self.assertIn("hello", decoded_text.lower())
    
    def test_vocab_access(self):
        """Test accessing the vocabulary."""
        tokenizer = LFMTokenizer(from_pretrained=self.pretrained_tokenizer_name)
        
        # Get the vocabulary
        vocab = tokenizer.get_vocab()
        
        # Check that it's a dictionary
        self.assertIsInstance(vocab, dict)
        
        # Check that it has entries
        self.assertGreater(len(vocab), 0)
        
        # Check that the property works too
        self.assertEqual(vocab, tokenizer.vocab)
    
    def test_special_token_ids(self):
        """Test accessing special token IDs."""
        tokenizer = LFMTokenizer(from_pretrained=self.pretrained_tokenizer_name)
        
        # Check that we can access special token IDs
        self.assertIsNotNone(tokenizer.bos_token_id)
        self.assertIsNotNone(tokenizer.eos_token_id)
        self.assertIsNotNone(tokenizer.pad_token_id)
        self.assertIsNotNone(tokenizer.unk_token_id)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()