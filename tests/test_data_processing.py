"""Unit tests for the LFM data processing pipeline."""

import os
import tempfile
import unittest
from typing import Dict, List

import pytest
import torch
from datasets import Dataset, DatasetDict

from liquid_foundation_model.training.data.dataset import (
    DataCollatorForLanguageModeling,
    LFMDatasetProcessor,
    LFMMultilingualProcessor,
)
from liquid_foundation_model.training.data.tokenizer.tokenizer import LFMTokenizer


class TestDataCollator(unittest.TestCase):
    """Test cases for the data collator."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a common pretrained tokenizer for testing
        self.tokenizer = LFMTokenizer(from_pretrained="gpt2")
        
        # Create a data collator
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create a data collator for MLM
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )
        
        # Sample input IDs
        self.sample_input_ids = [
            [1, 2, 3, 4, 5],
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6, 7],
        ]
    
    def test_collator_causal_lm(self):
        """Test the data collator for causal language modeling."""
        # Create a batch
        batch = self.collator(self.sample_input_ids)
        
        # Check that we have the expected keys
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("labels", batch)
        
        # Check shapes
        batch_size = len(self.sample_input_ids)
        max_length = max(len(ids) for ids in self.sample_input_ids)
        
        self.assertEqual(batch["input_ids"].shape, (batch_size, max_length))
        self.assertEqual(batch["attention_mask"].shape, (batch_size, max_length))
        self.assertEqual(batch["labels"].shape, (batch_size, max_length))
        
        # Check that padding is correct
        for i, ids in enumerate(self.sample_input_ids):
            # Check that the original IDs are preserved
            self.assertTrue(torch.all(batch["input_ids"][i, :len(ids)] == torch.tensor(ids)))
            
            # Check that padding is correct
            self.assertTrue(torch.all(batch["attention_mask"][i, :len(ids)] == 1))
            self.assertTrue(torch.all(batch["attention_mask"][i, len(ids):] == 0))
            
            # Check that labels are correct
            self.assertTrue(torch.all(batch["labels"][i, :len(ids)] == torch.tensor(ids)))
            self.assertTrue(torch.all(batch["labels"][i, len(ids):] == -100))
    
    def test_collator_mlm(self):
        """Test the data collator for masked language modeling."""
        # Create a batch
        batch = self.mlm_collator(self.sample_input_ids)
        
        # Check that we have the expected keys
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("labels", batch)
        
        # Check shapes
        batch_size = len(self.sample_input_ids)
        max_length = max(len(ids) for ids in self.sample_input_ids)
        
        self.assertEqual(batch["input_ids"].shape, (batch_size, max_length))
        self.assertEqual(batch["attention_mask"].shape, (batch_size, max_length))
        self.assertEqual(batch["labels"].shape, (batch_size, max_length))
        
        # Check that some tokens are masked
        mask_token_id = self.tokenizer.mask_token_id
        masked_tokens = (batch["input_ids"] == mask_token_id).sum().item()
        self.assertGreater(masked_tokens, 0)
        
        # Check that labels are -100 for non-masked tokens
        non_masked_labels = (batch["labels"] != -100).sum().item()
        self.assertEqual(non_masked_labels, masked_tokens)


class TestDatasetProcessor(unittest.TestCase):
    """Test cases for the dataset processor."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a common pretrained tokenizer for testing
        self.tokenizer = LFMTokenizer(from_pretrained="gpt2")
        
        # Create a dataset processor
        self.processor = LFMDatasetProcessor(
            tokenizer=self.tokenizer,
            max_length=128,
        )
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample text file
        self.text_file = os.path.join(self.temp_dir, "sample.txt")
        with open(self.text_file, "w") as f:
            f.write("This is a sample text file.\n")
            f.write("It contains multiple lines.\n")
            f.write("We will use it for testing the dataset processor.\n")
        
        # Create a sample JSON file
        self.json_file = os.path.join(self.temp_dir, "sample.json")
        with open(self.json_file, "w") as f:
            f.write('{"text": "This is a JSON sample.", "label": 1}\n')
            f.write('{"text": "Another JSON sample.", "label": 0}\n')
        
        # Create a sample instruction dataset
        self.instruction_data = {
            "prompt": [
                "What is the capital of France?",
                "Explain quantum computing.",
            ],
            "response": [
                "The capital of France is Paris.",
                "Quantum computing is a type of computing that uses quantum phenomena.",
            ],
        }
        self.instruction_dataset = Dataset.from_dict(self.instruction_data)
    
    def test_load_dataset_from_file(self):
        """Test loading a dataset from a file."""
        # Load from text file
        text_dataset = self.processor.load_dataset(self.text_file)
        self.assertIsInstance(text_dataset, Dataset)
        self.assertGreater(len(text_dataset), 0)
        
        # Load from JSON file
        json_dataset = self.processor.load_dataset(self.json_file)
        self.assertIsInstance(json_dataset, Dataset)
        self.assertGreater(len(json_dataset), 0)
    
    def test_preprocess_for_language_modeling(self):
        """Test preprocessing for language modeling."""
        # Load dataset
        dataset = self.processor.load_dataset(self.json_file)
        
        # Preprocess for language modeling
        processed_dataset = self.processor.preprocess_for_language_modeling(
            dataset, text_column_name="text"
        )
        
        # Check that we have the expected columns
        self.assertIn("input_ids", processed_dataset.column_names)
        
        # Check that we have some examples
        self.assertGreater(len(processed_dataset), 0)
    
    def test_preprocess_for_instruction_tuning(self):
        """Test preprocessing for instruction tuning."""
        # Preprocess for instruction tuning
        processed_dataset = self.processor.preprocess_for_instruction_tuning(
            self.instruction_dataset,
            prompt_column_name="prompt",
            response_column_name="response",
        )
        
        # Check that we have the expected columns
        self.assertIn("input_ids", processed_dataset.column_names)
        self.assertIn("labels", processed_dataset.column_names)
        
        # Check that we have the same number of examples
        self.assertEqual(len(processed_dataset), len(self.instruction_dataset))
        
        # Check that labels have -100 for prompt tokens
        for example in processed_dataset:
            labels = example["labels"]
            # Some labels should be -100 (for prompt)
            self.assertIn(-100, labels)
            # Some labels should not be -100 (for response)
            self.assertTrue(any(label != -100 for label in labels))
    
    def test_create_dataloader(self):
        """Test creating a dataloader."""
        # Load dataset
        dataset = self.processor.load_dataset(self.json_file)
        
        # Preprocess for language modeling
        processed_dataset = self.processor.preprocess_for_language_modeling(
            dataset, text_column_name="text"
        )
        
        # Create dataloader
        dataloader = self.processor.create_dataloader(
            processed_dataset, batch_size=2
        )
        
        # Check that we can iterate over the dataloader
        batch = next(iter(dataloader))
        
        # Check that we have the expected keys
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("labels", batch)
        
        # Check shapes
        self.assertEqual(batch["input_ids"].shape[0], 2)  # batch size
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)


class TestMultilingualProcessor(unittest.TestCase):
    """Test cases for the multilingual dataset processor."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a common pretrained tokenizer for testing
        self.tokenizer = LFMTokenizer(from_pretrained="gpt2")
        
        # Create a multilingual processor
        self.processor = LFMMultilingualProcessor(
            tokenizer=self.tokenizer,
            max_length=128,
            supported_languages=["en", "fr", "de"],
        )
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample text files for different languages
        self.en_file = os.path.join(self.temp_dir, "en.txt")
        with open(self.en_file, "w") as f:
            f.write("This is an English text.\n")
            f.write("It contains multiple lines.\n")
        
        self.fr_file = os.path.join(self.temp_dir, "fr.txt")
        with open(self.fr_file, "w") as f:
            f.write("Ceci est un texte français.\n")
            f.write("Il contient plusieurs lignes.\n")
        
        self.de_file = os.path.join(self.temp_dir, "de.txt")
        with open(self.de_file, "w") as f:
            f.write("Dies ist ein deutscher Text.\n")
            f.write("Er enthält mehrere Zeilen.\n")
    
    def test_load_multilingual_dataset(self):
        """Test loading multilingual datasets."""
        # Define dataset paths
        dataset_paths = {
            "en": self.en_file,
            "fr": self.fr_file,
            "de": self.de_file,
            "es": os.path.join(self.temp_dir, "nonexistent.txt"),  # This should be skipped
        }
        
        # Load multilingual datasets
        datasets = self.processor.load_multilingual_dataset(dataset_paths)
        
        # Check that we have datasets for supported languages
        self.assertIn("en", datasets)
        self.assertIn("fr", datasets)
        self.assertIn("de", datasets)
        
        # Check that unsupported languages are skipped
        self.assertNotIn("es", datasets)
        
        # Check that datasets have content
        for lang, dataset in datasets.items():
            self.assertGreater(len(dataset), 0)
    
    def test_combine_multilingual_datasets(self):
        """Test combining multilingual datasets."""
        # Load individual datasets
        en_dataset = self.processor.load_dataset(self.en_file)
        fr_dataset = self.processor.load_dataset(self.fr_file)
        de_dataset = self.processor.load_dataset(self.de_file)
        
        # Combine datasets
        combined_dataset = self.processor.combine_multilingual_datasets(
            {"en": en_dataset, "fr": fr_dataset, "de": de_dataset},
            add_language_tags=True,
        )
        
        # Check that we have a combined dataset
        self.assertIsInstance(combined_dataset, Dataset)
        
        # Check that the combined dataset has the expected size
        expected_size = len(en_dataset) + len(fr_dataset) + len(de_dataset)
        self.assertEqual(len(combined_dataset), expected_size)
        
        # Check that we have language columns
        self.assertIn("language", combined_dataset.column_names)
        
        # Check that we have language tags in the text
        for example in combined_dataset:
            lang = example["language"]
            text = example["text"]
            self.assertTrue(text.startswith(f"[{lang}]"))
    
    def test_preprocess_for_multilingual(self):
        """Test preprocessing for multilingual training."""
        # Load individual datasets
        en_dataset = self.processor.load_dataset(self.en_file)
        fr_dataset = self.processor.load_dataset(self.fr_file)
        
        # Add language column
        en_dataset = en_dataset.add_column("language", ["en"] * len(en_dataset))
        fr_dataset = fr_dataset.add_column("language", ["fr"] * len(fr_dataset))
        
        # Combine datasets
        combined_dataset = Dataset.from_dict({
            "text": en_dataset["text"] + fr_dataset["text"],
            "language": en_dataset["language"] + fr_dataset["language"],
        })
        
        # Preprocess for multilingual training
        processed_dataset = self.processor.preprocess_for_multilingual(
            combined_dataset,
            text_column_name="text",
            language_column_name="language",
            languages=["en"],  # Only English
        )
        
        # Check that we have the expected columns
        self.assertIn("input_ids", processed_dataset.column_names)
        
        # Check that we have some examples
        self.assertGreater(len(processed_dataset), 0)
        
        # Check that we only have English examples
        self.assertEqual(len(processed_dataset), len(en_dataset))
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()