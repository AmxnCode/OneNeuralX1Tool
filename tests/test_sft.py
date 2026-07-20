"""Unit tests for Supervised Fine-Tuning (SFT) components."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from liquid_foundation_model.model.configuration.config import TrainingConfig
from liquid_foundation_model.training.sft.sft import (
    SFTLoss,
    SFTDataProcessor,
    SFTTrainer,
    create_sft_trainer,
)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.mask_token_id = 3
    
    def encode(self, text, add_special_tokens=True, truncation=True, max_length=None, return_tensors=None):
        """Mock encode method."""
        # Generate some fake token IDs based on the length of the text
        if isinstance(text, list):
            result = []
            for t in text:
                length = min(len(t), max_length) if max_length else len(t)
                ids = list(range(10, 10 + length))
                if add_special_tokens:
                    ids = [self.bos_token_id] + ids
                result.append(ids)
            
            if return_tensors == "pt":
                # Convert to padded tensor
                max_len = max(len(ids) for ids in result)
                padded = []
                for ids in result:
                    padded.append(ids + [self.pad_token_id] * (max_len - len(ids)))
                return torch.tensor(padded)
            return result
        else:
            length = min(len(text), max_length) if max_length else len(text)
            ids = list(range(10, 10 + length))
            if add_special_tokens:
                ids = [self.bos_token_id] + ids
            
            if return_tensors == "pt":
                return torch.tensor([ids])
            return ids


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 32)
        self.linear = nn.Linear(32, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Mock forward method."""
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings)
        
        loss = None
        if labels is not None:
            # Simple cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        # Return an object with logits and loss attributes
        class Output:
            pass
        
        output = Output()
        output.logits = logits
        output.loss = loss
        
        return output


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, size=100):
        self.size = size
        self.data = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.randint(0, 100, (20,)),
            }
            for _ in range(size)
        ]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


class TestSFTLoss(unittest.TestCase):
    """Test cases for SFTLoss."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = SFTLoss()
        self.batch_size = 2
        self.seq_len = 10
        self.vocab_size = 100
        
        # Create sample inputs
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones(self.batch_size, self.seq_len)
    
    def test_forward(self):
        """Test forward pass of SFTLoss."""
        # Test without attention mask
        loss, loss_dict = self.loss_fn(self.logits, self.labels)
        
        # Check that loss is a scalar
        self.assertIsInstance(loss.item(), float)
        
        # Check that loss_dict contains sft_loss
        self.assertIn("sft_loss", loss_dict)
        self.assertEqual(loss, loss_dict["sft_loss"])
        
        # Test with attention mask
        loss, loss_dict = self.loss_fn(self.logits, self.labels, self.attention_mask)
        
        # Check that loss is a scalar
        self.assertIsInstance(loss.item(), float)
        
        # Check that loss_dict contains sft_loss
        self.assertIn("sft_loss", loss_dict)
        self.assertEqual(loss, loss_dict["sft_loss"])
    
    def test_label_smoothing(self):
        """Test label smoothing in SFTLoss."""
        # Create loss function with label smoothing
        loss_fn_smooth = SFTLoss(label_smoothing=0.1)
        
        # Compute losses with and without label smoothing
        loss_no_smooth, _ = self.loss_fn(self.logits, self.labels)
        loss_smooth, _ = loss_fn_smooth(self.logits, self.labels)
        
        # Label smoothing should change the loss value
        self.assertNotEqual(loss_no_smooth.item(), loss_smooth.item())


class TestSFTDataProcessor(unittest.TestCase):
    """Test cases for SFTDataProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = MockTokenizer()
        self.processor = SFTDataProcessor(
            tokenizer=self.tokenizer,
            max_length=128,
            prompt_template="### Instruction: {prompt}",
            response_template="### Response: {response}",
        )
    
    def test_format_prompt(self):
        """Test prompt formatting."""
        prompt = "Tell me a joke"
        formatted = self.processor.format_prompt(prompt)
        self.assertEqual(formatted, "### Instruction: Tell me a joke")
    
    def test_format_response(self):
        """Test response formatting."""
        response = "Why did the chicken cross the road?"
        formatted = self.processor.format_response(response)
        self.assertEqual(formatted, "### Response: Why did the chicken cross the road?")
    
    def test_create_prompt_response_pair(self):
        """Test creation of prompt-response pairs."""
        prompt = "Tell me a joke"
        response = "Why did the chicken cross the road?"
        
        result = self.processor.create_prompt_response_pair(prompt, response)
        
        # Check that result contains expected keys
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)
        
        # Check that input_ids and labels have the same length
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))
        
        # Check that attention_mask is all ones
        self.assertTrue((result["attention_mask"] == 1).all())
        
        # Check that labels for prompt part are -100 (ignored in loss)
        prompt_ids_len = len(self.tokenizer.encode(
            self.processor.format_prompt(prompt),
            add_special_tokens=True,
            return_tensors=None,
        ))
        self.assertTrue((result["labels"][:prompt_ids_len] == -100).all())
        
        # Check that labels for response part are not -100
        self.assertTrue((result["labels"][prompt_ids_len:] != -100).any())


class TestSFTTrainer(unittest.TestCase):
    """Test cases for SFTTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=10,
            max_steps=100,
            gradient_accumulation_steps=1,
            fp16=False,
            bf16=False,
            gradient_checkpointing=False,
        )
        self.train_dataset = MockDataset(size=10)
        self.eval_dataset = MockDataset(size=5)
        
        # Create dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=2)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=2)
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            device="cpu",
        )
    
    def test_initialization(self):
        """Test trainer initialization."""
        # Check that model is set
        self.assertEqual(self.trainer.model, self.model)
        
        # Check that config is set
        self.assertEqual(self.trainer.config, self.config)
        
        # Check that dataloaders are set
        self.assertEqual(self.trainer.train_dataloader, self.train_dataloader)
        self.assertEqual(self.trainer.eval_dataloader, self.eval_dataloader)
        
        # Check that optimizer and scheduler are created
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.lr_scheduler)
    
    def test_training_step(self):
        """Test a single training step."""
        # Get a batch from the dataloader
        batch = next(iter(self.train_dataloader))
        
        # Perform a training step
        loss = self.trainer._training_step(batch)
        
        # Check that loss is a float
        self.assertIsInstance(loss, float)
    
    def test_evaluate(self):
        """Test evaluation."""
        # Run evaluation
        metrics = self.trainer.evaluate()
        
        # Check that metrics contains eval_loss
        self.assertIn("eval_loss", metrics)
        self.assertIsInstance(metrics["eval_loss"], float)
    
    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            self.trainer.save_checkpoint(tmpdir)
            
            # Check that checkpoint files exist
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "pytorch_model.bin")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "training_state.pt")))
            
            # Create a new trainer
            new_trainer = SFTTrainer(
                model=MockModel(),  # New model instance
                config=self.config,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                device="cpu",
            )
            
            # Load checkpoint
            new_trainer.load_checkpoint(tmpdir)
            
            # Check that global_step is the same
            self.assertEqual(self.trainer.global_step, new_trainer.global_step)
            
            # Check that epoch is the same
            self.assertEqual(self.trainer.epoch, new_trainer.epoch)
            
            # Check that best_eval_loss is the same
            self.assertEqual(self.trainer.best_eval_loss, new_trainer.best_eval_loss)
    
    @patch("liquid_foundation_model.training.sft.sft.tqdm")
    def test_train(self, mock_tqdm):
        """Test training for a few steps."""
        # Mock tqdm to avoid progress bar in tests
        mock_tqdm.return_value = self.train_dataloader
        
        # Train for a few steps
        metrics = self.trainer.train(max_steps=2, log_steps=1, eval_steps=2)
        
        # Check that metrics contains expected keys
        self.assertIn("train_loss", metrics)
        self.assertIn("eval_loss", metrics)
        self.assertIn("learning_rate", metrics)
        self.assertIn("steps_per_second", metrics)
        
        # Check that global_step has been updated
        self.assertEqual(self.trainer.global_step, 2)


class TestCreateSFTTrainer(unittest.TestCase):
    """Test cases for create_sft_trainer function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.tokenizer = MockTokenizer()
        self.train_dataset = MockDataset(size=10)
        self.eval_dataset = MockDataset(size=5)
    
    def test_create_trainer(self):
        """Test creation of SFT trainer."""
        # Create trainer
        trainer = create_sft_trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            batch_size=4,
            device="cpu",
        )
        
        # Check that trainer is an instance of SFTTrainer
        self.assertIsInstance(trainer, SFTTrainer)
        
        # Check that model is set
        self.assertEqual(trainer.model, self.model)
        
        # Check that batch size is set correctly
        self.assertEqual(trainer.config.batch_size, 4)
        
        # Check that dataloaders are created
        self.assertIsNotNone(trainer.train_dataloader)
        self.assertIsNotNone(trainer.eval_dataloader)


if __name__ == "__main__":
    unittest.main()