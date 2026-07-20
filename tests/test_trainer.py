"""Unit tests for the LFMTrainer."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from liquid_foundation_model.model.configuration.config import TrainingConfig
from liquid_foundation_model.training.trainer import LFMTrainer


class MockModel(nn.Module):
    """Mock model for testing the trainer."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        outputs = self.linear(input_ids)
        loss = torch.mean(outputs)
        
        # Create a mock output object with a loss attribute
        class MockOutput:
            def __init__(self, loss):
                self.loss = loss
        
        return MockOutput(loss)
    
    def save_pretrained(self, output_dir):
        """Mock save_pretrained method."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "model.pt"))
    
    @classmethod
    def from_pretrained(cls, checkpoint_dir):
        """Mock from_pretrained method."""
        model = cls()
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
        return model
    
    def gradient_checkpointing_enable(self):
        """Mock gradient_checkpointing_enable method."""
        pass


class TestLFMTrainer(unittest.TestCase):
    """Test cases for the LFMTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model
        self.model = MockModel()
        
        # Create a mock dataset and dataloader
        input_ids = torch.randn(100, 10)
        attention_mask = torch.ones(100, 10)
        labels = torch.randn(100, 1)
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        
        def collate_fn(batch):
            input_ids = torch.stack([item[0] for item in batch])
            attention_mask = torch.stack([item[1] for item in batch])
            labels = torch.stack([item[2] for item in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        
        self.train_dataloader = DataLoader(
            dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
        )
        self.eval_dataloader = DataLoader(
            dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
        )
        
        # Create a training config
        self.config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=10,
            max_steps=100,
            gradient_accumulation_steps=1,
            fp16=False,
            bf16=False,
            gradient_checkpointing=False,
        )
        
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_trainer_initialization(self):
        """Test that the trainer initializes correctly."""
        trainer = LFMTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )
        
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.epoch, 0)
        self.assertEqual(trainer.best_eval_loss, float("inf"))
    
    def test_training_step(self):
        """Test that a single training step works correctly."""
        trainer = LFMTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )
        
        # Get a batch from the dataloader
        batch = next(iter(self.train_dataloader))
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        
        # Perform a training step
        loss = trainer._training_step(batch)
        
        # Check that the loss is a float
        self.assertIsInstance(loss, float)
        
        # Check that the global step was incremented
        self.assertEqual(trainer.global_step, 1)
    
    def test_evaluate(self):
        """Test that evaluation works correctly."""
        trainer = LFMTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )
        
        # Evaluate the model
        metrics = trainer.evaluate()
        
        # Check that the metrics include eval_loss
        self.assertIn("eval_loss", metrics)
        self.assertIsInstance(metrics["eval_loss"], float)
    
    def test_save_and_load_checkpoint(self):
        """Test that saving and loading checkpoints works correctly."""
        trainer = LFMTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )
        
        # Save a checkpoint
        checkpoint_dir = os.path.join(self.temp_dir.name, "checkpoint")
        trainer.save_checkpoint(checkpoint_dir)
        
        # Check that the checkpoint files exist
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "model.pt")))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "training_state.pt")))
        
        # Create a new trainer
        new_trainer = LFMTrainer(
            model=MockModel(),
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )
        
        # Load the checkpoint
        new_trainer.load_checkpoint(checkpoint_dir)
        
        # Check that the training state was loaded correctly
        self.assertEqual(new_trainer.global_step, trainer.global_step)
        self.assertEqual(new_trainer.epoch, trainer.epoch)
        self.assertEqual(new_trainer.best_eval_loss, trainer.best_eval_loss)
    
    @patch("tqdm.auto.tqdm")
    def test_train(self, mock_tqdm):
        """Test that training works correctly."""
        # Mock tqdm to avoid progress bar in tests
        mock_tqdm.return_value = range(10)
        
        trainer = LFMTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )
        
        # Train for a small number of steps
        metrics = trainer.train(
            max_steps=5,
            eval_steps=5,
            save_steps=5,
            output_dir=os.path.join(self.temp_dir.name, "training"),
            log_steps=1,
        )
        
        # Check that the metrics include train_loss and eval_loss
        self.assertIn("train_loss", metrics)
        self.assertIn("eval_loss", metrics)
        self.assertIn("learning_rate", metrics)
        self.assertIn("steps_per_second", metrics)
        
        # Check that the global step was incremented
        self.assertEqual(trainer.global_step, 5)
        
        # Check that checkpoints were saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "training", "checkpoint-5")))


if __name__ == "__main__":
    unittest.main()