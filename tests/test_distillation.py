"""Unit tests for knowledge distillation components."""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquid_foundation_model.model.configuration.config import TrainingConfig
from liquid_foundation_model.training.distillation.distillation import (
    DistillationLoss,
    TeacherModelHandler,
    DistillationTrainer,
)


class TestDistillationLoss(unittest.TestCase):
    """Test cases for the DistillationLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 10
        self.vocab_size = 100
        
        # Create sample logits and labels
        self.student_logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.teacher_logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones(self.batch_size, self.seq_len)
    
    def test_init(self):
        """Test initialization with different parameters."""
        # Default parameters
        loss_fn = DistillationLoss()
        self.assertEqual(loss_fn.temperature, 2.0)
        self.assertEqual(loss_fn.alpha, 0.5)
        self.assertEqual(loss_fn.loss_type, "kl")
        
        # Custom parameters
        loss_fn = DistillationLoss(temperature=3.0, alpha=0.7, loss_type="mse")
        self.assertEqual(loss_fn.temperature, 3.0)
        self.assertEqual(loss_fn.alpha, 0.7)
        self.assertEqual(loss_fn.loss_type, "mse")
        
        # Invalid loss type
        with self.assertRaises(ValueError):
            DistillationLoss(loss_type="invalid")
    
    def test_kl_loss(self):
        """Test KL divergence loss calculation."""
        loss_fn = DistillationLoss(loss_type="kl")
        loss, loss_dict = loss_fn(
            student_logits=self.student_logits,
            teacher_logits=self.teacher_logits,
        )
        
        # Check that loss is a scalar
        self.assertTrue(isinstance(loss.item(), float))
        
        # Check that loss_dict contains expected keys
        self.assertIn("distillation_loss", loss_dict)
        self.assertIn("total_loss", loss_dict)
        
        # Check that loss equals distillation_loss when no labels
        self.assertEqual(loss.item(), loss_dict["distillation_loss"].item())
    
    def test_mse_loss(self):
        """Test MSE loss calculation."""
        loss_fn = DistillationLoss(loss_type="mse")
        loss, loss_dict = loss_fn(
            student_logits=self.student_logits,
            teacher_logits=self.teacher_logits,
        )
        
        # Check that loss is a scalar
        self.assertTrue(isinstance(loss.item(), float))
        
        # Check that loss_dict contains expected keys
        self.assertIn("distillation_loss", loss_dict)
        self.assertIn("total_loss", loss_dict)
    
    def test_ce_loss(self):
        """Test cross-entropy loss calculation."""
        loss_fn = DistillationLoss(loss_type="ce")
        loss, loss_dict = loss_fn(
            student_logits=self.student_logits,
            teacher_logits=self.teacher_logits,
        )
        
        # Check that loss is a scalar
        self.assertTrue(isinstance(loss.item(), float))
        
        # Check that loss_dict contains expected keys
        self.assertIn("distillation_loss", loss_dict)
        self.assertIn("total_loss", loss_dict)
    
    def test_with_labels(self):
        """Test loss calculation with labels."""
        loss_fn = DistillationLoss(alpha=0.3)
        loss, loss_dict = loss_fn(
            student_logits=self.student_logits,
            teacher_logits=self.teacher_logits,
            labels=self.labels,
        )
        
        # Check that loss is a scalar
        self.assertTrue(isinstance(loss.item(), float))
        
        # Check that loss_dict contains expected keys
        self.assertIn("distillation_loss", loss_dict)
        self.assertIn("task_loss", loss_dict)
        self.assertIn("total_loss", loss_dict)
        
        # Check that total_loss is a weighted combination of task_loss and distillation_loss
        expected_loss = (1 - 0.3) * loss_dict["task_loss"] + 0.3 * loss_dict["distillation_loss"]
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)
    
    def test_with_attention_mask(self):
        """Test loss calculation with attention mask."""
        # Create a mask with some padding
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        attention_mask[:, -2:] = 0  # Mask out the last two tokens
        
        loss_fn = DistillationLoss()
        loss, loss_dict = loss_fn(
            student_logits=self.student_logits,
            teacher_logits=self.teacher_logits,
            attention_mask=attention_mask,
        )
        
        # Check that loss is a scalar
        self.assertTrue(isinstance(loss.item(), float))
        
        # Check that loss_dict contains expected keys
        self.assertIn("distillation_loss", loss_dict)
        self.assertIn("total_loss", loss_dict)


class TestTeacherModelHandler(unittest.TestCase):
    """Test cases for the TeacherModelHandler class."""
    
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_huggingface_model(self, mock_tokenizer, mock_model):
        """Test loading a model from HuggingFace."""
        # Mock the model and tokenizer
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        # Create handler and load model
        handler = TeacherModelHandler("gpt2")
        model = handler.load_model()
        
        # Check that model was loaded
        self.assertIsNotNone(model)
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
    
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_get_logits(self, mock_tokenizer, mock_model):
        """Test getting logits from the teacher model."""
        # Mock the model and tokenizer
        mock_model_instance = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 10, 100)
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = MagicMock()
        
        # Set device property on mock model
        mock_model_instance.device = torch.device("cpu")
        
        # Create handler and manually set model
        handler = TeacherModelHandler("gpt2")
        handler.model = mock_model_instance
        
        # Create input tensors
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        # Get logits
        logits = handler.get_logits(input_ids, attention_mask)
        
        # Check that model was called
        mock_model_instance.assert_called_once()


class TestDistillationTrainer(unittest.TestCase):
    """Test cases for the DistillationTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock student model
        self.student_model = MagicMock()
        self.student_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        # Create mock dataloader
        self.dataloader = MagicMock()
        self.dataloader.__iter__.return_value = [
            {
                "input_ids": torch.randint(0, 100, (2, 10)),
                "attention_mask": torch.ones(2, 10),
                "labels": torch.randint(0, 100, (2, 10)),
            }
        ]
        self.dataloader.__len__.return_value = 1
        
        # Create config
        self.config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-4,
            max_steps=10,
            fp16=False,
        )
    
    @patch.object(TeacherModelHandler, "load_model")
    @patch.object(TeacherModelHandler, "get_logits")
    @patch.object(DistillationLoss, "forward")
    def test_training_step(self, mock_loss_forward, mock_get_logits, mock_load_model):
        """Test a single training step."""
        # Mock teacher model
        mock_load_model.return_value = MagicMock()
        mock_get_logits.return_value = torch.randn(2, 10, 100)
        
        # Mock student model outputs
        self.student_model.return_value = torch.randn(2, 10, 100)
        
        # Mock loss function
        mock_loss = torch.tensor(0.5, requires_grad=True)
        mock_loss_dict = {
            "total_loss": torch.tensor(0.5),
            "distillation_loss": torch.tensor(0.3),
            "task_loss": torch.tensor(0.2),
        }
        mock_loss_forward.return_value = (mock_loss, mock_loss_dict)
        
        # Create trainer
        trainer = DistillationTrainer(
            student_model=self.student_model,
            teacher_model_name_or_path="gpt2",
            config=self.config,
            train_dataloader=self.dataloader,
        )
        
        # Create batch
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 100, (2, 10)),
        }
        
        # Mock optimizer and scheduler to avoid backward pass issues
        trainer.optimizer = MagicMock()
        trainer.lr_scheduler = MagicMock()
        
        # Perform training step
        losses = trainer._training_step(batch)
        
        # Check that losses dictionary contains expected keys
        self.assertIn("total_loss", losses)
        self.assertIn("distillation_loss", losses)
        self.assertIn("task_loss", losses)
        
        # Check that student and teacher models were called
        self.student_model.assert_called_once()
        mock_get_logits.assert_called_once()


if __name__ == "__main__":
    unittest.main()