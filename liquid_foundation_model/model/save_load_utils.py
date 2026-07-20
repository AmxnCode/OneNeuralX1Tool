import os
import json
import torch

from liquid_foundation_model.model.configuration.config import LFMConfig
from liquid_foundation_model.model.liquid_foundation_model import LiquidFoundationModel, LiquidFoundationModelForCausalLM

def save_pretrained(model, save_directory):
    """
    Save a model and its configuration to a directory.
    
    Args:
        model: The model to save
        save_directory: Directory to save the model to
    """
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    # Save model weights
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
    
    # Save configuration
    config = model.config
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

def from_pretrained(model_class, pretrained_model_path, **kwargs):
    """
    Load a model from a pretrained model directory.
    
    Args:
        model_class: The model class to instantiate
        pretrained_model_path: Path to the pretrained model directory
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        The loaded model
    """
    # Load configuration
    config_path = os.path.join(pretrained_model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = LFMConfig.from_dict(config_dict)
    
    # Create model
    model = model_class(config, **kwargs)
    
    # Load weights
    model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    return model

# Add save_pretrained and from_pretrained methods to model classes
LiquidFoundationModel.save_pretrained = save_pretrained
LiquidFoundationModel.from_pretrained = classmethod(from_pretrained)
LiquidFoundationModelForCausalLM.save_pretrained = save_pretrained
LiquidFoundationModelForCausalLM.from_pretrained = classmethod(from_pretrained)