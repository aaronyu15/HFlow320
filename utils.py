import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from snn.models import *

models = {
    'EventSNNFlowNetLite': EventSNNFlowNetLite,
    'OtherModel': None,  
}


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model(config: dict) -> torch.nn.Module:
    model_type = config.get('model_type', 'EventSNNFlowNetLite')

    model = models[model_type](config=config)

    return model


def build_model(config: dict, device='cuda', train=True, checkpoint_path=None, strict=False) -> torch.nn.Module:
    """Build model from configuration"""
    
    if train:

        model = get_model(config)

        model = model.to(device)
    
        return model
    else:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
        config = checkpoint.get('config', {})

        model = get_model(config)
    
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()
    
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Best validation EPE: {checkpoint.get('best_val_epe', 'unknown')}")
    
        return model, config


def load_teacher_model(checkpoint_path: str, device='cuda') -> torch.nn.Module:
    """
    Load a pre-trained teacher model from checkpoint for distillation.
    
    The teacher model is loaded using the config stored in the checkpoint,
    ensuring compatibility even if the current model code has changed.
    The model is set to eval mode and all parameters are frozen.
    
    Args:
        checkpoint_path: Path to teacher checkpoint file
        device: Device to load model on
        
    Returns:
        Frozen teacher model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    teacher_config = checkpoint.get('config', {})
    
    # Override model type to use Teacher class for versioning
    original_model_type = teacher_config.get('model_type', 'EventSNNFlowNetLite')
    
    # Map old model types to teacher versions
    teacher_model_map = {
        'EventSNNFlowNetLite': 'EventSNNFlowNetTeacher',
    }
    
    teacher_model_type = teacher_model_map.get(original_model_type, original_model_type)
    teacher_config['model_type'] = teacher_model_type
    
    print(f"Loading teacher model: {original_model_type} -> {teacher_model_type}")
    
    # Build model
    model = get_model(teacher_config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    
    # Freeze model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Loaded teacher model from {checkpoint_path}")
    print(f"Teacher trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Teacher best validation EPE: {checkpoint.get('best_val_epe', 'unknown')}")
    
    return model
