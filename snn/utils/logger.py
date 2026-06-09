import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any
import json


class Logger:
    """
    Logger class for training metrics and visualization
    Supports Tensorboard and file logging
    """
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Text log file
        self.log_file = self.log_dir / 'training.log'
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalars as individual metrics under a common prefix"""
        for key, value in values.items():
            self.writer.add_scalar(f'{tag}/{key}', value, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log image"""
        self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram"""
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """Log text"""
        self.writer.add_text(tag, text, step)
    
    def log_to_file(self, message: str):
        """Log message to text file"""
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
    
    def log_config(self, config: Dict[str, Any], model: torch.nn.Module = None):
        """Log configuration and optionally model structure"""
        config_file = self.log_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        config_str = json.dumps(config, indent=2)
        self.log_text('config', config_str)
        
        if model is not None:
            model_str = str(model)
            model_file = self.log_dir / 'model_structure.txt'
            with open(model_file, 'w') as f:
                f.write(model_str)
            self.log_text('model_structure', f'```\n{model_str}\n```')
    
    def close(self):
        """Close logger"""
        self.writer.close()
