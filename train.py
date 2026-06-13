import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from snn.models import EmFlow
from snn.dataset import OpticalFlowDataset
#from comparisons.mvsec.mvsec_dataset import MVSECDataset as OpticalFlowDataset
#from comparisons.dsec.dsec_dataset import DSECOpticalFlowDataset as OpticalFlowDataset
from snn.training import SNNTrainer
from snn.utils.logger import Logger
import random

from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train SNN for Optical Flow')
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--name', type=str, default='01_base_model',
                      help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                      help='Directory for logs')
    parser.add_argument('--seed', type=int, default=42,
                      help='Directory for logs')
    parser.add_argument('--train-data-root', type=str, default=None,
                      help='Override training dataset root (config fallback: data_root)')
    parser.add_argument('--val-data-root', type=str, default=None,
                      help='Validation dataset root (config fallback: val_data_root)')
    
    return parser.parse_args()


def build_dataloaders(config: dict, train_root: str = None, val_root: str = None):
    train_root = train_root or config.get('data_root', 'data/hflow320/train_set')
    val_root = val_root or config.get('val_data_root', 'data/hflow320/valid_set')

    #regular Train dataset
    train_config = config.copy()
    train_config['data_root'] = train_root
    train_dataset = OpticalFlowDataset(config=train_config)

    # Val dataset
    val_config = config.copy()
    val_config['data_root'] = val_root
    val_dataset = OpticalFlowDataset(config=val_config)

    #KEEP for MVSEC # MVSEC Train dataset
    #KEEP for MVSEC train_config = config.copy()
    #KEEP for MVSEC train_config['dt'] = 1
    #KEEP for MVSEC train_config['data_root'] = train_root
    #KEEP for MVSEC train_config['sequences'] = ['outdoor_day2']
    #KEEP for MVSEC train_config['filter'] = {'outdoor_day2': (0, 3474)}
    #KEEP for MVSEC train_config['crop_size'] = (256, 344)
    #KEEP for MVSEC train_dataset = OpticalFlowDataset(config=train_config)

    #KEEP for MVSEC # MVSEC Val dataset
    #KEEP for MVSEC val_config = config.copy()
    #KEEP for MVSEC val_config['dt'] = 1
    #KEEP for MVSEC val_config['data_root'] = train_root
    #KEEP for MVSEC val_config['sequences'] = ['outdoor_day2']
    #KEEP for MVSEC val_config['filter'] = {'outdoor_day2': (3474, 4343)}
    #KEEP for MVSEC val_config['crop_size'] = (256, 344)
    #KEEP for MVSEC val_dataset = OpticalFlowDataset(config=val_config)

    # DSEC dataset
    #train_config = config.copy()
    #train_config['data_root'] = train_root
    #train_dataset = OpticalFlowDataset(config=train_config)

    ## Val dataset
    #val_config = config.copy()
    #val_config['data_root'] = val_root
    #val_dataset = OpticalFlowDataset(config=val_config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    args = parse_args()
    
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    #seed = args.seed
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #if device == 'cuda':
    #    torch.cuda.manual_seed_all(seed)
    checkpoint_dir = Path(args.checkpoint_dir) / args.name
    log_dir = Path(args.log_dir) / args.name
    
    model = build_model(config, device=device, train=True)
    print(f"Built model: {config.get('model_type', 'SpikingFlowNet')}")
    
    logger = Logger(log_dir=log_dir)
    model.set_logger(logger)
    
    if config.get('quantize_weights', False) or config.get('quantize_activations', False) or config.get('quantize_mem', False):
        print(f"Quantization enabled: W{config.get('weight_bit_width', 8)}A{config.get('act_bit_width', 8)}M{config.get('mem_bit_width', 16)}")
        print(f"  Weight quantization: {config.get('quantize_weights', False)}")
        print(f"  Activation quantization: {config.get('quantize_activations', False)}")
        print(f"  Membrane quantization: {config.get('quantize_mem', False)}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    train_root = args.train_data_root or config.get('data_root', None)
    val_root = args.val_data_root or config.get('val_data_root', None)
    train_loader, val_loader = build_dataloaders(config, train_root, val_root)

    print(f"Train data root: {train_root}")
    print(f"Val data root: {val_root}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    
    trainer = SNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        logger=logger 
    )
    
    # Log configuration
    logger.log_config(config, model=model)
    
    # Train
    num_epochs = config.get('num_epochs', 100)
    trainer.train(num_epochs=num_epochs, resume=args.resume)


if __name__ == '__main__':
    main()
