import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from pathlib import Path
from typing import Dict, Optional
import time
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from ..utils.logger import Logger
from ..utils.visualization import visualize_flow
from .losses import CombinedLoss, calculate_outliers

class SNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        logger: Optional[Logger] = None 
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger 
        
        self.criterion = CombinedLoss(
            endpoint_weight=config.get('endpoint_weight', 1.0),
            angular_weight=config.get('angular_weight', 0.5),
            epe_ang_weight=config.get('epe_ang_weight', 1.0),
            vertical_weight=config.get('vertical_weight', 1.0),
        )
        
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.get('lr_milestones', [100, 150, 200]),
            gamma=config.get('lr_gamma', 0.5)
        )

        self.event_mask = True
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_epe = float('inf')
        self.do_evaluation = config.get('do_evaluation', True)
    
    def _visualize_batch(self, inputs: torch.Tensor, gt_flow: torch.Tensor, 
                         valid_mask: torch.Tensor, pred_flow: Optional[torch.Tensor] = None,
                         max_images: int = 4) -> Dict[str, torch.Tensor]:
        """
        Visualize batch data for TensorBoard logging
        
        Args:
            inputs: Event tensor [B, num_bins, C, H, W] where C is pos/neg polarities
            gt_flow: Ground truth flow [B, 2, H, W]
            valid_mask: Valid mask [B, 1, H, W]
            pred_flow: Predicted flow [B, 2, H, W] (optional)
            max_images: Maximum number of images to visualize from batch
            
        Returns:
            Dictionary of visualizations as tensors
        """
        batch_size = min(inputs.shape[0], max_images)
        visualizations = {}
        
        # Visualize event input (sum across temporal bins and polarities)
        # Input is [B, num_bins, C, H, W] where C=2 for pos/neg polarities
        event_sum = inputs[:batch_size].sum(dim=(2), keepdim=True)  # [B, T, 1, H, W] sum polarities

        # Convert to RGB by repeating grayscale
        event_vis = event_sum.repeat(1, 1, 3, 1, 1)  # [B, 3, H, W]
        visualizations['events'] = event_vis.view(-1, 3, event_vis.shape[3], event_vis.shape[4])  
        
        # Visualize valid mask
        valid_vis = valid_mask[:batch_size].repeat(1, 1, 3, 1, 1)  # [B, 3, H, W]
        visualizations['valid_mask'] = valid_vis.view(-1, 3, valid_vis.shape[3], valid_vis.shape[4])  # [B, 3, H, W]
        
        max_flow = min(torch.norm(gt_flow, dim=1).max().item(), 1.0)
        # Visualize GT flow using Middlebury color scheme
        gt_flow_vis = []
        gt_flow_mask_vis = []

        for i in range(batch_size):
            flow_np = gt_flow[i].cpu()  # [2, H, W]

            flow_mask = gt_flow[i] * valid_mask[i]  # Mask out invalid pixels
            flow_np_mask = flow_mask.cpu()  # [2, H, W]

            flow_color = visualize_flow(flow_np, max_flow=max_flow)
            flow_color_mask = visualize_flow(flow_np_mask, max_flow=max_flow)

            flow_color = torch.from_numpy(flow_color).permute(2, 0, 1).float() / 255.0
            flow_color_mask = torch.from_numpy(flow_color_mask).permute(2, 0, 1).float() / 255.0

            gt_flow_vis.append(flow_color)
            gt_flow_mask_vis.append(flow_color_mask)

        visualizations['gt_flow'] = torch.stack(gt_flow_vis)  # [B, 3, H, W]
        visualizations['gt_flow_masked'] = torch.stack(gt_flow_mask_vis)  # [B, 3, H, W]
        
        # Visualize predicted flow if provided
        if pred_flow is not None:
            pred_flow_vis = []
            pred_flow_mask_vis = []
            for i in range(batch_size):
                flow_np = pred_flow[i].cpu()  # [2, H, W]

                flow_mask = pred_flow[i] * valid_mask[i]  # Mask out invalid pixels
                flow_np_mask = flow_mask.cpu()  # [2, H, W]

                flow_color = visualize_flow(flow_np, max_flow=max_flow)
                flow_color_mask = visualize_flow(flow_np_mask, max_flow=max_flow)

                flow_color = torch.from_numpy(flow_color).permute(2, 0, 1).float() / 255.0
                flow_color_mask = torch.from_numpy(flow_color_mask).permute(2, 0, 1).float() / 255.0

                pred_flow_vis.append(flow_color)
                pred_flow_mask_vis.append(flow_color_mask)

            visualizations['pred_flow'] = torch.stack(pred_flow_vis)  # [B, 3, H, W]
            visualizations['pred_flow_masked'] = torch.stack(pred_flow_mask_vis)  # [B, 3, H, W]
        
        return visualizations
        
    def _log_flow_histograms(
        self,
        pred_flow: torch.Tensor,
        valid_mask: torch.Tensor,
        prefix: str,
        postfix: str,
        step: int
    ):
        """
        Log histograms of predicted flow u and v components with same scale
        
        Args:
            pred_flow: Predicted flow [B, 2, H, W]
            valid_mask: Valid mask [B, 1, H, W]
            prefix: Prefix for logging (e.g., 'train' or 'val')
            step: Current training step
        """
        # Extract u and v components (only valid pixels)
        u_flow = pred_flow[:, 0, :, :]  # [B, H, W]
        v_flow = pred_flow[:, 1, :, :]  # [B, H, W]
        valid = valid_mask[:, 0, :, :] > 0.5  # [B, H, W]
        
        # Flatten and filter by valid mask
        u_valid = u_flow[valid].cpu()
        v_valid = v_flow[valid].cpu()
        
        if len(u_valid) > 0 and len(v_valid) > 0:
            # Compute common range for both histograms
            min_val = min(u_valid.min().item(), v_valid.min().item())
            max_val = max(u_valid.max().item(), v_valid.max().item())
            
            # Add small padding to range
            range_pad = (max_val - min_val) * 0.05
            min_val -= range_pad
            max_val += range_pad
            
            # Create histogram figure
            import io
            from PIL import Image
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            try:
                # U component histogram
                axes[0].hist(u_valid.detach().numpy(), bins=50, range=(min_val, max_val), 
                            alpha=0.7, color='blue', edgecolor='black')
                axes[0].set_xlabel('U Flow (horizontal)')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title(f'U Component Distribution\nMin: {u_valid.min():.2f}, Max: {u_valid.max():.2f}, Mean: {u_valid.mean():.2f}')
                axes[0].grid(True, alpha=0.3)
                axes[0].set_xlim(min_val, max_val)
                axes[0].set_yscale('log')  # Use log scale for better visibility of distribution tails
            
                # V component histogram
                axes[1].hist(v_valid.detach().numpy(), bins=50, range=(min_val, max_val),
                            alpha=0.7, color='green', edgecolor='black')
                axes[1].set_xlabel('V Flow (vertical)')
                axes[1].set_ylabel('Frequency')
                axes[1].set_title(f'V Component Distribution\nMin: {v_valid.min():.2f}, Max: {v_valid.max():.2f}, Mean: {v_valid.mean():.2f}')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_xlim(min_val, max_val)
                axes[1].set_yscale('log')  # Use log scale for better visibility of distribution tails
            
                plt.tight_layout()
            
                # Convert figure to image tensor
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)
                img_array = np.array(img)
            finally:
                plt.close(fig)
            
            # Convert to torch tensor [C, H, W] in range [0, 1]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            
            # Log as image
            self.logger.log_image(f'{prefix}/flow_histograms_{postfix}', img_tensor[:3], step)
            
            # Also log raw histogram data for TensorBoard's native histogram viewer
            self.logger.log_histogram(f'{prefix}/flow_u_{postfix}', u_valid, step)
            self.logger.log_histogram(f'{prefix}/flow_v_{postfix}', v_valid, step)
        
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        log_interval = self.config.get('scalar_log_interval', 20)
        running_epe = 0.0
        running_ang = 0.0
        running_count = 0
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            if self.event_mask:
                activity_patch = inputs.sum(dim=(1, 2))
                low_activity_mask = (activity_patch < 1)
                low_activity_mask = low_activity_mask.unsqueeze(1)

                valid_mask[low_activity_mask] = 0.0
            
            gt_mask = gt_flow.norm(dim=1, keepdim=True) > 1e-3
            valid_mask = valid_mask[:, :, :, :] > 0.5
            valid_mask = valid_mask & gt_mask
            
            losses = self.criterion(outputs, gt_flow, inputs, valid_mask)
            
            losses['total_loss'].backward()

            self.optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'epe': losses['endpoint_loss'].item(),
                'ang': losses['angular_loss'].item(),
            })
            
            # Log images at specified interval (less frequent than scalars)
            if self.global_step % self.config.get('image_log_interval', 100) == 0:
                visualizations = self._visualize_batch(
                    inputs, gt_flow, valid_mask, 
                    pred_flow=outputs['flow'],
                    max_images=self.config.get('max_images_log', 4)
                )
                
                # Log each visualization type
                for vis_name, vis_tensor in visualizations.items():
                    if vis_name == 'events' or vis_name == 'valid_mask':
                        # For events and valid mask, vis_tensor is [B*T, 3, H, W]
                        nrow = self.config.get('num_bins', 10)
                    else:
                        nrow = 2  # For flow visualizations
                    # TensorBoard expects grid of images, so we make a grid
                    grid = make_grid(vis_tensor, nrow=nrow, normalize=False, pad_value=1.0)
                    self.logger.log_image(f'train/{vis_name}', grid, self.global_step)
                
                # Log flow histograms (u and v components with same scale)
                self._log_flow_histograms(outputs['flow'], valid_mask, 'train', 'pred', self.global_step)

                # Log flow histograms (u and v components with same scale)
                self._log_flow_histograms(gt_flow, valid_mask, 'train', 'gt', self.global_step)
                
            
            self.global_step += 1

            running_epe += losses['endpoint_loss'].item()
            running_ang += losses['angular_loss'].item()
            running_count += 1
            if (batch_idx + 1) % log_interval == 0:
                avg_epe = running_epe / running_count
                avg_ang = running_ang / running_count
                self.logger.log_scalar('train/epe_intermediate', avg_epe, self.global_step)
                self.logger.log_scalar('train/ang_intermediate', avg_ang, self.global_step)
                running_epe = 0.0
                running_ang = 0.0
                running_count = 0

        # Compute epoch metrics with fixed weights after optimization is complete.
        if self.do_evaluation:
            return self.evaluate_train_epoch()
        else:
            return {}
        # ...existing code...
    
    @torch.no_grad()
    def evaluate_train_epoch(self) -> Dict[str, float]:
        """Evaluate training-set metrics using fixed model weights."""
        self.model.eval()

        # Force batch size 1 for evaluation
        eval_train_loader = torch.utils.data.DataLoader(
            self.train_loader.dataset, batch_size=32, shuffle=False, num_workers=16
        )
        train_losses = {
            'total_loss': 0.0,
            'endpoint_loss': 0.0,
            'angular_loss': 0.0,
            'epe_ang_loss': 0.0,
        }
        train_outliers = 0.0
        train_flow_max = 0.0
        train_flow_avg = 0.0
        num_batches = 0

        for batch in tqdm(eval_train_loader, desc="Train Eval"):
            inputs = batch['input'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)

            if self.event_mask:
                activity_patch = inputs.sum(dim=(1, 2))
                low_activity_mask = (activity_patch < 1)
                low_activity_mask = low_activity_mask.unsqueeze(1)
                valid_mask[low_activity_mask] = 0.0

            outputs = self.model(inputs)
            losses = self.criterion(outputs, gt_flow, inputs, valid_mask)

            outliers = calculate_outliers(outputs['flow'], gt_flow, valid_mask, threshold=1.0)

            flow_pred = outputs['flow']
            flow_mag = torch.norm(flow_pred, dim=1)
            flow_max = flow_mag.max().item()
            flow_avg = flow_mag.abs().mean().item()

            for key in train_losses:
                train_losses[key] += losses[key].item()
            train_outliers += outliers
            train_flow_max += flow_max
            train_flow_avg += flow_avg
            num_batches += 1

        for key in train_losses:
            train_losses[key] /= num_batches
        train_outliers /= num_batches
        train_flow_max /= num_batches
        train_flow_avg /= num_batches

        return {
            **train_losses,
            'outliers': train_outliers,
            'flow_max': train_flow_max,
            'flow_avg': train_flow_avg,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'endpoint_loss': 0.0,
            'angular_loss': 0.0,
            'epe_ang_loss': 0.0,
        }
        val_outliers = 0.0
        val_flow_max = 0.0
        val_flow_avg = 0.0
        num_batches = 0
        
        # Store first batch for visualization
        first_batch_vis = None
        
        # Force batch size 1 for validation
        eval_val_loader = torch.utils.data.DataLoader(
            self.val_loader.dataset, batch_size=32, shuffle=False, num_workers=16
        )
        for batch_idx, batch in enumerate(tqdm(eval_val_loader, desc="Validation")):
            # Move to device
            inputs = batch['input'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)

            if self.event_mask:
                activity_patch = inputs.sum(dim=(1,2))
                low_activity_mask = (activity_patch < 1)
                low_activity_mask = low_activity_mask.unsqueeze(1)

                valid_mask[low_activity_mask] = 0.0
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute losses
            losses = self.criterion(outputs, gt_flow, inputs, valid_mask)
            
            # Compute metrics
            outliers = calculate_outliers(outputs['flow'], gt_flow, valid_mask, threshold=1.0)

            flow_pred = outputs['flow']
            flow_mag = torch.norm(flow_pred, dim=1)
            flow_max = flow_mag.max().item()
            flow_avg = flow_mag.abs().mean().item()
            
            for key in val_losses:
                val_losses[key] += losses[key].item()
            val_outliers += outliers

            val_flow_max += flow_max
            val_flow_avg += flow_avg

            num_batches += 1
            
            # Store first batch for visualization
            if batch_idx == 0:
                first_batch_vis = {
                    'inputs': inputs.cpu(),
                    'gt_flow': gt_flow.cpu(),
                    'valid_mask': valid_mask.cpu(),
                    'pred_flow': outputs['flow'].cpu()
                }
        
        # Average
        for key in val_losses:
            val_losses[key] /= num_batches
        val_outliers /= num_batches
        val_flow_max /= num_batches
        val_flow_avg /= num_batches
        
        # Log validation images
        if first_batch_vis is not None:
            visualizations = self._visualize_batch(
                first_batch_vis['inputs'].cpu(),
                first_batch_vis['gt_flow'].cpu(),
                first_batch_vis['valid_mask'].cpu(),
                pred_flow=first_batch_vis['pred_flow'].cpu(),
                max_images=self.config.get('max_images_log', 4)
            )
            
            for vis_name, vis_tensor in visualizations.items():
                if vis_name == 'events' or vis_name == 'valid_mask':
                    # For events and valid mask, vis_tensor is [B*T, 3, H, W]
                    nrow = self.config.get('num_bins', 10)
                else:
                    nrow = 2  # For flow visualizations
                # TensorBoard expects grid of images, so we make a grid
                grid = make_grid(vis_tensor, nrow=nrow, normalize=False, pad_value=1.0)
                self.logger.log_image(f'val/{vis_name}', grid, self.global_step)
            
            # Log flow histograms (u and v components with same scale)
            self._log_flow_histograms(
                first_batch_vis['pred_flow'].cpu(), 
                first_batch_vis['valid_mask'].cpu(), 
                'val', 
                'pred',
                self.global_step
            )
        
        return {
            **val_losses,
            'outliers': val_outliers,
            'flow_max': val_flow_max,
            'flow_avg': val_flow_avg,
        }
    
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth', is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_epe': self.best_val_epe,
            'config': self.config
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
        
        if is_best:
            best_filepath = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_filepath)
            print(f"Saved best model to {best_filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_epe = checkpoint['best_val_epe']
        
        print(f"Loaded checkpoint from {filepath} (epoch {self.epoch})")

    def update_settings(self, epoch: int):
        """Update training settings based on epoch"""
        # Example: Disable event mask after certain epoch
        if self.config.get('event_mask_disable_epoch') is not None:
            if epoch >= self.config['event_mask_disable_epoch'] and self.event_mask:
                self.event_mask = False
                print(f"Epoch {epoch}: Disabled event mask for training")

        if self.config.get('disable_skip_epoch') is not None:
            if epoch >= self.config['disable_skip_epoch'] and self.model.disable_skip is False:
                self.model.disable_skip = True
                print(f"Epoch {epoch}: Disabled skip connections for training")

    
    def train(self, num_epochs: int, resume: Optional[str] = None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            resume: Path to checkpoint to resume from
        """
        # Resume if checkpoint provided
        if resume is not None:
            self.load_checkpoint(resume)
        
        print(f"Starting training from epoch {self.epoch}")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        start_epoch = self.epoch + 1

        for epoch in range(start_epoch, num_epochs+1):
            self.epoch = epoch

            self.update_settings(epoch)

            # Train
            train_metrics = self.train_epoch()

            # Validate
            if self.do_evaluation:
                val_metrics = self.validate()
            else:
                val_metrics = {}

            # Learning rate scheduling
            self.scheduler.step()

            # Log epoch metrics
            print(f"\nEpoch {epoch} Summary:")
            if self.do_evaluation:
                print(f"  Train - Loss: {train_metrics.get('total_loss', 0):.4f}, AngEPE: {train_metrics.get('epe_ang_loss', 0):.4f}, EPE: {train_metrics.get('endpoint_loss', 0):.4f}, Ang: {train_metrics.get('angular_loss', 0):.4f}, Outliers: {train_metrics.get('outliers', 0):.2f}%")
                print(f"  Val   - Loss: {val_metrics.get('total_loss', 0):.4f}, AngEPE: {val_metrics.get('epe_ang_loss', 0):.4f}, EPE: {val_metrics.get('endpoint_loss', 0):.4f}, Ang: {val_metrics.get('angular_loss', 0):.4f}, Outliers: {val_metrics.get('outliers', 0):.2f}%")
            else:
                print("  Evaluation skipped.")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6g}")

            for key, value in train_metrics.items():
                self.logger.log_scalar(f'epoch/train_{key}', value, epoch)
            for key, value in val_metrics.items():
                self.logger.log_scalar(f'epoch/val_{key}', value, epoch)
            self.logger.log_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)

            # Save checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            # Explicitly clear CUDA cache to help prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save best model
            if self.do_evaluation and 'epe_ang_loss' in val_metrics and val_metrics['epe_ang_loss'] < self.best_val_epe:
                self.best_val_epe = val_metrics['epe_ang_loss']
                self.save_checkpoint(is_best=True)
                print(f"  New best validation EPE: {self.best_val_epe:.4f}")
        
        print("Training completed!")
        print(f"Best validation EPE: {self.best_val_epe:.4f}")
