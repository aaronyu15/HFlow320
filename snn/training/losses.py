import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import sys
import traceback

def epe (pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
    # [B, 2, H, W] -> [B, 1, H, W]
    return torch.norm(pred_flow - gt_flow, p=2, dim=1, keepdim=True)

def apply_mask(flow: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        if mask.shape != flow.shape:
            print(f"Mask shape {mask.shape} does not match flow shape {flow.shape}")
            traceback.print_stack()
            sys.exit(1)
        return flow * mask, mask
    else:
        mask = torch.ones_like(flow)
        return flow, mask


def endpoint_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                   mask: Optional[torch.Tensor] = None, return_vec: bool = False) -> torch.Tensor:
    error = epe(pred_flow, gt_flow)
    error, mask = apply_mask(error, mask)

    if return_vec:
        return error, mask
    else:
        return error.sum() / (mask.sum() + 1e-8)


def angular_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                  mask: Optional[torch.Tensor] = None, return_vec: bool = False) -> torch.Tensor:
    # Extract u and v components
    pu, pv = pred_flow[:, 0], pred_flow[:, 1]
    gu, gv = gt_flow[:, 0], gt_flow[:, 1]

    # 3D dot: (u,v,1)·(u',v',1) = u*u' + v*v' + 1
    dot = pu * gu + pv * gv + 1.0

    # 3D norms: sqrt(u^2 + v^2 + 1)
    pnorm = torch.sqrt(pu * pu + pv * pv + 1.0)
    gnorm = torch.sqrt(gu * gu + gv * gv + 1.0)

    cos = dot / (pnorm * gnorm + 1e-8)
    cos = torch.clamp(cos, -0.999, 0.999)

    ang = torch.acos(cos) # radians 
    ang = ang.unsqueeze(1)  # [B, 1, H, W]

    nonzero_mask = (gt_flow.norm(dim=1) > 0.1).float().unsqueeze(1)  # [B, 1, H, W]
    mask = nonzero_mask if mask is None else (mask * nonzero_mask)
    ang, mask = apply_mask(ang, mask)

    if return_vec:
        return ang, mask
    return ang.sum() / (mask.sum() + 1e-8)

def epe_weighted_angular_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor, inputs: torch.Tensor,
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute angular error weighted by EPE magnitude and event activity.
    Regions with more events and larger errors contribute more to the loss.
    """
    ang_error, ang_mask = angular_error(pred_flow, gt_flow, mask, return_vec=True)
    epe_error, epe_mask = endpoint_error(pred_flow, gt_flow, mask, return_vec=True)

    combined_mask = ang_mask * epe_mask 
    
    # Sum event activity over time and polarity bins to get [B, 1, H, W]
    event_activity = inputs.sum(dim=1).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # Normalize event activity to [0, 1] range per sample to prevent extreme weighting
    # This ensures samples with different overall event counts are treated fairly
    max_activity = event_activity.view(event_activity.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
    event_weight = event_activity / (max_activity + 1e-8)
    
    # Combine weights: angular error * EPE error * event activity
    weighted_ang_error = ang_error * epe_error * event_weight 

    # Normalize by sum of weights (not just mask count) to account for variable event activity
    total_weight = (event_weight * combined_mask).sum()
    
    return weighted_ang_error.sum() / (total_weight + 1e-8)




class CombinedLoss(nn.Module):
    """
    Combined loss for SNN optical flow training
    """
    def __init__(
        self,
        endpoint_weight: float = 1.0,
        angular_weight: float = 0.0,
        epe_ang_weight: float = 0.0,
        vertical_weight: float = 0.0,
    ):
        super().__init__()
        self.endpoint_weight = endpoint_weight
        self.angular_weight = angular_weight
        self.epe_ang_weight = epe_ang_weight

        self.vertical_weight = vertical_weight

    def forward(
        self,
        outputs: Dict,
        gt_flow: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            outputs: Model outputs with 'flow', 'flow_pyramid', 'spike_stats'
            gt_flow: Ground truth flow
            mask: Valid mask
            model: Model for quantization loss
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        losses['endpoint_loss'] = endpoint_error(outputs['flow'], gt_flow, mask)
        losses['angular_loss'] = angular_error(outputs['flow'], gt_flow, mask)
        losses['epe_ang_loss'] = epe_weighted_angular_error(outputs['flow'], gt_flow, inputs, mask)

        losses['total_loss'] = (
            self.endpoint_weight * losses['endpoint_loss'] +
            self.angular_weight * losses['angular_loss'] +
            self.epe_ang_weight * losses['epe_ang_loss']
        )
        
        return losses

# Metrics
def calculate_outliers(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       threshold: float = 3.0) -> float:

    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=1, keepdim=True))

    epe = torch.norm(pred_flow - gt_flow, p=2, dim=1, keepdim=True)
    
    outliers = (epe > threshold) & (epe > 0.05 * gt_mag)

    outliers, mask = apply_mask(outliers.float(), mask)

    return (outliers.sum() / (mask.sum() + 1e-8) * 100).item()



