"""
Visualization utilities for optical flow
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path
from typing import Optional, Tuple


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow: np.ndarray, max_flow: Optional[float] = None, clip_flow: Optional[float] = None) -> np.ndarray:
    """
    Convert optical flow to color visualization using the Middlebury color scheme

    Args:
        flow: Flow array [H, W, 2] or [2, H, W]
        max_flow: Maximum flow magnitude for normalization (used when provided)
        clip_flow: Clip maximum of flow values before normalization

    Returns:
        Color image [H, W, 3] in range [0, 255]
    """
    # Convert to [H, W, 2] format
    if flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))
    
    assert flow.ndim == 3, 'input flow must have three dimensions'
    assert flow.shape[2] == 2, 'input flow must have shape [H,W,2]'
    
    if clip_flow is not None:
        flow = np.clip(flow, -clip_flow, clip_flow)

    u = flow[:, :, 0]
    v = flow[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))

    # Respect caller-provided max_flow to keep consistent scaling across frames/maps
    if max_flow is not None and max_flow > 0:
        rad_max = max_flow
    else:
        rad_max = np.max(rad)

    epsilon = 1e-5
    norm = rad_max + epsilon
    u = u / norm
    v = v / norm
    
    return flow_uv_to_colors(u, v, convert_to_bgr=False)


def visualize_flow(flow: torch.Tensor, max_flow: Optional[float] = None) -> np.ndarray:
    """
    Visualize flow tensor
    
    Args:
        flow: Flow tensor [2, H, W] or [B, 2, H, W]
        max_flow: Maximum flow for normalization
    
    Returns:
        Color visualization [H, W, 3] or [B, H, W, 3]
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    
    if flow.ndim == 4:
        # Batch of flows
        batch_size = flow.shape[0]
        visualizations = []
        for i in range(batch_size):
            vis = flow_to_color(flow[i], max_flow)
            visualizations.append(vis)
        return np.stack(visualizations)
    else:
        # Single flow
        return flow_to_color(flow, max_flow)




def visualize_events(event_voxel: np.ndarray, brightness_scale: float = 0.5) -> np.ndarray:
    """
    Visualize event voxel grid as RGB image (red=positive, blue=negative events)
    
    Args:
        event_voxel: Event voxel grid in one of two formats:
            - [num_bins, 2, H, W] - polarity-separated (channel 0=positive, 1=negative)
            - [num_bins, H, W] - old format with signed values
        brightness_scale: Scale factor for brightness (default 0.5 gives 2x brightness boost)
    
    Returns:
        RGB image [H, W, 3] in range [0, 1]
    """
    # Convert to numpy if needed
    if isinstance(event_voxel, torch.Tensor):
        event_voxel = event_voxel.detach().cpu().numpy()
    
    if event_voxel.shape[2] == 2:
        event_sum = event_voxel.sum(axis=0)  # [2, H, W]
        pos_events = event_sum[0]  # Positive events
        neg_events = event_sum[1]  # Negative events
    else:
        event_sum = event_voxel.sum(axis=0)  # [2, H, W]
        pos_events = event_sum[0]  # Positive events
        neg_events = np.zeros_like(pos_events)  # No negative events
    
    h, w = pos_events.shape
    event_rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # Positive events -> red channel
    if pos_events.max() > 0:
        event_rgb[:, :, 0] = pos_events / (pos_events.max() * brightness_scale)
    
    # Negative events -> blue channel
    if neg_events.max() > 0:
        event_rgb[:, :, 2] = neg_events / (neg_events.max() * brightness_scale)
    
    # Clip to valid range
    event_rgb = np.clip(event_rgb, 0, 1)
    
    return event_rgb

