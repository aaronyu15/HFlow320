"""Canonical HFlow320 mask construction and optical-flow metrics."""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


def _normalize_flow_hw2(flow: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow)
    if flow.ndim != 3:
        raise ValueError(f"Expected 3D flow array, got shape {flow.shape}")
    if flow.shape[0] == 2 and flow.shape[-1] != 2:
        return np.transpose(flow, (1, 2, 0))
    if flow.shape[-1] == 2:
        return flow
    raise ValueError(f"Expected CHW or HWC flow with two channels, got {flow.shape}")


def _normalize_mask_hw(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    return mask > 0.5


def resolve_target_interval_bounds(
    flow_h5: h5py.File,
    event_h5: h5py.File,
    frame_idx: int,
    num_frames: Optional[int] = None,
) -> Tuple[int, int]:
    """Return the half-open raw-event slice for one stored target interval."""
    if 'flow/frame_event_start' in flow_h5 and 'flow/frame_event_end' in flow_h5:
        start_idx = int(flow_h5['flow/frame_event_start'][frame_idx])
        end_idx = int(flow_h5['flow/frame_event_end'][frame_idx]) + 1
        return max(0, start_idx), max(start_idx, end_idx)

    all_t = event_h5['events/t']
    if 'flow/event_start' in flow_h5 and 'flow/event_end' in flow_h5:
        start_t = float(flow_h5['flow/event_start'][frame_idx])
        end_t = float(flow_h5['flow/event_end'][frame_idx])
        start_idx = int(np.searchsorted(all_t, start_t, side='left'))
        end_idx = int(np.searchsorted(all_t, end_t, side='left'))
        return max(0, start_idx), max(start_idx, end_idx)

    if len(all_t) == 0:
        return 0, 0

    total_duration = float(all_t[-1] - all_t[0])
    total_frames = max(1, int(num_frames if num_frames is not None else frame_idx + 1))
    frame_interval = total_duration / total_frames
    start_t = float(all_t[0]) + frame_idx * frame_interval
    end_t = float(all_t[0]) + (frame_idx + 1) * frame_interval
    start_idx = int(np.searchsorted(all_t, start_t, side='left'))
    end_idx = int(np.searchsorted(all_t, end_t, side='left'))
    return max(0, start_idx), max(start_idx, end_idx)


def event_mask_from_xy(x: np.ndarray, y: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if len(x) == 0:
        return mask

    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    mask[y[in_bounds], x[in_bounds]] = True
    return mask


def load_target_interval_event_mask(
    event_h5_path: str,
    flow_h5_path: str,
    frame_idx: int,
    height: int,
    width: int,
    num_frames: Optional[int] = None,
) -> Dict[str, np.ndarray | int]:
    with h5py.File(event_h5_path, 'r') as event_h5, h5py.File(flow_h5_path, 'r') as flow_h5:
        start_idx, end_idx = resolve_target_interval_bounds(
            flow_h5=flow_h5,
            event_h5=event_h5,
            frame_idx=frame_idx,
            num_frames=num_frames,
        )
        x = np.asarray(event_h5['events/x'][start_idx:end_idx])
        y = np.asarray(event_h5['events/y'][start_idx:end_idx])

    mask = event_mask_from_xy(x, y, height, width)
    return {
        'mask': mask,
        'event_count': int(len(x)),
        'active_pixels': int(mask.sum()),
        'start_idx': int(start_idx),
        'end_idx': int(end_idx),
    }


def center_crop_mask(mask: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    mask = _normalize_mask_hw(mask)
    crop_h, crop_w = map(int, crop_size)
    height, width = mask.shape
    top = int(round((height - crop_h) / 2.0))
    left = int(round((width - crop_w) / 2.0))
    return mask[top:top + crop_h, left:left + crop_w]


def center_crop_array(array: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    array = np.asarray(array)
    crop_h, crop_w = map(int, crop_size)
    height, width = array.shape[-2:]
    if crop_h > height or crop_w > width:
        raise ValueError(f"Crop {crop_size} exceeds spatial shape {(height, width)}")
    top = int(round((height - crop_h) / 2.0))
    left = int(round((width - crop_w) / 2.0))
    return array[..., top:top + crop_h, left:left + crop_w]


def canonical_mask_hash(mask: np.ndarray) -> str:
    mask = _normalize_mask_hw(mask)
    header = f"{mask.shape[0]}x{mask.shape[1]}:".encode('ascii')
    packed = np.packbits(mask.reshape(-1), bitorder='little').tobytes()
    return hashlib.sha256(header + packed).hexdigest()


def canonical_mask_aggregate_hash(records: list[tuple[str, int, str]]) -> str:
    payload = '\n'.join(
        f"{sequence}\t{frame_idx}\t{mask_hash}"
        for sequence, frame_idx, mask_hash in sorted(records)
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def masked_flow_metrics_numpy(
    pred_flow: np.ndarray,
    gt_flow: np.ndarray,
    valid_mask: np.ndarray,
    event_mask: np.ndarray,
) -> Dict[str, float | int]:
    pred = _normalize_flow_hw2(pred_flow)
    target = _normalize_flow_hw2(gt_flow)
    valid = _normalize_mask_hw(valid_mask)
    active = _normalize_mask_hw(event_mask)

    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch: {pred.shape} vs {target.shape}")
    if valid.shape != active.shape or valid.shape != pred.shape[:2]:
        raise ValueError(
            f"Mask/flow mismatch: valid={valid.shape}, active={active.shape}, flow={pred.shape[:2]}"
        )

    support = valid & active
    valid_pixels = int(valid.sum())
    support_pixels = int(support.sum())
    coverage = 100.0 * support_pixels / valid_pixels if valid_pixels else float('nan')
    if support_pixels == 0:
        return {
            'epe': float('nan'),
            'aae_deg': float('nan'),
            'outlier_pct': float('nan'),
            'valid_pixels': valid_pixels,
            'support_pixels': support_pixels,
            'coverage_pct': coverage,
        }

    error = np.linalg.norm(pred - target, axis=-1)
    target_magnitude = np.linalg.norm(target, axis=-1)
    outlier = (error > 3.0) & (error > 0.05 * target_magnitude)

    pu, pv = pred[..., 0], pred[..., 1]
    gu, gv = target[..., 0], target[..., 1]
    dot = pu * gu + pv * gv + 1.0
    denominator = np.sqrt(pu * pu + pv * pv + 1.0) * np.sqrt(gu * gu + gv * gv + 1.0)
    cosine = np.clip(dot / (denominator + 1e-8), -0.999, 0.999)
    angular_error = np.degrees(np.arccos(cosine))
    angular_support = support & (target_magnitude > 0.1)

    return {
        'epe': float(np.mean(error[support])),
        'aae_deg': float(np.mean(angular_error[angular_support])) if angular_support.any() else float('nan'),
        'outlier_pct': float(np.mean(outlier[support]) * 100.0),
        'valid_pixels': valid_pixels,
        'support_pixels': support_pixels,
        'coverage_pct': coverage,
    }


def common_support_masked_metrics(
    pred_flow: np.ndarray,
    gt_flow: np.ndarray,
    valid_mask: np.ndarray,
    event_mask: np.ndarray,
    crop_size: Tuple[int, int],
) -> Dict[str, float | int | str]:
    pred = np.transpose(_normalize_flow_hw2(pred_flow), (2, 0, 1))
    target = np.transpose(_normalize_flow_hw2(gt_flow), (2, 0, 1))
    valid = center_crop_mask(valid_mask, crop_size)
    active = center_crop_mask(event_mask, crop_size)
    metrics = masked_flow_metrics_numpy(
        center_crop_array(pred, crop_size),
        center_crop_array(target, crop_size),
        valid,
        active,
    )
    metrics['canonical_mask_hash'] = canonical_mask_hash(valid & active)
    return metrics
