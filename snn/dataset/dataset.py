"""
Optical Flow Dataset Loader
Loads event data and optical flow from blink_sim outputs
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional
import h5py
import sys


class OpticalFlowDataset(Dataset):
    """
    Dataset for loading optical flow data from blink_sim output
    
    Expected structure:
    data/hflow320/train_set/
        sequence_name_0/
            events.h5
            flow.h5
    """
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Args:
            config: Configuration dictionary containing all parameters. If provided,
                   individual arguments will override config values.
            data_root: Split directory containing sequence folders.
            use_events: Use event data (vs RGB images)
            num_bins: Number of temporal bins for event representation
            data_size: Full image size (height, width)
            flow_clip_range: Optional (min, max) to clip flow values
            min_event_count: Minimum total event count for a sample (retries if below threshold)
        """
        # Use config as base, with individual args as overrides
        if config is None:
            config = {}
        
        self.data_root = config.get('data_root', 'data/hflow320/train_set')
        self.use_events = config.get('use_events', True)
        self.num_bins = config.get('num_bins', 5)
        self.bin_interval_us = config.get('bin_interval_us', 10000)
        self.use_polarity = config.get('use_polarity', False)
        self.data_size = config.get('data_size', (320, 320))

        self.flow_clip_range = config.get('flow_clip_range', None)
        
        # Find all sequences
        self.sequences = self._find_sequences()
        
        # Build sample list
        self.samples = self._build_sample_list()

    
    def _find_sequences(self) -> List[Path]:
        split_dir = Path(self.data_root)
        
        sequences = []
        for seq_dir in sorted(split_dir.iterdir()):
            if seq_dir.is_dir() and (seq_dir / 'flow.h5').exists():
                sequences.append(seq_dir)
        
        return sequences
    
    def _build_sample_list(self) -> List[Dict]:
        samples = []

        for seq_dir in self.sequences:
            flow_h5_file = seq_dir / 'flow.h5'

            if flow_h5_file.exists():
                with h5py.File(flow_h5_file, 'r') as f:
                    num_frames = f['flow/forward'].shape[0]

            if self.use_events:
                event_h5_file = seq_dir / 'events.h5'
                if event_h5_file.exists():
                    for frame_idx in range(num_frames):
                        sample = {
                            'sequence': seq_dir.name,
                            'event_h5_path': event_h5_file,
                            'flow_h5_path': flow_h5_file,
                            'index': frame_idx,
                            'num_frames': num_frames,
                        }

                        samples.append(sample)

        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample
        
        Returns:
            Dictionary containing:
                - 'input': Event representation or RGB image [C, H, W]
                - 'flow': Optical flow ground truth [2, H, W]
                - 'valid_mask': Valid flow mask [1, H, W]
                - 'metadata': Dictionary with sequence info
        """
        sample_info = self.samples[idx]
        frame_idx = sample_info['index']
        
        # Load optical flow and per-frame timing info
        with h5py.File(sample_info['flow_h5_path'], 'r') as f:
            flow_data = f['flow/forward'][frame_idx]          # [H, W, 2]
            valid_data = f['flow/valid'][frame_idx]           # [H, W, 1]
            event_end_idx = int(f['flow/frame_event_end'][frame_idx])  # last event index for this frame
            event_start_idx = int(f['flow/frame_event_start'][frame_idx])  # first event index for this frame
            event_end_time = float(f['flow/event_end'][frame_idx])      # absolute end time (us)
            event_start_time = float(f['flow/event_start'][frame_idx])  # absolute start time (us)
        
        flow = torch.from_numpy(flow_data).permute(2, 0, 1).float()        # [2, H, W]
        valid_mask = torch.from_numpy(valid_data).permute(2, 0, 1).float()  # [1, H, W]

        try:
            with h5py.File(sample_info['event_h5_path'], 'r') as f:
                # Use event_end_time as t1, and window backwards
                time_window_us = self.num_bins * self.bin_interval_us
                t1 = event_end_time
                t0 = t1 - time_window_us

                # Use frame_event_end as a tight upper bound to avoid scanning the full stream
                all_t_window = f['events/t'][:event_end_idx + 1]
                start_idx = np.searchsorted(all_t_window, t0, side='left')
                end_idx = event_end_idx + 1  # slice end (exclusive)

                x = np.array(f['events/x'][start_idx:end_idx])
                y = np.array(f['events/y'][start_idx:end_idx])
                t = np.array(f['events/t'][start_idx:end_idx])
                p = np.array(f['events/p'][start_idx:end_idx])

                p = np.where(p == 0, -1, 1)

                events = np.column_stack([x, y, t, p]).astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to load events for frame {frame_idx}: {e}")
            sys.exit(1)
            

        input_tensor = self._events_to_voxel_grid(events)  # [num_bins, C, H, W]
        
        # Note: valid_mask was already created when loading flow
        # If not created (shouldn't happen), create default
        if 'valid_mask' not in locals():
            valid_mask = torch.ones(1, flow.shape[1], flow.shape[2])


        # Apply flow clipping if specified
        if self.flow_clip_range is not None:
            flow = torch.clamp(flow, self.flow_clip_range[0], self.flow_clip_range[1])
    
        return {
            'input': input_tensor,
            'flow': flow,
            'valid_mask': valid_mask,
            'metadata': {
                'sequence': sample_info['sequence'],
                'index': sample_info['index']
            }
        }
    
    def _events_to_voxel_grid(self, events: np.ndarray) -> torch.Tensor:
        """
        Convert event array to voxel grid representation
        
        Events format: [N, 4] where columns are [x, y, t, p]
        - x, y: pixel coordinates
        - t: timestamp
        - p: polarity (+1 or -1)
        
        Returns:
            Voxel grid [num_bins, 2, H, W] for EmFlow (polarity-separated)
            OR [num_bins, H, W] for other models (mixed polarity)
        """
        if len(events) == 0:
            # Return zeros if no events - use polarity-separated format
            if self.use_polarity:
                return torch.zeros(self.num_bins, 2, 320, 320)  # Default size
            else:
                return torch.zeros(self.num_bins, 1, 320, 320)  # Default size      
        
        # Parse events
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2]
        p = events[:, 3]
        
        # Get image dimensions
        height = self.data_size[0]
        width = self.data_size[1]
        
        # Normalize timestamps to [0, num_bins)
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min) * (self.num_bins - 1e-6)
        else:
            t_norm = np.zeros_like(t)
        
        # Create voxel grid with separate polarity channels [num_bins, 2, H, W]
        # Channel 0 = positive events, Channel 1 = negative events
        if self.use_polarity:
            voxel_grid = np.zeros((self.num_bins, 2, height, width), dtype=np.float32)
        else:
            voxel_grid = np.zeros((self.num_bins, 1, height, width), dtype=np.float32)
        
        # Distribute events into temporal bins with polarity separation
        for i in range(len(events)):
            bin_idx = int(t_norm[i])
            if 0 <= bin_idx < self.num_bins:
                if 0 <= x[i] < width and 0 <= y[i] < height:
                    if self.use_polarity:
                        pol_idx = 0 if p[i] > 0 else 1  # Channel 0 = positive, 1 = negative
                    else:
                        pol_idx = 0 
                    voxel_grid[bin_idx, pol_idx, y[i], x[i]] += 1.0
        
        return torch.from_numpy(voxel_grid)
