"""Exercise the canonical HFlow320 mask and metric helpers."""

from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hflow320_eval_utils import (
    canonical_mask_hash,
    center_crop_mask,
    common_support_masked_metrics,
    event_mask_from_xy,
    load_target_interval_event_mask,
    masked_flow_metrics_numpy,
)


def main() -> None:
    active = event_mask_from_xy(
        x=np.array([0, 3, 3, 9]),
        y=np.array([0, 1, 1, 9]),
        height=4,
        width=5,
    )
    assert active.shape == (4, 5)
    assert int(active.sum()) == 2
    assert active[0, 0] and active[1, 3]

    cropped = center_crop_mask(active, (2, 3))
    assert cropped.shape == (2, 3)
    assert int(cropped.sum()) == 1

    prediction = np.zeros((2, 4, 5), dtype=np.float32)
    target = np.zeros((2, 4, 5), dtype=np.float32)
    target[0, 0, 0] = 4.0
    valid = np.ones((4, 5), dtype=np.float32)
    metrics = masked_flow_metrics_numpy(prediction, target, valid, active)
    assert metrics['support_pixels'] == 2
    assert abs(metrics['epe'] - 2.0) < 1e-6
    assert abs(metrics['outlier_pct'] - 50.0) < 1e-6

    common = common_support_masked_metrics(prediction, target, valid, active, (2, 3))
    assert common['support_pixels'] == 1
    assert common['canonical_mask_hash'] == canonical_mask_hash(cropped)

    with tempfile.TemporaryDirectory() as directory:
        event_path = Path(directory) / 'events.h5'
        flow_path = Path(directory) / 'flow.h5'
        with h5py.File(event_path, 'w') as event_h5:
            events = event_h5.create_group('events')
            events.create_dataset('x', data=np.array([0, 2, 3, 4], dtype=np.uint16))
            events.create_dataset('y', data=np.array([0, 1, 2, 3], dtype=np.uint16))
            events.create_dataset('t', data=np.array([10, 20, 30, 40], dtype=np.uint64))
        with h5py.File(flow_path, 'w') as flow_h5:
            flow = flow_h5.create_group('flow')
            flow.create_dataset('frame_event_start', data=np.array([1], dtype=np.uint64))
            flow.create_dataset('frame_event_end', data=np.array([2], dtype=np.uint64))

        interval = load_target_interval_event_mask(
            str(event_path),
            str(flow_path),
            frame_idx=0,
            height=4,
            width=5,
            num_frames=1,
        )
        assert interval['event_count'] == 2
        assert interval['mask'][1, 2] and interval['mask'][2, 3]

    print('canonical_mask_selfcheck: PASS')


if __name__ == '__main__':
    main()
