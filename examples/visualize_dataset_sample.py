"""Save a simple HFlow320 event/flow visualization panel."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from snn.dataset import OpticalFlowDataset
from snn.utils.visualization import flow_to_color


def event_image(event_bins: np.ndarray) -> np.ndarray:
    """Convert event bins [T, C, H, W] into an RGB image."""
    if event_bins.ndim != 4:
        raise ValueError(f"Expected event bins with shape [T, C, H, W], got {event_bins.shape}")

    if event_bins.shape[1] >= 2:
        pos = event_bins[:, 0].sum(axis=0)
        neg = event_bins[:, 1].sum(axis=0)
        img = np.zeros((event_bins.shape[2], event_bins.shape[3], 3), dtype=np.float32)
        if pos.max() > 0:
            img[..., 0] = pos / pos.max()
        if neg.max() > 0:
            img[..., 2] = neg / neg.max()
        return np.clip(img, 0.0, 1.0)

    activity = event_bins.sum(axis=(0, 1))
    if activity.max() > 0:
        activity = activity / activity.max()
    return np.repeat(activity[..., None], 3, axis=2)


def build_dataset(args: argparse.Namespace) -> OpticalFlowDataset:
    return OpticalFlowDataset(
        config={
            "data_root": args.data_root,
            "num_bins": args.num_bins,
            "bin_interval_us": args.bin_interval_us,
            "use_polarity": args.use_polarity,
            "data_size": (args.height, args.width),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one HFlow320 sample.")
    parser.add_argument("--data-root", default="data/hflow320/train_set", help="Path to a split directory.")
    parser.add_argument("--index", type=int, default=0, help="Dataset sample index.")
    parser.add_argument("--num-bins", type=int, default=5, help="Number of temporal event bins.")
    parser.add_argument("--bin-interval-us", type=int, default=5000, help="Duration of each event bin in microseconds.")
    parser.add_argument("--use-polarity", action="store_true", help="Use two event polarity channels.")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--output", default="sample_preview.png", help="Output PNG path.")
    args = parser.parse_args()

    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found under {args.data_root}")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index must be in [0, {len(dataset) - 1}]")

    sample = dataset[args.index]
    events = sample["input"].numpy()
    flow = sample["flow"].numpy()
    valid = sample["valid_mask"].numpy()[0]
    event_active = (events.sum(axis=(0, 1)) > 0).astype(np.float32)

    event_rgb = event_image(events)
    flow_rgb = flow_to_color(flow) / 255.0
    masked_flow_rgb = flow_to_color(flow * valid[None, ...]) / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    panels = [
        ("Events", event_rgb),
        ("Flow", flow_rgb),
        ("Masked flow", masked_flow_rgb),
        ("Event-active mask", event_active),
    ]

    for ax, (title, image) in zip(axes, panels):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    meta = sample["metadata"]
    fig.suptitle(f"{meta.get('sequence', 'unknown')} | frame {meta.get('index', args.index)}", fontsize=10)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
