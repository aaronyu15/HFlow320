"""Load one HFlow320 sample and print its tensor shapes/statistics."""

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from snn.dataset import OpticalFlowDataset


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
    parser = argparse.ArgumentParser(description="Inspect one HFlow320 dataset sample.")
    parser.add_argument("--data-root", default="data/hflow320/train_set", help="Path to a split directory.")
    parser.add_argument("--index", type=int, default=0, help="Dataset sample index.")
    parser.add_argument("--num-bins", type=int, default=5, help="Number of temporal event bins.")
    parser.add_argument("--bin-interval-us", type=int, default=5000, help="Duration of each event bin in microseconds.")
    parser.add_argument("--use-polarity", action="store_true", help="Use two event polarity channels.")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found under {args.data_root}")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index must be in [0, {len(dataset) - 1}]")

    sample = dataset[args.index]
    events = sample["input"]
    flow = sample["flow"]
    valid = sample["valid_mask"]

    event_pixels = (events.sum(dim=(0, 1)) > 0).sum().item()
    flow_mag = torch.linalg.vector_norm(flow, dim=0)
    valid_pixels = valid.sum().item()
    valid_flow_mag = flow_mag[valid[0] > 0]

    print("HFlow320 sample")
    print(f"  data root:       {args.data_root}")
    print(f"  dataset samples: {len(dataset):,}")
    print(f"  sample index:    {args.index}")
    print(f"  metadata:        {sample['metadata']}")
    print(f"  input shape:     {tuple(events.shape)}  [T, C, H, W]")
    print(f"  flow shape:      {tuple(flow.shape)}    [2, H, W]")
    print(f"  valid shape:     {tuple(valid.shape)}   [1, H, W]")
    print(f"  event pixels:    {event_pixels:,}")
    print(f"  valid pixels:    {int(valid_pixels):,}")

    if valid_flow_mag.numel() > 0:
        print(f"  valid flow mean: {valid_flow_mag.mean().item():.4f} px/frame")
        print(f"  valid flow max:  {valid_flow_mag.max().item():.4f} px/frame")


if __name__ == "__main__":
    main()
