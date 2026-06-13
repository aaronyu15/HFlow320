"""Smoke tests for the cleaned HFlow320/EmFlow repository.

This script is intentionally small: it verifies that the public-facing modules
can be imported, that EmFlow can run a dummy forward pass from the default YAML
configuration, and optionally that one HFlow320 dataset sample can be loaded.
"""

import argparse
from pathlib import Path
import sys

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from snn.dataset import OpticalFlowDataset
from snn.models import EmFlow


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_model(config: dict, device: str) -> None:
    model = EmFlow(config).to(device)
    model.eval()

    batch_size = 1
    num_bins = int(config.get("num_bins", 5))
    channels = 2 if config.get("use_polarity", False) else 1
    height, width = config.get("camera_size", [320, 320])

    x = torch.zeros(batch_size, num_bins, channels, height, width, device=device)
    with torch.no_grad():
        output = model(x)

    flow = output["flow"]
    expected_shape = (batch_size, 2, height, width)
    if tuple(flow.shape) != expected_shape:
        raise AssertionError(f"Expected flow shape {expected_shape}, got {tuple(flow.shape)}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Model OK: {params:,} parameters, output shape {tuple(flow.shape)}")


def test_dataset(data_root: str, config: dict) -> None:
    dataset_config = dict(config)
    dataset_config["data_root"] = data_root
    dataset_config["num_workers"] = 0

    dataset = OpticalFlowDataset(config=dataset_config)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found under {data_root}")

    sample = dataset[0]
    print("Dataset OK:")
    print(f"  samples: {len(dataset):,}")
    print(f"  input:   {tuple(sample['input'].shape)}")
    print(f"  flow:    {tuple(sample['flow'].shape)}")
    print(f"  mask:    {tuple(sample['valid_mask'].shape)}")
    print(f"  first:   {sample['metadata']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HFlow320/EmFlow smoke tests.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "snn/configs/event_snn_lite.yaml"),
        help="Path to an EmFlow YAML config.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional HFlow320 split directory, such as data/hflow320/train_set.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

    config = load_config(Path(args.config))
    test_model(config, device)

    if args.data_root is not None:
        test_dataset(args.data_root, config)
    else:
        print("Dataset test skipped. Pass --data-root to test HFlow320 loading.")


if __name__ == "__main__":
    main()
