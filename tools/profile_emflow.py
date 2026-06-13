"""Report basic EmFlow model resource counts from a YAML config."""

import argparse
from pathlib import Path
import sys

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from snn.models import EmFlow


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile EmFlow parameter and weight storage counts.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "snn/configs/event_snn_lite.yaml"),
        help="Path to an EmFlow YAML config.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    model = EmFlow(config)
    params = sum(p.numel() for p in model.parameters())

    print("EmFlow profile")
    print(f"  config:       {args.config}")
    print(f"  parameters:   {params:,}")
    print(f"  FP32 weights: {params * 4 / 1024:.2f} KiB")
    print(f"  INT8 weights: {params / 1024:.2f} KiB")
    print(f"  INT4 weights: {params / 2 / 1024:.2f} KiB")


if __name__ == "__main__":
    main()
