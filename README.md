# HFlow320 and EmFlow

This repository contains the core software for working with HFlow320 and
EmFlow.

HFlow320 is a compact synthetic event-based optical-flow dataset focused on
human motion at 320 x 320 resolution. Each sample provides event data, optical
flow ground truth, and validity masks. The dataset is intended for rapid
iteration on event-based optical-flow methods, especially compact software
models and ablation studies.

EmFlow is a lightweight spiking neural network baseline for event-based optical
flow. It is included here as a reference implementation showing how to load
HFlow320, convert event streams into temporal bins, train a compact model, and
evaluate masked endpoint error and angular error.

## Repository Layout

```text
snn/
  configs/              EmFlow configuration files.
  dataset/              HFlow320 HDF5 dataset loader.
  models/               EmFlow model, spiking layers, and quantized layers.
  training/             Losses, metrics, and trainer.
  utils/                Logging and flow-color utilities.
train.py                Training entry point.
evaluate.py             Evaluation entry point.
utils.py                Shared config/model helpers.
examples/smoke_test.py  Import, model, and optional dataset smoke test.
examples/load_dataset_sample.py
                         Print tensor shapes and sample statistics.
examples/visualize_dataset_sample.py
                         Save an event/flow preview panel.
tools/profile_emflow.py Small parameter and weight-storage summary script.
```

Generated files such as checkpoints, logs, datasets, and outputs are ignored by
Git.

## Installation

Create or activate a Python environment with PyTorch, then install the Python
dependencies:

```bash
pip install -r requirements.txt
```

On the author's machine, the code is tested with the `flow` conda environment:

```bash
conda activate flow
```

## Dataset Layout

Create/extract the dataset inside the repository-local `data/hflow320/`
directory. The placeholder folders are included so the repository is
self-contained, while the actual dataset files remain ignored by Git.

The HFlow320 loader expects one split directory containing sequence folders.
Each sequence folder should contain `events.h5` and `flow.h5`:

```text
data/hflow320/
  train_set/
    sequence_name/
      events.h5
      flow.h5
  valid_set/
    sequence_name/
      events.h5
      flow.h5
  test_set/
    sequence_name/
      events.h5
      flow.h5
```

The main fields used by the loader are:

```text
events.h5/events/x
events.h5/events/y
events.h5/events/t
events.h5/events/p
flow.h5/flow/forward
flow.h5/flow/valid
flow.h5/flow/frame_event_start
flow.h5/flow/frame_event_end
flow.h5/flow/event_start
flow.h5/flow/event_end
```

`OpticalFlowDataset` slices events ending at the target flow timestamp, bins
them into a tensor with shape `[T, C, H, W]`, and returns:

```python
{
    "input": event_bins,      # [T, C, H, W]
    "flow": flow,             # [2, H, W]
    "valid_mask": valid_mask, # [1, H, W]
    "metadata": {...},
}
```

Minimal dataset example:

```python
from snn.dataset import OpticalFlowDataset

dataset = OpticalFlowDataset(config={
    "data_root": "data/hflow320/train_set",
    "num_bins": 5,
    "bin_interval_us": 5000,
    "use_polarity": False,
    "data_size": (320, 320),
})

sample = dataset[0]
print(sample["input"].shape, sample["flow"].shape, sample["valid_mask"].shape)
```

The `examples/` directory also contains small scripts for getting started:

```bash
python examples/load_dataset_sample.py \
  --data-root data/hflow320/train_set \
  --index 0

python examples/visualize_dataset_sample.py \
  --data-root data/hflow320/train_set \
  --index 0 \
  --output outputs/sample_preview.png
```

The visualization example saves a four-panel Matplotlib image showing the event
activity image, optical flow color image, masked optical flow, and event-active
mask.

## Smoke Test

Run the import/model smoke test from the repository root:

```bash
python examples/smoke_test.py --device cpu
```

To also test dataset loading, pass a split path:

```bash
python examples/smoke_test.py \
  --data-root data/hflow320/train_set \
  --device cpu
```

## Training

The default EmFlow configuration is `snn/configs/event_snn_lite.yaml`.

```bash
python train.py \
  --config snn/configs/event_snn_lite.yaml \
  --train-data-root data/hflow320/train_set \
  --val-data-root data/hflow320/valid_set \
  --name emflow_hflow320
```

By default, checkpoints are written under `checkpoints/` and TensorBoard logs
under `logs/`.

## Evaluation

Evaluate a trained checkpoint on a held-out split:

```bash
python evaluate.py \
  --checkpoint checkpoints/emflow_hflow320/best_model.pth \
  --config snn/configs/event_snn_lite.yaml \
  --data-root data/hflow320/test_set \
  --name emflow_hflow320_test
```

The evaluation script reports masked endpoint error (EPE), average angular error
(AAE), and related metrics using the dataset validity mask and event activity.

## EmFlow Profile

To print the parameter count and basic weight-storage sizes:

```bash
python tools/profile_emflow.py --config snn/configs/event_snn_lite.yaml
```

## Notes for Researchers

HFlow320 is not meant to replace broad driving or general-scene event-flow
benchmarks. It is a controlled human-motion dataset intended to make compact
model development, ablation, and reproducible comparison easier. When reporting
results, please include the temporal window, number of event bins, bin duration,
mask policy, checkpoint-selection rule, and any changes to the default config.

If you use HFlow320 or EmFlow in research, cite the accompanying paper and the
dataset release.
