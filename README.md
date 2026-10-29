# HFlow320: A Human-Motion Event-Based Optical Flow Dataset for Compact Model Evaluation

![HFlow320 preview](hflow320_preview.gif)

[Dataset and RGB files](https://doi.org/10.5281/zenodo.19669509)

HFlow320 is a synthetic event-based optical-flow dataset centered on single-actor human motion at 320 × 320 resolution. Each sequence contains raw events, dense forward-flow labels, and flow-validity masks. Optional RGB images are provided separately for visualization and multimodal experiments.

The fixed splits contain 100 training sequences with 22,160 flow frames, 20 validation sequences with 4,466 frames, and 24 test sequences with 5,563 frames. Five character identities appear in training and validation, while one identity is held out for testing. The dataset is intended for controlled experiments with compact event-flow models; it is not a substitute for real-sensor benchmarks or a broad human-activity dataset.

EmFlow is included as a 10,008-parameter spiking baseline. The repository provides its model definition, training configuration, HDF5 loader, canonical evaluation support, and small dataset examples.

## Repository layout

```text
snn/configs/                    EmFlow configurations
snn/dataset/                    HFlow320 HDF5 loader
snn/models/                     EmFlow and quantized layers
snn/training/                   losses, metrics, and trainer
train.py                        training entry point
evaluate.py                     evaluation entry point
hflow320_eval_utils.py          canonical mask and metric definitions
examples/                       loading, visualization, and smoke tests
tools/canonical_mask_selfcheck.py
tools/profile_emflow.py
```

## Installation

The reported experiments used Python 3.10.6, PyTorch 2.10.0 with ROCm 7.1, torchvision 0.25.0, and an AMD Instinct MI210 GPU. Install the appropriate PyTorch build for the local accelerator, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset layout

Extract the event/flow archive under `data/hflow320/`:

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

The loader uses these fields:

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

A sample contains an event tensor, dense flow, the stored flow-valid mask, and sequence metadata:

```python
from snn.dataset import OpticalFlowDataset

dataset = OpticalFlowDataset(config={
    "data_root": "data/hflow320/train_set",
    "num_bins": 5,
    "fixed_window_us": 33333,
    "use_polarity": False,
    "data_size": (320, 320),
})

sample = dataset[0]
print(sample["input"].shape)       # [5, 1, 320, 320]
print(sample["flow"].shape)        # [2, 320, 320]
print(sample["valid_mask"].shape)  # [1, 320, 320]
```

The examples can inspect or visualize one sample:

```bash
python examples/load_dataset_sample.py \
  --data-root data/hflow320/train_set \
  --index 0

python examples/visualize_dataset_sample.py \
  --data-root data/hflow320/train_set \
  --index 0 \
  --output outputs/sample_preview.png
```

## Primary benchmark protocol

Each target is the forward displacement between consecutive 30 FPS frames. The model input ends at the target endpoint and spans 33,333 μs. EmFlow divides this window into five temporal bins.

The canonical evaluation mask is the intersection of the stored flow-valid mask and the pixels containing at least one raw event during the stored target interval. It is constructed from the inclusive `frame_event_start` and `frame_event_end` indices before voxelization. The first target in every test sequence is omitted, leaving 5,539 evaluated frames. Results are unweighted means of finite per-frame values at the original 320 × 320 coordinates.

The evaluator reports endpoint error (EPE), the combined absolute/relative outlier rate `O_3,5`, average angular error (AAE), frame count, and canonical-mask coverage. AAE excludes target magnitudes at or below 0.1 px. Run the mask and metric self-check with:

```bash
python tools/canonical_mask_selfcheck.py
```

The optional `--common-crop 256 256` argument applies the shared center crop used for descriptive reference-model comparisons. Those values should not be mixed with the full-resolution primary benchmark.

## Smoke test

Run a model forward pass from the repository root:

```bash
python examples/smoke_test.py --device cpu
```

Pass a split path to test data loading as well:

```bash
python examples/smoke_test.py \
  --data-root data/hflow320/train_set \
  --device cpu
```

## Training

Train the three baseline seeds with the final epoch-50 checkpoint as the reporting checkpoint:

```bash
for seed in 1 2 3; do
  python train.py \
    --config snn/configs/event_snn_lite.yaml \
    --train-data-root data/hflow320/train_set \
    --val-data-root data/hflow320/valid_set \
    --seed "$seed" \
    --name "emflow_seed${seed}"
done
```

Checkpoints are written under `checkpoints/`, and TensorBoard logs are written under `logs/`.

## Evaluation

Evaluate each final checkpoint on the primary test protocol:

```bash
for seed in 1 2 3; do
  python evaluate.py \
    --checkpoint "checkpoints/emflow_seed${seed}/checkpoint_epoch_50.pth" \
    --config snn/configs/event_snn_lite.yaml \
    --data-root data/hflow320/test_set \
    --eval-window-us 33333 \
    --skip-first-frame \
    --no-log \
    --name "emflow_seed${seed}_test"
done
```

Each run writes a summary and `per_sample_metrics.csv` under `logs/<name>/`. Across the three reported seeds, EmFlow obtains 2.3029 ± 0.0108 px EPE, 24.21 ± 0.30% `O_3,5`, and 34.11° ± 0.01° AAE. These uncertainties are the sample standard deviations of the three run-level means. The mean canonical-mask coverage is 4.92%.

## Model profile

```bash
python tools/profile_emflow.py --config snn/configs/event_snn_lite.yaml
```

The five-bin EmFlow model has 10,008 trainable parameters. The paper reports 175.1 million convolutional multiply-accumulate operations per prediction; interpolation, additions, threshold comparisons, and memory traffic are excluded from that count.

## Licenses

The source code in this repository is distributed under the [MIT License](LICENSE). The HFlow320 dataset deposited on Zenodo is distributed under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

HFlow320 was created with resources from [Blender](https://www.blender.org/), [BlenderProc](https://github.com/DLR-RM/BlenderProc), [BlinkSim/BlinkFlow](https://github.com/zju3dv/blink_sim), [DVS-Voltmeter](https://github.com/Lynn0306/DVS-Voltmeter), [MakeHuman](https://static.makehumancommunity.org/makehuman.html), [Poly Haven](https://polyhaven.com/hdris), and the [CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/), including its [MotionBuilder-friendly BVH conversion](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-motionbuilder-friendly-bvh-conversion-release-of-cmus-motion-capture-database).

If you use HFlow320 or EmFlow, please cite the dataset DOI and the accompanying paper.
