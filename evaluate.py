"""
Evaluation script for trained SNN models.

Supports both full-precision and quantized (PTQ) models.
Logs TensorBoard visualizations (events, flow, masks) alongside numeric metrics.

Usage:
    # Evaluate full-precision model (config embedded in checkpoint)
    python evaluate.py --checkpoint checkpoints/teacher_10000u/best_model.pth

    # Evaluate quantized model (config contains quantized: true)
    python evaluate.py \
        --checkpoint checkpoints/ptq_8bit/ptq_model.pth \
        --config snn/configs/event_snn_lite_8bit.yaml

    # Integer-only simulation for quantized models
    python evaluate.py \
        --checkpoint checkpoints/teacher_nod1/ptq_model.pth \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --name 02_no_d1 \
        --integer-sim

    # Custom data root and TensorBoard log dir
    python evaluate.py \
        --checkpoint checkpoints/teacher_10000u/best_model.pth \
        --data-root data/hflow320/test_set \
        --log-dir logs/eval_teacher
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from snn.models import EmFlow
from snn.dataset import OpticalFlowDataset
#from comparisons.dsec.dsec_dataset import DSECOpticalFlowDataset as OpticalFlowDataset
#from comparisons.mvsec.mvsec_dataset import MVSECDataset as OpticalFlowDataset
from snn.training import endpoint_error, calculate_outliers, angular_error, epe_weighted_angular_error
from snn.utils.logger import Logger
from snn.utils.visualization import visualize_flow
from torchvision.utils import make_grid
from snn.models.quant_utils import print_scale_summary, log_all_overflow_stats, print_overflow_summary

from utils import *


def _register_spike_rate_hooks(model, layer_names=None):
    """Register forward hooks that accumulate spike-rate stats per spiking layer."""
    if layer_names is None:
        layer_names = ['e1', 'e2', 'm1', 'm2', 'm3', 'm4', 'd1']

    spike_totals = {}
    hook_handles = []
    active_layers = []

    for layer_name in layer_names:
        module = getattr(model, layer_name, None)
        if module is None:
            continue

        spike_totals[layer_name] = {'spikes': 0, 'neurons': 0}
        active_layers.append(layer_name)

        def _hook(_module, _inputs, output, lname=layer_name):
            # Spiking blocks return (spikes, membrane); skip non-spiking outputs.
            if not isinstance(output, tuple):
                return

            spk = output[0]
            if not torch.is_tensor(spk):
                return

            spike_totals[lname]['spikes'] += int((spk > 0).sum().item())
            spike_totals[lname]['neurons'] += int(spk.numel())

        hook_handles.append(module.register_forward_hook(_hook))

    return spike_totals, hook_handles, active_layers


def _spike_rate_pct(spikes, neurons):
    if neurons <= 0:
        return float('nan')
    return 100.0 * float(spikes) / float(neurons)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SNN Optical Flow Model')

    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to config YAML (required for quantized models, optional for full-precision)')
    parser.add_argument('--data-root', type=str, default=None,
                      help='Root directory for dataset (overrides config)')
    parser.add_argument('--output-dir', type=str, default='./logs',
                      help='Directory to save text results')
    parser.add_argument('--name', type=str, default='02_no_d1',
                      help='Directory for TensorBoard logs')

    parser.add_argument('--log-dir', type=str, default='./logs',
                      help='Directory for TensorBoard logs')
    parser.add_argument('--num-samples', type=int, default=None,
                      help='Number of samples to evaluate (None = all)')
    parser.add_argument('--log-interval', type=int, default=2,
                      help='Log TensorBoard images every N batches')
    parser.add_argument('--max-images', type=int, default=4,
                      help='Max images per visualization grid')
    parser.add_argument('--num-workers', type=int, default=0,
                      help='DataLoader workers (use 0 for lower memory usage on MVSEC)')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--strict-load', action='store_true',
                      help='Use strict mode when loading weights')
    parser.add_argument('--integer-sim', action='store_true',
                      help='Use integer-only forward pass for quantized models')
    parser.add_argument('--stats-interval', type=int, default=20,
                      help='Collect integer pipeline stats every N samples (default: 200)')
    parser.add_argument('--no-log', action='store_true',
                      help='Disable all TensorBoard logging')

    return parser.parse_args()


def build_eval_model(args, device):
    """
    Build and load model for evaluation.

    For full-precision models: loads config from checkpoint.
    For quantized models: loads config from --config YAML (detects quantized: true),
    builds quantized model, then loads weights from checkpoint.

    Returns:
        (model, config, is_quantized) tuple
    """
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = None

    is_quantized = config.get('quantized', False) if config else False

    if is_quantized:
        # --- Quantized model ---

        # Build model with quantization config
        model = get_model(config)

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            checkpoint = {'state_dict': state_dict}

        missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        model = model.to(device)
        model.eval()

        print_scale_summary(model)

        quant_info = (f"W{config.get('weight_bit_width', 32)}"
                      f"A{config.get('act_bit_width', 32)}"
                      f"M{config.get('mem_bit_width', 32)}")
        print(f"Loaded quantized model ({quant_info}) from {args.checkpoint}")
        if 'epoch' in checkpoint:
            print(f"  Trained for {checkpoint['epoch']} epochs")
        if 'best_val_epe' in checkpoint:
            print(f"  Best validation EPE: {checkpoint['best_val_epe']:.4f}")

    else:
        # --- Full-precision model ---
        if config is not None:
            model = get_model(config)

            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                checkpoint = {'state_dict': state_dict}

            model.load_state_dict(state_dict, strict=args.strict_load)
            model = model.to(device)
            model.eval()

            print(f"Loaded model from {args.checkpoint} (with external config)")
            if 'epoch' in checkpoint:
                print(f"  Trained for {checkpoint['epoch']} epochs")
            if 'best_val_epe' in checkpoint:
                print(f"  Best validation EPE: {checkpoint['best_val_epe']:.4f}")
        else:
            # Use config embedded in checkpoint
            model, config = build_model(None, device, train=False,
                                         checkpoint_path=args.checkpoint,
                                         strict=args.strict_load)

    model.disable_skip = True

    # Optionally wrap in integer-only inference model
    if getattr(args, 'integer_sim', False):
        if not is_quantized:
            raise ValueError("--integer-sim requires a quantized model config")
        from snn.models.integer_inference import IntegerInferenceModel
        model = IntegerInferenceModel(model, config,
                                      accum_bit_width=config.get('accum_bit_width', 32))
        model = model.to(device)
        print("[IntegerSim] Using integer-only forward pass")

    return model, config, is_quantized


def _render_hist_image(values, title):
    """Render a histogram as a torch image tensor [3, H, W] for TensorBoard."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    import torchvision.transforms.functional as TF

    vals = values.numpy() if isinstance(values, torch.Tensor) else values
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(vals, bins=100, color='steelblue', edgecolor='none', alpha=0.8, log=True)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Count')
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return TF.to_tensor(img)


def _plot_integer_histograms(all_metrics, output_dir):
    """Plot per-layer membrane and x_int histograms aggregated across all samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    samples_with_stats = [m for m in all_metrics if 'int_stats' in m]
    if not samples_with_stats:
        return

    # Collect raw tensors per layer across all samples
    # key_map: {raw_key: (label, color)}
    raw_keys = {
        'lif_mem_raw': ('Membrane Potential', 'steelblue'),
        'lif_x_int_raw': ('LIF Input (x_int)', 'coral'),
    }

    layer_data = {}  # {layer_name: {raw_key: [tensors]}}
    for m in samples_with_stats:
        for layer_name, layer_stats in m['int_stats'].items():
            if layer_name == '_peak':
                continue
            for rk in raw_keys:
                if rk in layer_stats:
                    layer_data.setdefault(layer_name, {}).setdefault(rk, []).append(
                        layer_stats[rk])

    if not layer_data:
        return

    hist_dir = output_dir / 'integer_histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Individual per-layer histograms
    for layer_name, data in layer_data.items():
        for rk, (label, color) in raw_keys.items():
            if rk not in data:
                continue
            vals = torch.cat(data[rk], dim=0).numpy()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(vals, bins=100, color=color, edgecolor='none', alpha=0.8, log=True)
            ax.set_title(f'{label} Distribution — {layer_name}')
            ax.set_xlabel('Value (integer)')
            ax.set_ylabel('Count')
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
            fig.tight_layout()
            short_key = rk.replace('lif_', '').replace('_raw', '')
            fig.savefig(hist_dir / f'{layer_name}_{short_key}_hist.png', dpi=150)
            plt.close(fig)

    # Combined figures — one per raw key, with all layers stacked
    for rk, (label, color) in raw_keys.items():
        layers_with_key = [(ln, d[rk]) for ln, d in layer_data.items() if rk in d]
        if not layers_with_key:
            continue
        n = len(layers_with_key)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
        if n == 1:
            axes = [axes]
        for ax, (layer_name, tensors) in zip(axes, layers_with_key):
            vals = torch.cat(tensors, dim=0).numpy()
            ax.hist(vals, bins=100, color=color, edgecolor='none', alpha=0.8, log=True)
            ax.set_title(layer_name)
            ax.set_ylabel('Count')
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[-1].set_xlabel('Value (integer)')
        short_key = rk.replace('lif_', '').replace('_raw', '')
        fig.suptitle(f'{label} Distributions (all samples)', fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(hist_dir / f'all_layers_{short_key}_hist.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    print(f"Integer histograms saved to {hist_dir}/")


def _write_integer_stats_summary(all_metrics, output_dir):
    """Write integer inference bit-width stats to a summary file."""
    # Collect all samples that have int_stats
    samples_with_stats = [m for m in all_metrics if 'int_stats' in m]
    if not samples_with_stats:
        return

    stats_file = output_dir / 'integer_stats.txt'

    # Gather all layer names and stat keys (excluding _peak)
    layer_names = []
    for m in samples_with_stats:
        for name in m['int_stats']:
            if name != '_peak' and name not in layer_names:
                layer_names.append(name)

    # Aggregate across all samples: take worst case (max) for bits, sum for counts
    global_worst = {}  # {layer: {stat: worst_value}}
    for m in samples_with_stats:
        for layer_name, layer_stats in m['int_stats'].items():
            if layer_name not in global_worst:
                global_worst[layer_name] = {}
            for key, val in layer_stats.items():
                if 'raw' in key:
                    continue  # skip raw tensors (handled by histogram plotter)
                if 'bits' in key:
                    global_worst[layer_name][key] = max(global_worst[layer_name].get(key, 0), val)
                elif 'count' in key:
                    global_worst[layer_name][key] = global_worst[layer_name].get(key, 0) + val
                elif 'pct' in key:
                    prev = global_worst[layer_name].get(key, [])
                    if not isinstance(prev, list):
                        prev = [prev]
                    prev.append(val)
                    global_worst[layer_name][key] = prev
                else:
                    if isinstance(val, (list, tuple)):
                        prev = global_worst[layer_name].get(key, [])
                        if not isinstance(prev, list):
                            prev = [prev]
                        prev.extend(val if isinstance(val, list) else list(val))
                        global_worst[layer_name][key] = prev
                    else:
                        global_worst[layer_name][key] = max(global_worst[layer_name].get(key, 0), val)

    # Average the pct lists
    for layer_stats in global_worst.values():
        for key in list(layer_stats.keys()):
            if isinstance(layer_stats[key], list):
                layer_stats[key] = sum(layer_stats[key]) / len(layer_stats[key])

    with open(stats_file, 'w') as f:
        n = len(samples_with_stats)
        f.write("INTEGER INFERENCE BIT-WIDTH ANALYSIS\n")
        f.write("=" * 72 + "\n")
        f.write(f"Samples analyzed: {n}\n\n")

        # Per-layer summary
        f.write("PER-LAYER SUMMARY (worst case across all samples & timesteps)\n")
        f.write("-" * 72 + "\n")
        f.write(f"{'Layer':<12s} {'Acc Bits':>10s} {'Product Bits':>14s} "
                f"{'Mem Bits':>10s} {'Acc Overflow':>14s} {'Mem Overflow':>14s} "
                f"{'Spike Rate':>12s}\n")
        f.write("-" * 86 + "\n")

        for name in layer_names:
            if name not in global_worst:
                continue
            s = global_worst[name]
            acc_bits = s.get('conv_acc_bits_max', '-')
            prod_bits = s.get('conv_product_bits_max', '-')
            mem_bits = s.get('lif_mem_bits_max', '-')
            acc_ovf = s.get('conv_acc_overflow_count', 0)
            mem_ovf = s.get('lif_mem_overflow_count', 0)
            spike_pct = s.get('lif_spike_rate_pct', None)

            acc_str = f"{acc_bits}" if isinstance(acc_bits, int) else str(acc_bits)
            prod_str = f"{prod_bits}" if isinstance(prod_bits, int) else str(prod_bits)
            mem_str = f"{mem_bits}" if isinstance(mem_bits, int) else str(mem_bits)
            acc_ovf_str = f"{acc_ovf}"
            mem_ovf_str = f"{mem_ovf}"
            spike_str = f"{spike_pct:.1f}%" if spike_pct is not None else "-"

            f.write(f"{name:<12s} {acc_str:>10s} {prod_str:>14s} "
                    f"{mem_str:>10s} {acc_ovf_str:>14s} {mem_ovf_str:>14s} "
                    f"{spike_str:>12s}\n")

        f.write("\n")

        # Global worst case
        if '_peak' in global_worst:
            peak = global_worst['_peak']
            f.write("GLOBAL PEAK (worst case across all layers, samples & timesteps)\n")
            f.write("-" * 72 + "\n")
            for key in sorted(peak.keys()):
                val = peak[key]
                if isinstance(val, float):
                    f.write(f"  {key:<40s} {val:.4f}\n")
                else:
                    f.write(f"  {key:<40s} {val}\n")
            f.write("\n")

        # Detailed per-layer stats
        f.write("DETAILED PER-LAYER STATS\n")
        f.write("-" * 72 + "\n")
        for name in layer_names:
            if name not in global_worst:
                continue
            f.write(f"\n  {name}:\n")
            for key in sorted(global_worst[name].keys()):
                val = global_worst[name][key]
                if isinstance(val, float):
                    f.write(f"    {key:<40s} {val:.4f}\n")
                else:
                    f.write(f"    {key:<40s} {val}\n")

        f.write("\n" + "=" * 72 + "\n")

    print(f"Integer stats summary saved to {stats_file}")


def evaluate(args):
    """Main evaluation function"""

    # Create output directory
    
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / 'eval_tensorboard_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Build model
    model, config, is_quantized = build_eval_model(args, device)
    spike_totals, spike_hook_handles, spike_layers = _register_spike_rate_hooks(model)

    # Build dataset
    data_root = args.data_root
    dataset_config = config.copy()
    dataset_config['data_root'] = data_root
    dataset_config['max_train_samples'] = args.num_samples

    # MVSEC Train dataset
    dataset_config['dt'] = 1
    dataset_config['sequences'] = ['outdoor_day2']
    dataset_config['crop_size'] = (256, 344)
    dataset = OpticalFlowDataset(config=dataset_config)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )

    print(f"Evaluating on {len(dataset)} samples from {data_root}")

    # Setup TensorBoard logger
    no_log = getattr(args, 'no_log', False)
    logger = Logger(log_dir=log_dir)
    if not no_log:
        logger.log_text('eval/checkpoint', args.checkpoint)
        logger.log_text('eval/data_root', data_root)
        logger.log_text('eval/quantized', str(is_quantized))
        if is_quantized:
            logger.log_text('eval/quant_config', args.config or 'embedded')
        if getattr(args, 'integer_sim', False):
            logger.log_text('eval/mode', 'integer_sim')

    vis_interval = args.log_interval
    max_vis_images = args.max_images
    num_bins = config.get('num_bins', 5)
    stats_interval = args.stats_interval
    is_integer_sim = getattr(args, 'integer_sim', False)

    # Metrics accumulator
    all_metrics = []

    # Evaluate
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Enable stats + raw tensor collection only on stats intervals
            do_stats = is_integer_sim and (idx % stats_interval == 0)
            if is_integer_sim and hasattr(model, 'enable_stats'):
                model.enable_stats(do_stats)

            # Move to device
            inputs = batch['input'].to(device)
            gt_flow = batch['flow'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            metadata = batch['metadata']

            # Preserve original mask for unmasked metrics before any downstream edits
            valid_mask_full = valid_mask.clone()

            # Forward pass
            pre_spike = {
                lname: (
                    spike_totals[lname]['spikes'],
                    spike_totals[lname]['neurons'],
                )
                for lname in spike_layers
            }

            outputs = model(inputs)
            pred_flow = outputs.get('flow', outputs.get('pred_flow',
                list(outputs.values())[0])) if isinstance(outputs, dict) else outputs

            # Compute metrics
            metrics = {}
            metrics['epe'] = endpoint_error(pred_flow, gt_flow, valid_mask_full)

            # Per-sample spike-rate percentage by layer: spikes / neurons.
            for lname in spike_layers:
                prev_s, prev_n = pre_spike[lname]
                ds = spike_totals[lname]['spikes'] - prev_s
                dn = spike_totals[lname]['neurons'] - prev_n
                pct = _spike_rate_pct(ds, dn)
                metrics_key = f'spike_rate_pct_{lname}'
                metrics[metrics_key] = pct
                logger.log_scalar(f'eval/{metrics_key}', pct, idx)

            metrics['outliers'] = calculate_outliers(pred_flow, gt_flow, valid_mask_full, threshold=3.0)
            # Angular metrics are produced in radians; convert once so all logs/reports use degrees.
            metrics['angular_error'] = np.degrees(angular_error(pred_flow, gt_flow, valid_mask_full).cpu())
            metrics['epe_weighted_angular_error'] = np.degrees(epe_weighted_angular_error(pred_flow, gt_flow, inputs, valid_mask).cpu())
            metrics['valid_pixels'] = valid_mask.sum().item()

            activity_patch = inputs.sum(dim=(1,2))
            low_activity_mask = (activity_patch < 1)
            low_activity_mask = low_activity_mask.unsqueeze(1)

            valid_mask[low_activity_mask] = 0.0
            metrics['epe_mask'] = endpoint_error(pred_flow, gt_flow, valid_mask)
            metrics['outliers_mask'] = calculate_outliers(pred_flow, gt_flow, valid_mask, threshold=3.0)
            metrics['angular_error_mask'] = np.degrees(angular_error(pred_flow, gt_flow, valid_mask).cpu())
            metrics['epe_weighted_angular_error_mask'] = np.degrees(epe_weighted_angular_error(pred_flow, gt_flow, inputs, valid_mask).cpu())
            metrics['valid_pixels_mask'] = valid_mask.sum().item()

            # Flow magnitude stats (matching trainer)
            flow_mag = torch.norm(pred_flow, dim=1)
            metrics['flow_max'] = flow_mag.max().item()
            metrics['flow_avg'] = flow_mag.abs().mean().item()

            metrics['sequence'] = metadata['sequence'][0]
            metrics['index'] = metadata['index'][0].item()

            all_metrics.append(metrics)

            # ---- TensorBoard scalar per sample ----
            if not no_log:
                epe_val = metrics['epe']
                if isinstance(epe_val, torch.Tensor):
                    epe_val = epe_val.detach().cpu().item()
                logger.log_scalar('eval/sample_epe', float(epe_val), idx)

                epe_mask_val = metrics['epe_mask']
                if isinstance(epe_mask_val, torch.Tensor):
                    epe_mask_val = epe_mask_val.detach().cpu().item()
                logger.log_scalar('eval/sample_epe_masked', float(epe_mask_val), idx)

                logger.log_scalar('eval/sample_flow_max', metrics['flow_max'], idx)
                logger.log_scalar('eval/sample_flow_avg', metrics['flow_avg'], idx)

            # ---- Integer inference stats (only on stats interval) ----
            if do_stats and hasattr(model, 'last_sample_stats') and model.last_sample_stats:
                metrics['int_stats'] = model.last_sample_stats
                if not no_log:
                    for layer_name, layer_stats in model.last_sample_stats.items():
                        for stat_key, val in layer_stats.items():
                            if isinstance(val, torch.Tensor):
                                # Histogram images logged at hist_interval
                                tag = f'int_stats/{layer_name}/{stat_key}'
                                hist_img = _render_hist_image(val, f'{layer_name} / {stat_key}')
                                logger.log_image(tag, hist_img, idx)
                            else:
                                tag = f'int_stats/{layer_name}/{stat_key}'
                                if isinstance(val, (list, tuple)):
                                    for ci, cv in enumerate(val):
                                        logger.log_scalar(f'{tag}/ch{ci}', float(cv), idx)
                                else:
                                    logger.log_scalar(tag, float(val), idx)

            # ---- TensorBoard visualizations ----
            # Use valid_mask (after low-activity masking) to match the trainer
            if not no_log and idx % vis_interval == 0:
                bs = min(inputs.shape[0], max_vis_images)

                # Events (sum over polarities, keep time bins)
                event_sum = inputs[:bs].sum(dim=2, keepdim=True)
                event_vis = event_sum.repeat(1, 1, 3, 1, 1)
                grid = make_grid(event_vis.view(-1, 3, event_vis.shape[3], event_vis.shape[4]),
                                 nrow=num_bins, normalize=False, pad_value=1.0)
                logger.log_image('eval/events', grid, idx)

                # Valid mask (after low-activity masking, matching trainer)
                vm = valid_mask[:bs].repeat(1, 1, 3, 1, 1)
                grid = make_grid(vm.view(-1, 3, vm.shape[3], vm.shape[4]),
                                 nrow=num_bins, normalize=False, pad_value=1.0)
                logger.log_image('eval/valid_mask', grid, idx)

                max_flow = min(torch.norm(gt_flow, dim=1).max().item(), 1.0)

                # GT flow, predicted flow, and masked versions
                gt_vis, pred_vis = [], []
                gt_mask_vis, pred_mask_vis = [], []
                for i in range(bs):
                    gt_c = visualize_flow(gt_flow[i].cpu(), max_flow=max_flow)
                    gt_vis.append(torch.from_numpy(gt_c).permute(2, 0, 1).float() / 255.0)

                    pr_c = visualize_flow(pred_flow[i].cpu(), max_flow=max_flow)
                    pred_vis.append(torch.from_numpy(pr_c).permute(2, 0, 1).float() / 255.0)

                    gt_m = visualize_flow((gt_flow[i] * valid_mask[i]).cpu(), max_flow=max_flow)
                    gt_mask_vis.append(torch.from_numpy(gt_m).permute(2, 0, 1).float() / 255.0)

                    pr_m = visualize_flow((pred_flow[i] * valid_mask[i]).cpu(), max_flow=max_flow)
                    pred_mask_vis.append(torch.from_numpy(pr_m).permute(2, 0, 1).float() / 255.0)

                logger.log_image('eval/gt_flow',
                    make_grid(torch.stack(gt_vis), nrow=2, pad_value=1.0), idx)
                logger.log_image('eval/pred_flow',
                    make_grid(torch.stack(pred_vis), nrow=2, pad_value=1.0), idx)
                logger.log_image('eval/gt_flow_masked',
                    make_grid(torch.stack(gt_mask_vis), nrow=2, pad_value=1.0), idx)
                logger.log_image('eval/pred_flow_masked',
                    make_grid(torch.stack(pred_mask_vis), nrow=2, pad_value=1.0), idx)

    # Remove hook handles now that evaluation is complete.
    for h in spike_hook_handles:
        h.remove()


    # Compute average metrics
    avg_metrics = {}
    numeric_keys = [k for k in all_metrics[0].keys() if k not in ['sequence', 'index', 'int_stats']]
    for key in numeric_keys:
        values = []
        for m in all_metrics:
            v = m[key]
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            values.append(v)
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)

    # Log summary scalars to TensorBoard
    if not no_log:
        logger.log_scalar('eval/avg_epe', avg_metrics['epe'], 0)
        logger.log_scalar('eval/avg_epe_masked', avg_metrics['epe_mask'], 0)
        logger.log_scalar('eval/avg_outliers', avg_metrics['outliers'], 0)
        logger.log_scalar('eval/avg_outliers_masked', avg_metrics['outliers_mask'], 0)
        logger.log_scalar('eval/avg_angular_error', avg_metrics['angular_error'], 0)
        logger.log_scalar('eval/avg_angular_error_masked', avg_metrics['angular_error_mask'], 0)
        logger.log_scalar('eval/avg_flow_max', avg_metrics['flow_max'], 0)
        logger.log_scalar('eval/avg_flow_avg', avg_metrics['flow_avg'], 0)

        for lname in spike_layers:
            total_spikes = spike_totals[lname]['spikes']
            total_neurons = spike_totals[lname]['neurons']
            layer_pct = _spike_rate_pct(total_spikes, total_neurons)
            logger.log_scalar(f'eval/avg_spike_rate_pct_{lname}', layer_pct, 0)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Checkpoint: {args.checkpoint}")
    if is_quantized:
        print(f"Quantized config: {args.config}")
    if getattr(args, 'integer_sim', False):
        print(f"Mode: Integer-only simulation")
    print(f"Number of samples: {len(all_metrics)}")
    print(f"\nAverage Metrics:")
    print(f"  EPE: {avg_metrics['epe']:.4f} ± {avg_metrics['epe_std']:.4f}")
    print(f"  Outliers: {avg_metrics['outliers']:.2f}% ± {avg_metrics['outliers_std']:.2f}%")
    print(f"  Angular Error: {avg_metrics['angular_error']:.2f}° ± {avg_metrics['angular_error_std']:.2f}°")
    print(f"  EPE Weighted Angular Error: {avg_metrics['epe_weighted_angular_error']:.2f}° ± {avg_metrics['epe_weighted_angular_error_std']:.2f}°")
    print("  Spike Rate (% = total spikes / total neurons):")
    for lname in spike_layers:
        layer_pct = _spike_rate_pct(spike_totals[lname]['spikes'], spike_totals[lname]['neurons'])
        print(f"    {lname}: {layer_pct:.4f}%")

    print(f"  Flow Max: {avg_metrics['flow_max']:.4f} ± {avg_metrics['flow_max_std']:.4f}")
    print(f"  Flow Avg: {avg_metrics['flow_avg']:.4f} ± {avg_metrics['flow_avg_std']:.4f}")
    print(f"  EPE (Masked): {avg_metrics['epe_mask']:.4f} ± {avg_metrics['epe_mask_std']:.4f}")
    print(f"  Outliers (Masked): {avg_metrics['outliers_mask']:.2f}% ± {avg_metrics['outliers_mask_std']:.2f}%")
    print(f"  Angular Error (Masked): {avg_metrics['angular_error_mask']:.2f}° ± {avg_metrics['angular_error_mask_std']:.2f}°")
    print(f"  EPE Weighted Angular Error (Masked): {avg_metrics['epe_weighted_angular_error_mask']:.2f}° ± {avg_metrics['epe_weighted_angular_error_mask_std']:.2f}°")
    print("="*50)

    # Save results to file
    results_file = output_dir / f'evaluate.txt'
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        if is_quantized:
            f.write(f"Quantized config: {args.config}\n")
        if getattr(args, 'integer_sim', False):
            f.write(f"Mode: Integer-only simulation\n")
        f.write(f"Number of samples: {len(all_metrics)}\n")
        f.write(f"\nAverage Metrics:\n")
        f.write(f"  EPE: {avg_metrics['epe']:.4f} ± {avg_metrics['epe_std']:.4f}\n")
        f.write(f"  Outliers: {avg_metrics['outliers']:.2f}% ± {avg_metrics['outliers_std']:.2f}%\n")
        f.write(f"  Angular Error: {avg_metrics['angular_error']:.2f}° ± {avg_metrics['angular_error_std']:.2f}°\n")
        f.write(f"  EPE Weighted Angular Error: {avg_metrics['epe_weighted_angular_error']:.2f} ± {avg_metrics['epe_weighted_angular_error_std']:.2f}\n")
        f.write("  Spike Rate (% = total spikes / total neurons):\n")
        for lname in spike_layers:
            layer_pct = _spike_rate_pct(spike_totals[lname]['spikes'], spike_totals[lname]['neurons'])
            f.write(f"    {lname}: {layer_pct:.4f}%\n")

        f.write(f"\n")
        f.write(f"  Flow Max: {avg_metrics['flow_max']:.4f} ± {avg_metrics['flow_max_std']:.4f}\n")
        f.write(f"  Flow Avg: {avg_metrics['flow_avg']:.4f} ± {avg_metrics['flow_avg_std']:.4f}\n")
        f.write(f"  EPE (Masked): {avg_metrics['epe_mask']:.4f} ± {avg_metrics['epe_mask_std']:.4f}\n")
        f.write(f"  Outliers (Masked): {avg_metrics['outliers_mask']:.2f}% ± {avg_metrics['outliers_mask_std']:.2f}%\n")
        f.write(f"  Angular Error (Masked): {avg_metrics['angular_error_mask']:.2f}° ± {avg_metrics['angular_error_mask_std']:.2f}°\n")
        f.write(f"  EPE Weighted Angular Error (Masked): {avg_metrics['epe_weighted_angular_error_mask']:.2f} ± {avg_metrics['epe_weighted_angular_error_mask_std']:.2f}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\nPer-sample results:\n")
        for m in all_metrics:
            f.write(f"{m['sequence']}_{m['index']:06d}: \n")
            f.write(f"EPE masked={m['epe_mask']:.4f}, Outliers masked={m['outliers_mask']:.2f}%, AngErr masked={m['angular_error_mask']:.2f}°, EPE Weighted AngErr masked={m['epe_weighted_angular_error_mask']:.2f}°, valid_pixels_mask={m['valid_pixels_mask']}\n")

    # Log and print overflow statistics (quantized models only)
    if is_quantized:
        if not no_log:
            log_all_overflow_stats(model, logger, step=0)
        print_overflow_summary(model)

    print(f"\nResults saved to {results_file}")

    # Write integer inference stats summary
    if getattr(args, 'integer_sim', False):
        _write_integer_stats_summary(all_metrics, output_dir)
        _plot_integer_histograms(all_metrics, output_dir)

    if not no_log:
        print(f"TensorBoard logs saved to {log_dir}")
        print(f"  View with: tensorboard --logdir {log_dir}")



if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
