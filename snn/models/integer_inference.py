"""
Integer-only inference model for fixed-point software simulation.

Replaces all floating-point arithmetic with integer ops:
  - int8 weights × int8/binary activations → int32 accumulator
  - Requantization via integer multiply-and-shift: out = clamp((M_0 * acc) >> shift)
  - Integer membrane potential with integer threshold comparison
  - Decay via right-shift (0.5 = >> 1)

Usage:
    from snn.models.integer_inference import IntegerInferenceModel
    int_model = IntegerInferenceModel(model, config)
    result = int_model(input_tensor)   # input is float [N,T,C,H,W], output is float flow

The wrapper extracts integer parameters from a calibrated fake-quant model,
then runs a fully integer forward pass. Only the final output is converted
back to float for metric computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .quant_utils import export_quantized_params
import math


def _signed_bits_needed(val):
    """Minimum two's complement bit-width to represent signed integer val."""
    if val == 0:
        return 1
    return math.floor(math.log2(abs(val))) + 2


def _twos_complement_wrap(x, n_bits):
    """Wrap integer tensor to n_bits signed two's complement range.

    Emulates fixed-width integer overflow: values outside
    [-2^(n-1), 2^(n-1)-1] wrap around modularly.
    """
    mod = 1 << n_bits  # 2^n
    half = mod >> 1    # 2^(n-1)
    return (x + half) % mod - half


def _overflow_handle(x, n_bits, mode='wrap'):
    """Handle integer overflow with configurable mode.

    mode='wrap':  two's complement wrap-around
    mode='clamp': saturate to [-(2^(n-1)), 2^(n-1)-1]
    """
    if mode == 'clamp':
        half = 1 << (n_bits - 1)
        return x.clamp(-half, half - 1)
    return _twos_complement_wrap(x, n_bits)


class IntegerConv2d:
    """
    Integer-only convolution + requantization.

    Computes:
        acc = conv2d(x_int, w_int)           # int32 accumulator
        out = clamp((M_0 * acc) >> shift)    # requantize to act_bit_width
    """
    def __init__(self, int_weight: torch.Tensor, M_0: torch.Tensor,
                 shift: torch.Tensor, act_bit_width: int,
                 stride: int, padding: int, groups: int,
                 accum_bit_width: int = 32,
                 overflow_mode: str = 'wrap'):
        # Store as int tensors
        self.weight = int_weight.to(torch.int32)       # [Cout, Cin/g, kH, kW]
        self.M_0 = M_0.to(torch.int64)                 # [Cout] or scalar
        self.shift = shift.to(torch.int64)              # [Cout] or scalar
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.act_bit_width = act_bit_width
        self.accum_bit_width = accum_bit_width

        self.act_qmax = 2 ** (act_bit_width - 1) - 1
        self.act_qmin = -self.act_qmax - 1

        self.accum_qmax = 2 ** (accum_bit_width - 1) - 1
        self.accum_qmin = -self.accum_qmax - 1

        self.overflow_mode = overflow_mode

        # Stats collection (disabled by default)
        self.collect_stats = False
        self.last_stats = {}

        # Debug capture (disabled by default)
        self.collect_debug = False
        self.debug_data = {}

    def __call__(self, x_int: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_int: integer input [N, C, H, W] (int32)
        Returns:
            out_int: requantized integer output [N, Cout, H', W'] (int32)
        """
        # Conv2d in integer domain — PyTorch needs float for conv2d, so we
        # cast to float, do the multiply-accumulate, then cast back.
        # The results are mathematically identical to integer MAC since inputs
        # are integers and float32 can represent all int32 products exactly
        # for the value ranges we care about (8-bit × 8-bit × ~1000 terms).
        weight_f = self.weight.float().to(x_int.device)
        x_f = x_int.float()

        acc = F.conv2d(x_f, weight_f, bias=None,
                       stride=self.stride, padding=self.padding,
                       groups=self.groups)

        # Round to int32 (should already be integer-valued)
        acc = acc.round().to(torch.int64)

        # Collect accumulator stats before clamping
        if self.collect_stats:
            acc_min_val = int(acc.min().item())
            acc_max_val = int(acc.max().item())
            acc_mean_abs = acc.float().abs().mean().item()
            self.last_stats['acc_bits_min'] = _signed_bits_needed(acc_min_val)
            self.last_stats['acc_bits_max'] = _signed_bits_needed(acc_max_val)
            self.last_stats['acc_bits_mean'] = _signed_bits_needed(int(round(acc_mean_abs))) if acc_mean_abs >= 1 else 1
            n_over = ((acc > self.accum_qmax) | (acc < self.accum_qmin)).sum().item()
            self.last_stats['acc_overflow_count'] = n_over
            self.last_stats['acc_overflow_pct'] = n_over / max(acc.numel(), 1) * 100

        # Capture raw convolution sum (before overflow handling)
        if self.collect_debug:
            self.debug_data['sum'] = acc.detach().clone()

        # Handle accumulator overflow (wrap or clamp)
        acc = _overflow_handle(acc, self.accum_bit_width, self.overflow_mode)

        # Requantize: out = (M_0 * acc) >> shift
        # Reshape M_0 and shift for broadcasting: [1, Cout, 1, 1]
        M_0 = self.M_0.view(1, -1, 1, 1).to(acc.device)
        shift = self.shift.view(1, -1, 1, 1).to(acc.device)

        # Integer multiply then arithmetic right-shift
        product = M_0 * acc

        # Collect product stats
        if self.collect_stats:
            prod_min_val = int(product.min().item())
            prod_max_val = int(product.max().item())
            self.last_stats['product_bits_min'] = _signed_bits_needed(prod_min_val)
            self.last_stats['product_bits_max'] = _signed_bits_needed(prod_max_val)

        # Arithmetic right-shift with rounding: add (1 << (shift-1)) before shift
        # This matches a common "round-half-up" fixed-point convention.
        #rounding = torch.where(shift > 0,
        #                       torch.ones_like(shift) << (shift - 1),
        #                       torch.zeros_like(shift))
        #out = (product + rounding) >> shift
        out = product >> shift

        if self.collect_stats:
            out_min_val = int(out.min().item())
            out_max_val = int(out.max().item())
            self.last_stats['out_bits_min'] = _signed_bits_needed(out_min_val)
            self.last_stats['out_bits_max'] = _signed_bits_needed(out_max_val)
            n_over = ((out > self.act_qmax) | (out < self.act_qmin)).sum().item()
            self.last_stats['out_overflow_count'] = n_over
            self.last_stats['out_overflow_pct'] = n_over / max(out.numel(), 1) * 100

        # Capture product (M_0 * acc, before shift)
        if self.collect_debug:
            self.debug_data['sum_prod'] = out.detach().clone()

        # Handle output overflow (wrap or clamp)
        out = _overflow_handle(out, self.act_bit_width, self.overflow_mode).to(torch.int32)

        return out


class IntegerLIF:
    """
    Integer-only LIF neuron.

    Operates entirely in the integer domain:
        mem_int = (mem_int >> 1) + x_int    # decay=0.5 via right-shift
        spike = (mem_int > threshold_int)
        mem_int = mem_int * (1 - spike)     # hard reset to 0

    For IF neurons (no decay): mem_int = mem_int + x_int
    """
    def __init__(self, threshold_int: torch.Tensor, decay: Optional[float],
                 mem_bit_width: int, lif_type: str,
                 option: Optional[str] = None,
                 overflow_mode: str = 'wrap'):
        # threshold_int is per-channel: [Cout]
        self.threshold_int = threshold_int.to(torch.int64)
        self.decay = decay
        self.mem_bit_width = mem_bit_width
        self.lif_type = lif_type
        self.option = option

        self.mem_qmax = 2 ** (mem_bit_width - 1) - 1
        self.mem_qmin = -self.mem_qmax - 1

        self.overflow_mode = overflow_mode

        # Stats collection (disabled by default)
        self.collect_stats = False
        self.collect_raw = False
        self.last_stats = {}

        # Debug capture (disabled by default)
        self.collect_debug = False
        self.debug_data = {}

        # Precompute decay shift if decay is a power of 2
        self.decay_shift = None
        if decay is not None and decay > 0:
            import math
            # Check if decay is a power of 0.5 (i.e., 2^(-n))
            log2_inv = math.log2(1.0 / decay)
            if abs(log2_inv - round(log2_inv)) < 1e-6:
                self.decay_shift = int(round(log2_inv))

    def __call__(self, x_int: torch.Tensor,
                 mem_int: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_int: integer input [N, C, H, W] (int32)
            mem_int: integer membrane [N, C, H, W] (int64) or None
        Returns:
            spike: binary output (0 or 1) as int32
            mem_int: updated membrane (int64)
        """
        x_int = x_int.to(torch.int64)

        if mem_int is None:
            mem_int = torch.zeros_like(x_int, dtype=torch.int64)

        if self.option == "spike_no_membrane":
            # No membrane state — just threshold the input directly
            threshold = self.threshold_int.view(1, -1, 1, 1).to(x_int.device)
            spike = (x_int > threshold).to(torch.int32)
            if self.collect_debug:
                self.debug_data['fm_out'] = spike.detach().clone()
            if self.collect_stats:
                active = (x_int != 0).sum().item()
                fired = spike.sum().item()
                self.last_stats['spike_rate_pct'] = fired / max(active, 1) * 100
                self.last_stats['spike_count'] = fired
                self.last_stats['active_count'] = active
                if self.collect_raw:
                    self.last_stats['x_int_raw'] = x_int.detach().cpu().to(torch.float32).flatten()
            return spike, mem_int


        # Save membrane before update (before decay, input, overflow, threshold, reset)
        if self.collect_debug:
            self.debug_data['memb_pre'] = mem_int.detach().clone()

        # Apply decay to membrane
        if self.decay is not None and self.lif_type == 'QuantizedLIF':
            if self.decay_shift is not None:
                # Exact power-of-2 decay: arithmetic right-shift
                # For decay=0.5, shift=1: mem = mem >> 1
                # Only apply decay where there's input activity (matching float behavior)
                mem_int = mem_int >> self.decay_shift
                #has_input = (x_int.abs() > 0).to(torch.int64)
                #decayed = mem_int >> self.decay_shift
                #mem_int = has_input * decayed + (1 - has_input) * mem_int
            else:
                # Non-power-of-2 decay — approximate with multiply + shift
                # decay ≈ D * 2^(-16) where D = round(decay * 2^16)
                D = int(round(self.decay * (1 << 16)))
                mem_int = (mem_int * D + (1 << 15)) >> 16
        # else: IF neuron — no decay, membrane just accumulates

        # Accumulate input
        mem_int = mem_int + x_int

        # (Removed memb_post: after update, before threshold/reset)

        # Collect membrane stats before clamping
        if self.collect_stats:
            mem_min_val = int(mem_int.min().item())
            mem_max_val = int(mem_int.max().item())
            self.last_stats['mem_bits_min'] = _signed_bits_needed(mem_min_val)
            self.last_stats['mem_bits_max'] = _signed_bits_needed(mem_max_val)
            n_over = ((mem_int > self.mem_qmax) | (mem_int < self.mem_qmin)).sum().item()
            self.last_stats['mem_overflow_count'] = n_over
            self.last_stats['mem_overflow_pct'] = n_over / max(mem_int.numel(), 1) * 100
            # Store raw values for histogram plotting (only when requested)
            if self.collect_raw:
                self.last_stats['mem_raw'] = mem_int.detach().cpu().to(torch.float32).flatten()
                self.last_stats['x_int_raw'] = x_int.detach().cpu().to(torch.float32).flatten()

        # Handle membrane overflow (wrap or clamp)
        mem_int = _overflow_handle(mem_int, self.mem_bit_width, self.overflow_mode)

        # Threshold comparison (integer)
        threshold = self.threshold_int.view(1, -1, 1, 1).to(mem_int.device)
        spike = (mem_int > threshold).to(torch.int32)

        # Reset: mem = mem * (1 - spike) — hard reset to 0 on spike
        mem_int = mem_int * (1 - spike.to(torch.int64))

        # Capture debug data: membrane after reset (now called memb_post)
        if self.collect_debug:
            self.debug_data['memb_post'] = mem_int.detach().clone()
            self.debug_data['fm_out'] = spike.detach().clone()

        if self.collect_stats:
            active = (x_int != 0).sum().item()
            fired = spike.sum().item()
            self.last_stats['spike_rate_pct'] = fired / max(active, 1) * 100
            self.last_stats['spike_count'] = fired
            self.last_stats['active_count'] = active
            # Per-channel spike counts: number of '1' values in each feature map
            # spike shape: [N, C, H, W] — use first sample only
            self.last_stats['spike_per_channel'] = spike[0].sum(dim=(1, 2)).tolist()
            self.last_stats['total_per_channel'] = spike.shape[2] * spike.shape[3]

        return spike, mem_int


class IntegerInferenceModel(nn.Module):
    """
    Integer-only forward pass for fixed-point software simulation.

    Takes a calibrated fake-quant model, extracts all integer parameters
    via export_quantized_params(), and runs inference using only integer
    arithmetic (multiply, add, shift, compare, clamp).

    The only float operations are:
      1. Quantizing the raw input events to int8
      2. Dequantizing the final flow output back to float for metrics

    Args:
        model: calibrated PyTorch model with QuantWeight/QuantAct modules
               (can be None if exported_params is provided)
        config: model configuration dict
        accum_bit_width: accumulator bit width (default: from config or 32)
        multiplier_bits: M_0 bit width for requantization (default: 16)
        exported_params: pre-exported integer params dict from checkpoint
                         (if provided, skips export_quantized_params call)
    """
    def __init__(self, model: Optional[nn.Module], config: dict,
                 accum_bit_width: Optional[int] = None,
                 multiplier_bits: Optional[int] = None,
                 exported_params: Optional[Dict] = None):
        super().__init__()
        self.config = config
        self.use_polarity = config.get('use_polarity', False)
        self.disable_skip = True
        self.base_ch = config.get('base_ch', 8)
        self.accum_bit_width = accum_bit_width or config.get('accum_bit_width', 32)
        self.multiplier_bits = multiplier_bits or config.get('multiplier_bits', 16)
        self.mem_bit_width = config.get('mem_bit_width', 8)
        self.act_bit_width = config.get('act_bit_width', 8)
        self.weight_bit_width = config.get('weight_bit_width', 8)
        self.overflow_mode = config.get('overflow_mode', 'wrap')

        # Build integer layer objects from exported params
        self.layers = {}
        self.layer_order = []

        if exported_params is not None:
            print("[IntegerSim] Using pre-exported integer parameters")
            self._build_layers_from_export(exported_params)
        elif model is not None:
            print("[IntegerSim] Exporting quantized parameters...")
            exported = export_quantized_params(model, multiplier_bits=self.multiplier_bits)
            if not exported:
                raise RuntimeError("No quantized layers found. "
                                   "Run calibration first.")
            self._build_layers(model, exported)
        else:
            raise ValueError("Either model or exported_params must be provided")

        print(f"[IntegerSim] Built {len(self.layers)} integer layers")
        self._print_layer_summary()
        # Stats collection (disabled by default)
        self._collect_stats = False
        self.last_sample_stats = {}

    def enable_stats(self, enabled: bool = True):
        """Enable or disable all per-sample integer pipeline statistics and raw tensor collection."""
        if self._collect_stats == enabled:
            return  # no change, skip loop
        self._collect_stats = enabled
        if not enabled:
            self.last_sample_stats = {}
        for name in self.layer_order:
            layer = self.layers[name]
            layer['conv'].collect_stats = enabled
            if layer['type'] == 'spiking_conv':
                layer['lif'].collect_stats = enabled
                layer['lif'].collect_raw = enabled

    def enable_debug(self, enabled: bool = True):
        """Enable or disable debug capture of intermediate tensors."""
        self._collect_debug = enabled
        for name in self.layer_order:
            layer = self.layers[name]
            layer['conv'].collect_debug = enabled
            layer['conv'].debug_data = {}
            if layer['type'] == 'spiking_conv':
                layer['lif'].collect_debug = enabled
                layer['lif'].debug_data = {}

    @torch.no_grad()
    def forward_single_timestep(self, x_single: torch.Tensor,
                                mems: Optional[Dict] = None) -> Dict:
        """Run one timestep of integer inference and return all intermediate data.

        Args:
            x_single: float input [N, C, H, W] for one timestep
            mems: membrane state dict (None to start fresh)

        Returns:
            dict with:
                'layer_debug': {layer_name: {acc_raw, out, mem_after_reset, spike, ...}}
                'flow_int': integer flow output
                'flow_float': dequantized float flow
                'mems': updated membrane potentials
        """
        if mems is None:
            mems = {name: None for name in self.layer_order
                    if self.layers[name]['type'] == 'spiking_conv'}

        self.enable_debug(True)

        xt = torch.clamp(x_single, 0, 1)
        x_int = self._quantize_input(xt)

        layer_debug = {}

        def _capture(name):
            """Capture debug data from a layer's conv and lif."""
            layer = self.layers[name]
            data = {}
            # Conv debug: acc_raw (before M0*shift), out (after requant)
            for k, v in layer['conv'].debug_data.items():
                data[k] = v.cpu()
            layer['conv'].debug_data = {}
            # LIF debug: memb_post, fm_out (if present)
            if 'lif' in layer and layer['lif'].debug_data:
                for k, v in layer['lif'].debug_data.items():
                    data[k] = v.cpu()
                layer['lif'].debug_data = {}
            layer_debug[name] = data

        # --- Encoder ---
        s1, mems['e1'] = self._run_spiking_layer('e1', x_int, mems['e1'])
        _capture('e1')

        s2, mems['e2'] = self._run_spiking_layer('e2', s1, mems['e2'])
        _capture('e2')

        s3, mems['m1'] = self._run_spiking_layer('m1', s2, mems['m1'])
        _capture('m1')

        s4, mems['m2'] = self._run_spiking_layer('m2', s3, mems['m2'])
        _capture('m2')

        # --- Decoder ---
        m3, mems['m3'] = self._run_spiking_layer('m3', s4, mems['m3'])
        if not self.disable_skip:
            m3 = m3 + s4
        _capture('m3')

        m4, mems['m4'] = self._run_spiking_layer('m4', m3, mems['m4'])
        if not self.disable_skip:
            m4 = m4 + s3
        _capture('m4')

        d1, mems['d1'] = self._run_spiking_layer('d1', m4, mems['d1'])
        _capture('d1')

        u = self._integer_upsample(d1, scale_factor=2)
        u = self._integer_upsample(u, scale_factor=2)
        s = self._integer_upsample(u, scale_factor=2)

        # Flow head
        dflow_int = self._run_conv_layer('h', s)
        _capture('h')

        output_scale = self.layers['h']['output_scale']
        dflow = self._dequantize_output(dflow_int, output_scale)

        self.enable_debug(False)

        return {
            'layer_debug': layer_debug,
            'flow_int': dflow_int.cpu(),
            'flow_float': dflow.cpu(),
            'mems': {k: v.cpu() if v is not None else None for k, v in mems.items()},
            'input_int': x_int.cpu(),
        }

    def _get_conv_params(self, module) -> dict:
        """Extract stride, padding, groups from a QuantizedConv2d module."""
        conv = module.conv
        return {
            'stride': conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride,
            'padding': conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding,
            'groups': conv.groups,
        }

    def _build_layers_from_export(self, exported: Dict):
        """Build integer layers directly from pre-exported params dict.

        This path does not require the original model — all conv params
        (stride, padding, groups) must be present in the exported dict.
        """
        for conv_name, params in exported.items():
            # Derive block name: "e1.conv" → "e1"
            name = conv_name.rsplit('.', 1)[0] if '.' in conv_name else conv_name

            if 'int_weight' not in params or 'M_0' not in params:
                continue  # skip entries without full export data

            int_conv = IntegerConv2d(
                int_weight=params['int_weight'],
                M_0=params['M_0'],
                shift=params['shift'],
                act_bit_width=params.get('act_bit_width', self.act_bit_width),
                stride=params['stride'],
                padding=params['padding'],
                groups=params['groups'],
                accum_bit_width=self.accum_bit_width,
                overflow_mode=self.overflow_mode,
            )

            if 'lif_type' in params:
                int_lif = IntegerLIF(
                    threshold_int=params['threshold_int'],
                    decay=params.get('decay'),
                    mem_bit_width=self.mem_bit_width,
                    lif_type=params['lif_type'],
                    option=params.get('lif_option'),
                    overflow_mode=self.overflow_mode,
                )
                self.layers[name] = {
                    'type': 'spiking_conv',
                    'conv': int_conv,
                    'lif': int_lif,
                }
            else:
                self.layers[name] = {
                    'type': 'conv_only',
                    'conv': int_conv,
                    'output_scale': params.get('act_scale', torch.tensor(1.0)),
                }
            self.layer_order.append(name)

    def _build_layers(self, model: nn.Module, exported: Dict):
        """Build integer layer objects from model structure and exported params."""

        # Walk the model architecture and match with exported params
        for name, module in model.named_modules():
            module_type = type(module).__name__

            if module_type == 'SpikingConvBlock':
                conv_name = f"{name}.conv"
                if conv_name not in exported:
                    print(f"  [WARN] {conv_name} not found in exported params, skipping")
                    continue

                params = exported[conv_name]
                conv_params = self._get_conv_params(module.conv)

                # Build IntegerConv2d
                int_conv = IntegerConv2d(
                    int_weight=params['int_weight'],
                    M_0=params['M_0'],
                    shift=params['shift'],
                    act_bit_width=params.get('act_bit_width', self.act_bit_width),
                    stride=conv_params['stride'],
                    padding=conv_params['padding'],
                    groups=conv_params['groups'],
                    accum_bit_width=self.accum_bit_width,
                    overflow_mode=self.overflow_mode,
                )

                # Build IntegerLIF
                lif_type = params.get('lif_type', 'QuantizedLIF')
                int_lif = IntegerLIF(
                    threshold_int=params['threshold_int'],
                    decay=params.get('decay'),
                    mem_bit_width=self.mem_bit_width,
                    lif_type=lif_type,
                    option=params.get('lif_option'),
                    overflow_mode=self.overflow_mode,
                )

                self.layers[name] = {
                    'type': 'spiking_conv',
                    'conv': int_conv,
                    'lif': int_lif,
                }
                self.layer_order.append(name)

            elif module_type == 'ConvBlock':
                conv_name = f"{name}.conv"
                if conv_name not in exported:
                    print(f"  [WARN] {conv_name} not in exported params, skipping")
                    continue

                params = exported[conv_name]
                conv_params = self._get_conv_params(module.conv)

                int_conv = IntegerConv2d(
                    int_weight=params['int_weight'],
                    M_0=params['M_0'],
                    shift=params['shift'],
                    act_bit_width=params.get('act_bit_width', self.act_bit_width),
                    stride=conv_params['stride'],
                    padding=conv_params['padding'],
                    groups=conv_params['groups'],
                    accum_bit_width=self.accum_bit_width,
                    overflow_mode=self.overflow_mode,
                )

                # Store the output scale for final dequantization
                self.layers[name] = {
                    'type': 'conv_only',
                    'conv': int_conv,
                    'output_scale': params.get('act_scale', torch.tensor(1.0)),
                }
                self.layer_order.append(name)

    def _print_layer_summary(self):
        """Print a summary of all integer layers."""
        print("\n  Integer Layer Summary:")
        print(f"  {'Name':<20s} {'Type':<16s} {'Weight':<12s} {'M_0 range':<20s} {'Shift range':<14s}")
        print("  " + "-" * 80)
        for name in self.layer_order:
            layer = self.layers[name]
            conv = layer['conv']
            w_shape = 'x'.join(str(d) for d in conv.weight.shape)
            m0_min, m0_max = conv.M_0.min().item(), conv.M_0.max().item()
            sh_min, sh_max = conv.shift.min().item(), conv.shift.max().item()
            ltype = layer['type']
            if ltype == 'spiking_conv':
                lif = layer['lif']
                theta = lif.threshold_int
                theta_str = f"θ=[{theta.min().item()},{theta.max().item()}]"
                ltype = f"spiking({lif.lif_type[:3]})"
            else:
                theta_str = ""
            print(f"  {name:<20s} {ltype:<16s} {w_shape:<12s} "
                  f"[{m0_min},{m0_max}]{'':<8s} [{sh_min},{sh_max}]{'':<6s} {theta_str}")
        print()

    def _quantize_input(self, x_float: torch.Tensor) -> torch.Tensor:
        """
        Quantize float input events to integer.

        SNN input events are binary (0 or 1 per polarity per pixel).
        They are in the same domain as inter-layer spikes, so we keep
        them as {0, 1} integers. This ensures the input scale matches
        the spike scale (S_in = 1.0) used in the requantization math.
        """
        x_int = x_float.round().clamp(0, 1).to(torch.int32)
        return x_int

    def _dequantize_output(self, out_int: torch.Tensor,
                           output_scale: torch.Tensor) -> torch.Tensor:
        """
        Convert integer output back to float for metric computation.

        flow_float = out_int * S_output
        """
        scale = output_scale.float().to(out_int.device)
        if scale.dim() == 1 and scale.size(0) > 1:
            scale = scale.view(1, -1, 1, 1)
        return out_int.float() * scale

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Integer-only forward pass.

        Args:
            x: float input [N, T, C, H, W] — will be quantized to int

        Returns:
            dict with 'flow' key (float, dequantized from integer output)
        """
        N, T, C, H, W = x.shape

        # Initialize membrane potentials (integer)
        mems = {name: None for name in self.layer_order
                if self.layers[name]['type'] == 'spiking_conv'}

        flow_acc = None

        # Stats accumulators: {layer_name: {stat_key: [val_per_timestep]}}
        stats_accum = {} if self._collect_stats else None

        for t in range(T):
            xt = x[:, t]
            xt = torch.clamp(xt, 0, 1)

            # Quantize input to integer
            x_int = self._quantize_input(xt)

            # --- Encoder ---
            # e1: conv + LIF (spike_no_membrane)
            s1, mems['e1'] = self._run_spiking_layer('e1', x_int, mems['e1'])

            # e2: conv + LIF (spike_no_membrane)
            s2, mems['e2'] = self._run_spiking_layer('e2', s1, mems['e2'])

            # m1: conv + LIF
            s3, mems['m1'] = self._run_spiking_layer('m1', s2, mems['m1'])

            # m2: conv + LIF
            s4, mems['m2'] = self._run_spiking_layer('m2', s3, mems['m2'])

            # --- Decoder ---
            # m3: conv + LIF
            m3, mems['m3'] = self._run_spiking_layer('m3', s4, mems['m3'])

            # m4: conv + LIF
            m4, mems['m4'] = self._run_spiking_layer('m4', m3, mems['m4'])

            # d1: conv + LIF (spike_no_membrane)
            d1, mems['d1'] = self._run_spiking_layer('d1', m4, mems['d1'])

            # Upsample (nearest neighbor — exact in integer domain)
            u = self._integer_upsample(d1, scale_factor=2)

            u = self._integer_upsample(u, scale_factor=2)

            s = self._integer_upsample(u, scale_factor=2)

            # Flow head (ConvBlock — conv only, no LIF)
            dflow_int = self._run_conv_layer('h', s)

            # Dequantize flow to float for accumulation
            output_scale = self.layers['h']['output_scale']
            dflow = self._dequantize_output(dflow_int, output_scale)

            if flow_acc is None:
                flow_acc = dflow
            else:
                flow_acc = flow_acc + dflow

            # Collect per-timestep stats from each layer
            if self._collect_stats:
                self._collect_timestep_stats(stats_accum)

        # Aggregate stats across timesteps (take worst-case / averages)
        if self._collect_stats:
            per_layer = self._aggregate_stats(stats_accum)

            # Compute global peak bit-widths across all layers
            peak = {}
            for layer_stats in per_layer.values():
                for key, val in layer_stats.items():
                    if 'bits' in key:
                        peak[key] = max(peak.get(key, 0), val)
                    elif 'overflow' in key:
                        peak[key] = peak.get(key, 0) + val
            per_layer['_peak'] = peak

            self.last_sample_stats = per_layer

        return {"flow": flow_acc}

    def _collect_timestep_stats(self, stats_accum: Dict):
        """Snapshot last_stats from each integer layer after one timestep."""
        for name in self.layer_order:
            layer = self.layers[name]
            conv = layer['conv']
            if conv.last_stats:
                if name not in stats_accum:
                    stats_accum[name] = {}
                for key, val in conv.last_stats.items():
                    stats_accum[name].setdefault(f'conv_{key}', []).append(val)
            if layer['type'] == 'spiking_conv':
                lif = layer['lif']
                if lif.last_stats:
                    if name not in stats_accum:
                        stats_accum[name] = {}
                    for key, val in lif.last_stats.items():
                        # Tensor stats (like mem_raw) get appended as-is
                        stats_accum[name].setdefault(f'lif_{key}', []).append(val)

    def _aggregate_stats(self, stats_accum: Dict) -> Dict:
        """Aggregate per-timestep stats into per-sample summaries.

        For bits_* stats: take the max across timesteps (worst-case bit-width).
        For overflow counts: sum across timesteps.
        For overflow pct: average across timesteps.
        For raw tensor stats: concatenate across timesteps.
        """
        result = {}
        for layer_name, layer_stats in stats_accum.items():
            result[layer_name] = {}
            for key, vals in layer_stats.items():
                if 'raw' in key:
                    # Concatenate raw tensors across timesteps
                    result[layer_name][key] = torch.cat(vals, dim=0)
                elif 'bits' in key:
                    result[layer_name][key] = max(vals)
                elif 'count' in key:
                    result[layer_name][key] = sum(vals)
                elif 'pct' in key:
                    result[layer_name][key] = sum(vals) / len(vals)
                else:
                    result[layer_name][key] = max(vals)
        return result

    def _run_spiking_layer(self, name: str, x_int: torch.Tensor,
                           mem_int: Optional[torch.Tensor]
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run integer conv → LIF for a spiking layer."""
        layer = self.layers[name]
        conv_out = layer['conv'](x_int)
        spike, mem = layer['lif'](conv_out, mem_int)
        return spike, mem

    def _run_conv_layer(self, name: str, x_int: torch.Tensor) -> torch.Tensor:
        """Run integer conv only (no LIF) for output layers."""
        layer = self.layers[name]
        return layer['conv'](x_int)

    def _integer_upsample(self, x_int: torch.Tensor,
                          scale_factor: int) -> torch.Tensor:
        """
        Nearest-neighbor upsampling in integer domain.
        Just repeats values — perfectly exact, no interpolation.
        """
        # F.interpolate works on float, but nearest-neighbor with integer
        # values gives the exact same result. Cast temporarily.
        return F.interpolate(
            x_int.float(), scale_factor=scale_factor, mode='nearest'
        ).to(x_int.dtype)
