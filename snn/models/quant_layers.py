import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
from .quant_utils import *


class BaseLayer(nn.Module):
    def __init__(
        self,
        config: Optional[dict] = None,
        weight_bit_width: int = None,
        act_bit_width: int = None,
        layer_name: str = "base",
    ):
        super().__init__()
        self.layer_name = layer_name
        
        # Get bit-widths from config if not specified, default to 8
        if weight_bit_width is None:
            weight_bit_width = config.get('weight_bit_width', 8) if config else 8
        if act_bit_width is None:
            act_bit_width = config.get('act_bit_width', 8) if config else 8
        
        # Separate bit-widths for weights and activations
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        self.quantize_weights = config.get('quantize_weights', False)
        self.quantize_activations = config.get('quantize_activations', False)

        self.enable_logging_params = config.get('enable_logging_params', False)
        self.logger = None

        self.weights = {}
    
    def log_params(self, x, out):
        if self.enable_logging_params and self.logger is not None and self.forward_count % 100 == 0:
            with torch.no_grad():
                self.logger.log_scalars(f'params/{self.layer_name}/input', {
                    'min': x.min().item(),
                    'max': x.max().item(),
                    'mean': x.mean().item(),
                    'std': x.std().item()
                }, self.forward_count)

                for name, weight in self.weights.items():
                    self.logger.log_scalars(f'params/{self.layer_name}/weights_{name}', {
                        'min': weight.min().item(),
                        'max': weight.max().item(),
                        'mean':weight.mean().item(),
                        'std': weight.std().item()
                    }, self.forward_count)
                    self.logger.log_histogram(f'params/{self.layer_name}/weights_{name}_hist', weight, self.forward_count)

                self.logger.log_scalars(f'params/{self.layer_name}/act', {
                    'min': out.min().item(),
                    'max': out.max().item(),
                    'mean':out.mean().item(),
                    'std': out.std().item()
                }, self.forward_count)
    

class QuantizedConv2d(BaseLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        s: int = 1,
        p: int = 0,
        groups: int = 1,
        use_norm = None,
        use_bias = None,
        config: Optional[dict] = None,
        weight_bit_width: int = None,
        act_bit_width: int = None,
        layer_name: str = "conv",
    ):
        super().__init__(config=config, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, layer_name=layer_name)

        self.forward_count = 0
        self.use_norm = config.get('use_norm', False) if use_norm is None else use_norm
        self.use_bias = config.get('use_bias', False) if use_bias is None else use_bias
        self.groups = groups
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, k,
            s, p, groups=self.groups, bias=self.use_bias
        )

        if self.use_norm: 
            self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=True)

        if self.quantize_weights:
            self.weight_quant = QuantWeight(
                bit_width=self.weight_bit_width,
                layer_name=f"{layer_name}_weight",
                num_channels=out_channels,  # Per-channel scales for output channels
                config=config,
            )
        
        if self.quantize_activations:
            self.act_quant = QuantAct(
                bit_width=self.act_bit_width,
                layer_name=f"{layer_name}_act",
                num_channels=out_channels,  # Per-channel scales for output channels
                config=config,
            )

        # Accumulator overflow tracker
        self.accum_bit_width = config.get('accum_bit_width', 32) if config else 32
        self.track_accum_overflow = config.get('track_accum_overflow', False) if config else False
        if self.track_accum_overflow:
            self.accum_overflow = OverflowTracker(
                bit_width=self.accum_bit_width,
                signed=True,
                name=f"{layer_name}_accum",
            )

        self.weights = {'conv': self.conv.weight}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_weights and self.weight_bit_width < 32:
            weight = self.weight_quant(self.conv.weight)
        else:
            weight = self.conv.weight
        
        out = F.conv2d(
            input=x, 
            weight=weight, 
            bias=self.conv.bias,
            groups=self.conv.groups,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )

        if self.use_norm:
            out = self.norm(out)

        # Check accumulator overflow BEFORE requantization
        if self.track_accum_overflow:
            # Scale to integer domain using weight scale (accum is in weight-scale units)
            scale = self.weight_quant.scale if self.quantize_weights else None
            self.accum_overflow.check(out, scale=scale)

        if self.quantize_activations and self.act_bit_width < 32:   
            out_act = self.act_quant(out)
        else:
            out_act = out
        
        self.forward_count += 1

        self.log_params(x, out_act)
        
        return out_act


class QuantizedLIF(nn.Module):
    def __init__(
        self,
        config: Optional[dict] = None,
        layer_name: str = "lif",
        option: Optional[str] = None,
    ):
        super().__init__()
        
        self.threshold = config.get('threshold', 1.0)
        self.decay = config.get('decay', 0.5)
        self.alpha = config.get('alpha', 10.0)
        self.reset = config.get('reset', 1.0)
        self.quantize_mem = config.get('quantize_mem', False)
        self.mem_bit_width = config.get('mem_bit_width', 16)

        self.mem = None
        self.option = option
        self.layer_name = layer_name

        # Membrane overflow tracker
        self.track_mem_overflow = config.get('track_mem_overflow', False) if config else False
        if self.track_mem_overflow:
            self.mem_overflow = OverflowTracker(
                bit_width=self.mem_bit_width,
                signed=True,
                name=f"{layer_name}_membrane",
            )

    def forward(self, x, mem) -> Any:

        if mem is None:
            mem = torch.zeros_like(x)

        if self.option == "spike_no_membrane":
            spk = SurrogateSpike.apply(x - self.threshold, self.alpha)
            return spk, mem

        if mem is None:
            mem = torch.zeros_like(x)

        decay_factor = torch.tensor(self.decay, device=x.device, dtype=x.dtype)
        mem = mem * decay_factor + x

        spk = SurrogateSpike.apply(mem - self.threshold, self.alpha)
        mem = mem * (self.threshold - spk) 
    
        # Check membrane range (no fake quantization, just warning if out of range)
        if self.quantize_mem:
            mem = check_membrane_range(mem, bit_width=self.mem_bit_width, mem_range=self.threshold * 2.0)

        # Track membrane overflow
        if self.track_mem_overflow:
            mem_scale = (self.threshold * 2.0) / (2 ** (self.mem_bit_width - 1) - 1)
            self.mem_overflow.check(mem, scale=torch.tensor(mem_scale))

        return spk, mem

class SurrogateSpike(torch.autograd.Function):
    """
    Hard threshold in forward; smooth surrogate gradient in backward.
    """
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        s = torch.sigmoid(alpha * x)
        grad = alpha * s * (1 - s)
        return grad_out * grad, None
