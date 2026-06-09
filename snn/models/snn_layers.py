import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .quant_layers import *

layer_params = {
    "QuantizedLIF": QuantizedLIF,
}

class SpikingConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        s=1,
        p=1,
        groups=1,
        config=None,
        weight_bit_width=None,
        act_bit_width=None,
        layer_name=None,
        option=None,
    ):
        super().__init__()

        self.logger = None
        self.layer_name = layer_name
        
        self.conv = QuantizedConv2d(
            in_ch, out_ch, 
            k=k, 
            s=s, 
            p=p, 
            groups=groups,
            config=config,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            layer_name=layer_name,
        )

        self.lif = layer_params[config.get("lif_type", "QuantizedLIF")](
            config=config,
            layer_name=layer_name,
            option=option,
        )
        
    def forward(self, x, mem):
        x = self.conv(x)

        spk, mem = self.lif(x, mem)

        return spk, mem

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        s=1,
        p=1,
        groups=1,
        use_norm=None,
        use_bias=None,
        config=None,
        weight_bit_width=None,
        act_bit_width=None,
        layer_name=None,
    ):
        super().__init__()

        self.logger = None
        
        self.conv = QuantizedConv2d(
            in_ch, out_ch, 
            k=k, 
            s=s, 
            p=p, 
            groups=groups,
            use_norm=use_norm,
            use_bias=use_bias,
            config=config,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            layer_name=layer_name,
        )

        
    def forward(self, x):
        x = self.conv(x)
        
        return x
