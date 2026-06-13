import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from .snn_layers import *
import torch.nn.functional as F


layers = {
    "SpikingConvBlock": SpikingConvBlock,
    "ConvBlock": ConvBlock,
}

class EmFlow(nn.Module):
    """
    This is a frozen copy of EmFlow architecture to ensure
    compatibility when loading older checkpoints as teachers.
    """
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config

        self.base_ch = config.get("base_ch", 32)
        self.use_polarity = config.get("use_polarity", True)

        self.logger = None  # TensorBoard logger
        self.disable_skip = False
        
        conv_layer = layers[config.get("conv_type", "SpikingConvBlock")]

        self.e1 = conv_layer(
            2 if self.use_polarity else 1,
            self.base_ch,
            k=3,
            s=2,
            p=1,
            config=config,
            layer_name="e1",
            option="spike_no_membrane"
        )

        self.e2 = conv_layer(
            self.base_ch, 
            self.base_ch, 
            k=3, 
            s=2, 
            p=1, 
            config=config,
            layer_name="e2",
            option="spike_no_membrane"
        )

        self.m1 = conv_layer(
            self.base_ch, 
            self.base_ch*2, 
            k=3, 
            s=2, 
            p=1, 
            config=config,
            layer_name="m1",
        )

        self.m2 = conv_layer(
            self.base_ch*2,
            self.base_ch*2,
            k=3,
            s=1,
            p=1,
            config=config,
            layer_name="m2",
        )

        self.m3 = conv_layer(
            self.base_ch*2,
            self.base_ch*2,
            k=3,
            s=1,
            p=1,
            config=config,
            layer_name="m3",
        )

        self.m4 = conv_layer(
            self.base_ch*2,
            self.base_ch*2,
            k=3,
            s=1,
            p=1,
            groups=1,
            config=config,
            layer_name="m4",
        )

        self.d1 = conv_layer(
            self.base_ch*2,
            self.base_ch,
            k=3,
            s=1,
            p=1,
            groups=1,
            config=config,
            layer_name="d1",
            option="spike_no_membrane"
        )
            
        # Flow prediction head
        self.h = ConvBlock(
            self.base_ch,
            2,
            k=3,
            s=1,
            p=1,
            groups=1,
            use_norm=False,
            use_bias=False,
            config=config,
            layer_name="h",
        )

        ##self.skip_proj_1 = nn.Conv2d(self.base_ch, self.base_ch * 2, kernel_size=1, stride=1, padding=0)
        ##self.skip_proj_2 = nn.Conv2d(self.base_ch, self.base_ch * 2, kernel_size=1, stride=1, padding=0)
        ### Initialize skip_proj_1 and skip_proj_2 weights to 1
        ##nn.init.constant_(self.skip_proj_1.weight, 1.0)
        ##if self.skip_proj_1.bias is not None:
        ##    nn.init.constant_(self.skip_proj_1.bias, 0.0)
        ##nn.init.constant_(self.skip_proj_2.weight, 1.0)
        ##if self.skip_proj_2.bias is not None:
        ##    nn.init.constant_(self.skip_proj_2.bias, 0.0)

    def load_state_dict(self, state_dict, strict: bool = True):
        legacy_prefix_map = {
            "e3.": "m1.",
            "e4.": "m2.",
            "d4.": "m3.",
            "d3.": "m4.",
            "d2.": "d1.",
            "flow_head.": "h.",
        }

        remapped = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in legacy_prefix_map.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            remapped[new_key] = value

        return super().load_state_dict(remapped, strict=strict)

    def set_logger(self, logger):
        self.logger = logger
        for module in self.modules():
            if hasattr(module, "logger"):
                module.logger = logger

    def forward(self, x):
        """Forward pass for full-image processing"""
        N, T, C, H, W = x.shape
        if self.use_polarity:
            assert C == 2, "Expected 2 polarity channels"
        else:
            assert C == 1, "Expected 1 channel when not using polarity"

        mem_e1 = mem_e2 = mem_m1 = mem_m2 = None
        mem_m3 = mem_m4 = mem_d1 = None

        spike_accum = None

        flow_acc = None

        for t in range(T):
            xt = x[:, t]
            xt = torch.clamp(xt, 0, 1)  

            s1, mem_e1 = self.e1(xt, mem_e1)
            s2, mem_e2 = self.e2(s1, mem_e2)
            s3, mem_m1 = self.m1(s2, mem_m1)
            s4, mem_m2 = self.m2(s3, mem_m2)

            ### 3X upscaling at end
            t3, mem_m3 = self.m3(s4, mem_m3)
            t3 = t3 + s4 if self.disable_skip is False else t3

            t2, mem_m4 = self.m4(t3, mem_m4)
            t2 = t2 + s3 if self.disable_skip is False else t2

            t1, mem_d1 = self.d1(t2, mem_d1)

            u = F.interpolate(t1, scale_factor=2, mode="nearest")
            u = u + s2 if self.disable_skip is False else u

            u = F.interpolate(u, scale_factor=2, mode="nearest")
            u = u + s1 if self.disable_skip is False else u
            #######################################################

            s = F.interpolate(u, scale_factor=2, mode="nearest")
            dflow = self.h(s)
            #######################################################

            ## incr_upscale t3, mem_m3 = self.m3(s4, mem_m3)
            ## incr_upscale t3 = t3 + s3 if self.disable_skip is False else t3

            ## incr_upscale t2 = F.interpolate(t3, scale_factor=2, mode="nearest")
            ## incr_upscale t2 = t2 + self.skip_proj_2(s2) if self.disable_skip is False else t2
            ## incr_upscale t2, mem_m4 = self.m4(t2, mem_m4)

            ## incr_upscale t1 = F.interpolate(t2, scale_factor=2, mode="nearest")
            ## incr_upscale t1 = t1 + self.skip_proj_1(s1) if self.disable_skip is False else t1
            ## incr_upscale t1, mem_d1 = self.d1(t1, mem_d1)

            ## incr_upscale u = F.interpolate(t1, scale_factor=2, mode="nearest")
            ## incr_upscale u = u + xt if self.disable_skip is False else u
            ## incr_upscale dflow = self.h(u)

            if flow_acc is None:
                flow_acc = dflow
            else:
                flow_acc = flow_acc + dflow
        
        return {"flow": flow_acc}

