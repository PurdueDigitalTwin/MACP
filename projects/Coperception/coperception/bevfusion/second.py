"""Reimplementation of SECOND backbone with PEFT supports."""
# Copyright 2023 Purdue Digital Twin Lab. All rights reserved.
# Code adapted from the MMDet3D library.
from __future__ import annotations
from typing import Optional, Sequence, Tuple

from torch import Tensor
from torch import nn as nn

from mmdet3d.models import SECOND
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig
from .._typing import OptPEFTConfig
from ..peft.config import PEFTConfigCollection, build_layers_from_configs
from ..peft.tools import freeze_module, get_peft_layer, unfreeze_module


@MODELS.register_module()
class SECONDPEFT(SECOND):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet."""

    blocks: nn.ModuleList
    """nn.ModuleList: Backbone blocks."""
    peft_cfg: Optional[PEFTConfigCollection]
    """Optional[PEFTConfigCollection]: Config dict of PEFT layers."""
    peft_layers: nn.ModuleDict
    """nn.ModuleDict: PEFT layers."""

    def __init__(
            self,
            in_channels: int = 128,
            out_channels: Sequence[int] = [128, 128, 256],
            layer_nums: Sequence[int] = [3, 5, 5],
            layer_strides: Sequence[int] = [2, 2, 2],
            norm_cfg: ConfigType = dict(type='BN', eps=1e-3, momentum=0.01),
            conv_cfg: ConfigType = dict(type='Conv2d', bias=False),
            init_cfg: OptMultiConfig = None,
            pretrained: Optional[str] = None,
            peft_cfg: OptPEFTConfig = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            layer_nums,
            layer_strides,
            norm_cfg,
            conv_cfg,
            init_cfg,
            pretrained,
        )
        if peft_cfg is not None:
            if isinstance(peft_cfg, dict):
                peft_cfg = [peft_cfg]
            self.peft_cfg = PEFTConfigCollection(peft_cfg)
            self.peft_layers = nn.ModuleDict(
                build_layers_from_configs(self.peft_cfg))
            freeze_module(self)
            unfreeze_module(self.peft_layers)

    def forward_with_peft_bw(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs = []
        prefix = '_'.join([self.__class__.__name__, 'blocks'])
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if hasattr(self, 'peft_cfg'):
                for name in self.peft_cfg.get_downstream_peft_modules(
                        layer_name='_'.join([prefix, str(i)])):
                    x = self.peft_layers[name](x)
            outs.append(x)
        return tuple(outs)

    def forward_with_peft_lw(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs = []
        prefix = '_'.join([self.__class__.__name__, 'blocks'])
        for i in range(len(self.blocks)):
            peft_idx = 0
            block = self.blocks[i]
            for f in block:
                x = f(x)
                if isinstance(f, nn.Conv2d):
                    peft_layer = get_peft_layer(self.peft_cfg,
                                                self.peft_layers,
                                                f'{prefix}_{i}_{peft_idx}')
                    if peft_layer is not None:
                        x = peft_layer(x)
                        peft_idx += 1
            outs.append(x)
        return tuple(outs)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        if not hasattr(self, 'peft_cfg'):
            return super().forward(x)
        else:
            return self.forward_with_peft_lw(x)
