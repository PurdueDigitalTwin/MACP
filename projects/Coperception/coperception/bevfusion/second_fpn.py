"""Reimplementation of SECOND FPN layer with PEFT supports."""
# Copyright 2023 Purdue Digital Twin Lab. All rights reserved.
# Code adapted from the MMDet3D library.
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor, nn

from mmdet3d.models.necks import SECONDFPN
from mmdet3d.registry import MODELS
from .._typing import OptPEFTConfig
from ..peft.config import PEFTConfigCollection, build_layers_from_configs
from ..peft.tools import freeze_module, unfreeze_module


@MODELS.register_module()
class SECONDFPNPEFT(SECONDFPN):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet with with PEFT layers.

    Attributes:
        in_channels (List[int]): Input channels of multi-scale feature maps.
        out_channels (List[int]): Output channels of feature maps.
        upsample_strides (List[int]): Strides used to upsample the feature maps.
        norm_cfg (Dict[str, Any]): Config dict of normalization layers.
        upsample_cfg (Dict[str, Any]): Config dict of upsample layers.
        conv_cfg (Dict[str, Any]): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
        peft_cfg (Optional[PEFTConfigCollection]): Config dict of PEFT layers.
        peft_layers (nn.ModuleDict): PEFT layers.
    """

    in_channels: List[int]
    """List[int]: Input channels of multi-scale feature maps."""
    out_channels: List[int]
    """List[int]: Output channels of feature maps."""
    deblocks: nn.ModuleList
    """nn.ModuleList: Deconvolution encoder blocks."""
    upsample_strides: List[int]
    """List[int]: Strides used to upsample the feature maps."""
    norm_cfg: Dict[str, Any]
    """Dict[str, Any]: Config dict of normalization layers."""
    upsample_cfg: Dict[str, Any]
    """Dict[str, Any]: Config dict of upsample layers."""
    conv_cfg: Dict[str, Any]
    """Dict[str, Any]: Config dict of conv layers."""
    use_conv_for_no_stride: bool
    """bool: Whether to use conv when stride is 1."""
    peft_cfg: Optional[PEFTConfigCollection]
    """Optional[PEFTConfigCollection]: Config dict of PEFT layers."""
    peft_layers: nn.ModuleDict
    """nn.ModuleDict: PEFT layers."""

    def __init__(
        self,
        in_channels: List[int] = [128, 128, 256],
        out_channels: List[int] = [256, 256, 256],
        upsample_strides: List[int] = [1, 2, 4],
        norm_cfg: Dict[str, Any] = dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg: Dict[str, Any] = dict(type='deconv', bias=False),
        conv_cfg: Dict[str, Any] = dict(type='Conv2d', bias=False),
        use_conv_for_no_stride: bool = False,
        init_cfg: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        peft_cfg: OptPEFTConfig = None,
    ) -> None:
        """Initialize FPN.

        Args:
            in_channels (Iterable[int]): Input channels of multi-scale feature maps.
            out_channels (Iterable[int]): Output channels of feature maps.
            upsample_strides (Iterable[int]): Strides used to upsample the feature maps.
            norm_cfg (Dict[str, Any]): Config dict of normalization layers.
            upsample_cfg (Dict[str, Any]): Config dict of upsample layers.
            conv_cfg (Dict[str, Any]): Config dict of conv layers.
            use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
            init_cfg (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]):
            Config dict of initialization.
            peft_cfg (OptPEFTConfig): Config dict of PEFT layers.
        """
        super().__init__(
            in_channels,
            out_channels,
            upsample_strides,
            norm_cfg,
            upsample_cfg,
            conv_cfg,
            use_conv_for_no_stride,
            init_cfg,
        )

        if peft_cfg is not None:
            if isinstance(peft_cfg, dict):
                peft_cfg = [peft_cfg]
            self.peft_cfg = PEFTConfigCollection(peft_cfg)
            self.peft_layers = nn.ModuleDict(
                build_layers_from_configs(
                    self.peft_cfg, repeats=len(self.deblocks)))
            freeze_module(self)
            unfreeze_module(self.peft_layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            List[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = []
        prefix = '_'.join([self.__class__.__name__, 'deblock'])
        for i, deblock in enumerate(self.deblocks):
            feat = deblock(x[i])
            if hasattr(self, 'peft_cfg'):
                for name in self.peft_cfg.get_downstream_peft_modules(prefix):
                    feat = self.peft_layers[f'{name}_{i}'](feat)
            ups.append(feat)

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]

        return [out]
