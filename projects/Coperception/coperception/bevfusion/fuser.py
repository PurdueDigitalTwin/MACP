from typing import List

import torch
from torch import nn

from mmdet3d.registry import MODELS
from .transfusion_head import ConvFuser


@MODELS.register_module()
class ConAda(nn.Module):
    def __init__(self,
                 num_channels: int = 256,
                 compress_ratio: int = 4,
                 aggregation: str = 'concat',
                 fusion: str = 'conv'):
        """
        Args:
            num_channels: Number of channels of the input feature map
            compress_ratio: The compression ratio of the input feature map
            aggregation: The aggregation method of the input feature map, 'sum' or 'mean' or 'weighted_mean', 'concat'
            fusion: The fusion method of the aggregated feature map, 'conv' or 'none'
        """
        super().__init__()
        assert num_channels % compress_ratio == 0
        assert aggregation in ['sum', 'mean', 'weighted_sum', 'concat']
        self.peft = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels // compress_ratio,
                kernel_size=1,
                bias=True),
            nn.GELU(),
            nn.Conv2d(
                in_channels=num_channels // compress_ratio,
                out_channels=num_channels,
                kernel_size=1,
                bias=True),
        )
        self.aggregation = aggregation
        self.fusion = fusion
        if self.fusion == 'conv':
            self.fuser = nn.Sequential(
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(True),
            )
        elif self.fusion == 'none':
            self.fuser = nn.Identity()
        else:
            raise NotImplementedError(f'Unknown fusion method {self.fusion}')
        if self.aggregation == 'concat':
            self.fuser = ConvFuser(
                in_channels=[num_channels, num_channels],
                out_channels=num_channels)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: A list of tensors, each with shape B x C x H x W
        Returns:
            torch.Tensor, shape B x C x H x W
        """
        ego_feature = inputs[0]  # (B, C, H, W)
        cav_feature = [self.peft(x) for x in inputs[1:]] if len(inputs) > 1 else [self.peft(ego_feature)]
        if self.aggregation == 'sum':
            fusion_feature = torch.stack([ego_feature] + cav_feature, dim=0)
            fusion_feature = fusion_feature.sum(dim=0)
        elif self.aggregation == 'mean':
            fusion_feature = torch.stack([ego_feature] + cav_feature, dim=0)
            fusion_feature = fusion_feature.mean(dim=0)
        elif self.aggregation == 'weighted_sum':
            cav_feature = torch.stack(cav_feature, dim=0)
            cav_feature = cav_feature.mean(dim=0)
            fusion_feature = ego_feature + cav_feature
        elif self.aggregation == 'concat':
            cav_feature = torch.stack(cav_feature, dim=0)
            cav_feature = cav_feature.mean(dim=0)
            fusion_feature = self.fuser([ego_feature, cav_feature])
        else:
            raise NotImplementedError(f'Unknown aggregation method {self.aggregation}')
        return self.fuser(fusion_feature)
