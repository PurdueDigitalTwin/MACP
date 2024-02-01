"""Adapter module for parameter-efficient fine-tuning (PEFT)."""
# Copyright 2023 Purdue Digital Twin Lab. All rights reserved
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Union

import spconv.pytorch as spconv
import torch
from spconv.pytorch import functional as Fsp
from torch import FloatTensor, Tensor, nn

from .base import BasePEFTModel
from .config import BasePEFTConfig


@dataclass
class AdapterConfig(BasePEFTConfig):
    """Configuration for PEFT adapter module."""

    hidden_size: int = 64
    """The dimensionality of the lower-dimensional hidden space. Defaults to 64.""" ''
    initial_scale: float = 0.001
    """The initialization standard deviation of the weights. Defaults to 0.001."""
    use_conv: bool = True
    """bool: If to use a 1 x 1 convolution layer instead of a linear layer."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = int(self.hidden_size)
        self.initial_scale = float(self.initial_scale)


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Unit (GELU) activation function
    element-wise.

    This is a smoother version of ReLU that has been adopted in Adapter modules.
    Please refer to the original paper "Gaussian Error Linear Units (GELUs)" by
    Dan Hendrycks and Kevin Gimpel (:url:`https://arxiv.org/abs/1606.08415`).

    Examples::

        >>> m = GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        """"""
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the GELU activation function.

        Args:
            x (Tensor): Input tensor to apply the GELU activation function.

        Returns:
            Tensor: Output tensor after applying the GELU activation function.
        """
        cdf = 0.5 * (1.0 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        if self.inplace:
            return x.mul_(cdf)
        else:
            return x * cdf

    def extra_repr(self) -> str:
        extra_repr = f'inplace={self.inplace}' if self.inplace else ''
        return extra_repr


class Adapter(BasePEFTModel):
    """Adapter module for running PEFT on MMDet3D models.

    This module is adapted from the original implementation of the ``Adapter`` module
    in the paper "Parameter-Efficient Transfer Learning for NLP" by Neil Houlsby, et al.
    (:url:`http://proceedings.mlr.press/v97/houlsby19a.html`).
    """

    down_layer: Union[nn.Conv2d, nn.Linear]
    """nn.Conv2d | nn.Linear: layer for the down-projection of the input features."""
    activation: nn.Module
    """nn.Module: Activation function for the adapter module. Defaults to GELU."""
    up_layer: Union[nn.Conv2d, nn.Linear]
    """nn.Conv2d | nn.Linear: layer for the up-projection of the latent features."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config=config)

        self.activation = GELU()
        if config.use_conv:
            self.down_layer = nn.Conv2d(
                in_channels=config.in_features,
                out_channels=config.hidden_size,
                kernel_size=1,
                bias=True,
            )
            self.up_layer = nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.out_features,
                kernel_size=1,
                bias=True,
            )
        else:
            self.down_layer = nn.Linear(
                in_features=config.in_features,
                out_features=config.hidden_size,
                bias=True,
            )

            self.up_layer = nn.Linear(
                in_features=config.hidden_size,
                out_features=config.out_features,
                bias=True,
            )

        self.reset_parameters()

    def forward(self, x: Tensor) -> FloatTensor:
        """Forward pass of the adapter module."""
        x = x.float()
        out = self.up_layer(self.activation(self.down_layer(x)))

        return x + out

    def reset_parameters(self) -> None:
        """Reset the parameters of the adapter module."""
        if self.config.use_conv:
            nn.init.kaiming_normal_(
                self.down_layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(
                self.up_layer.weight, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.trunc_normal_(
                self.down_layer.weight, std=self.config.initial_scale)
            nn.init.trunc_normal_(
                self.up_layer.weight, std=self.config.initial_scale)
        nn.init.zeros_(self.down_layer.bias)
        nn.init.zeros_(self.up_layer.bias)


@dataclass
class SPAdapterConfig(BasePEFTConfig):
    """Configuration for PEFT sp-adapter module."""
    hidden_size: int = 64
    """The dimensionality of the lower-dimensional hidden space. Defaults to 64.""" ''

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = int(self.hidden_size)


class SPAdapter(BasePEFTModel):
    """Adapter module with sparse convolution."""

    def __init__(self, config: SPAdapterConfig) -> None:
        super().__init__(config=config)
        self.adapter = spconv.SparseSequential(
            spconv.SubMConv3d(
                config.in_features,
                config.hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.GELU(),
            spconv.SubMConv3d(
                config.hidden_size,
                config.out_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Forward pass of the adapter module."""
        return Fsp.sparse_add(x, self.adapter(x))

    def __repr__(self):
        return f'SPAdapter({self.config.in_features}, ' \
               f'{self.config.hidden_size}, ' \
               f'{self.config.out_features})'
