"""COPEFT module for cooperative perception parameter-efficient fine-tuning
(PEFT)."""
# Copyright 2023 Purdue Digital Twin Lab. All rights reserved
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import FloatTensor, Tensor, nn

from .base import BasePEFTModel
from .config import BasePEFTConfig
from .typing import PEFTType


@dataclass
class COPEFTConfig(BasePEFTConfig):
    """Configurations for COPEFT module."""

    hidden_size: int = 64
    """int: The dimensionality of the lower-dimensional hidden space. Defaults to 64."""
    num_branches: int = 2
    """int: The number of parallel branches. Defaults to 2."""
    kernel_sizes: Sequence[int] = field(default=(1, 3, 5))
    """Sequence[int]: The kernel sizes of the parallel convolutional layers."""
    aggregation: str = 'concatenate'
    """str: The aggregation method for the convolutional output."""

    def __post_init__(self):
        self.type = 'COPEFT'
        self.peft_type = PEFTType.MODULE
        if self.out_features is None:
            self.out_features = self.in_features

        super().__post_init__()
        assert self.out_features == self.in_features, ValueError(
            "'in_features' must be equal to 'out_features', "
            f'but got <{self.in_features} != {self.out_features}>.')
        assert isinstance(self.num_branches,
                          int) and self.num_branches > 0, ValueError(
                              "'num_branches' must be a positive integer, "
                              f'but got <{self.num_branches}>.')
        assert (
            isinstance(self.kernel_sizes, Sequence)
            and len(self.kernel_sizes) > 0
            and all(isinstance(k, int) and k > 0 for k in self.kernel_sizes)
        ), ValueError(
            "'kernel_sizes' must be an non-empty sequence of positive integers, "
            f'but got <{self.kernel_sizes}>.')
        assert self.aggregation in ['sum', 'mean', 'concatenate'], ValueError(
            "'aggregation' must be one of ['sum', 'mean', 'concatenate'], "
            f'but got <{self.aggregation}>.')


class COPEFTBlock(nn.Module):
    """COPEFT block with parallel convolutional up-projection and down-
    projection.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        kernel_sizes (Sequence[int]): The kernel sizes of the convolutional layers.
        aggregation (str): The aggregation method for the convolutional output.
            Must be one of ['sum', 'mean', 'concatenate'].

    Examples:
        >>> from coperception.peft import COPEFTBlock
        >>> m = COPEFTBlock(4, 128)
        >>> input = torch.randn(2, 4, 64)
        >>> output = m(input)
    """

    in_features: int
    """int: The number of input features."""
    out_features: int
    """int: The number of output features."""
    kernel_sizes: Sequence[int]
    """Sequence[int]: The kernel sizes of the parallel convolutional layers."""
    aggregation: str
    """str: The aggregation method for the convolutional output.
    Must be one of ['sum', 'mean', 'concatenate']."""
    upconv: nn.ModuleList
    """nn.ModuleList: The parallel up-projection convolutional layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_sizes: Sequence[int] = [1, 3, 5],
        aggregation: str = 'concatenate',
    ) -> None:
        assert isinstance(in_features, int), ValueError(
            "'in_features' must be an integer, "
            f'but got <{in_features}: {type(in_features)}>.')
        assert isinstance(out_features, int), ValueError(
            "'out_features' must be an integer, "
            f'but got <{out_features}: {type(out_features)}>.')
        assert isinstance(kernel_sizes, Sequence) and all(
            isinstance(k, int) for k in kernel_sizes), ValueError(
                "'kernel_sizes' must be a sequence of integers, "
                f'but got <{kernel_sizes}: {type(kernel_sizes)}>.')
        assert aggregation in ['sum', 'mean', 'concatenate'], ValueError(
            "'aggregation' must be one of ['sum', 'mean', 'concatenate'], "
            f'but got <{aggregation}: {type(aggregation)}>.')

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_sizes = kernel_sizes
        if 1 not in kernel_sizes:
            self.kernel_sizes = [1] + list(kernel_sizes)
        self.aggregation = aggregation

        super().__init__()

        self.upconv = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=k,
                padding='same',
                stride=1,
            ) for k in self.kernel_sizes
        ])
        self.activation = nn.ReLU(inplace=True)

        if self.aggregation == 'concatenate':
            hidden_size = out_features * len(self.kernel_sizes)
        else:
            hidden_size = out_features

        self.downconv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=in_features,
            kernel_size=1,
            padding='same',
            stride=1,
        )

        self.reset_parameters()

    def forward(self, x: Tensor) -> FloatTensor:
        """Forward pass of the COPEFT block."""
        squeezed: bool = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeezed = True

        batch_size = x.size(0)
        src = x.float()[..., 0:self.in_features]
        src = src.transpose(2, 1)

        conv_output = [conv(src).unsqueeze(1) for conv in self.upconv]
        conv_output = torch.cat(conv_output, dim=1)
        if self.aggregation == 'concatenate':
            conv_output = conv_output.view(batch_size, -1,
                                           conv_output.size(-1))
        elif self.aggregation == 'sum':
            conv_output = torch.sum(conv_output, dim=1, keepdim=False)
        elif self.aggregation == 'mean':
            conv_output = torch.mean(conv_output, dim=1, keepdim=False)

        conv_output = self.activation(conv_output)
        conv_output = self.downconv(conv_output)
        x[...,
          0:self.in_features] = conv_output.transpose(2, 1)[...,
                                                            0:self.in_features]

        if squeezed:
            return x.squeeze(0)
        return x

    def reset_parameters(self) -> None:
        """Reset the parameters of the COPEFT block."""
        for conv in self.upconv:
            assert isinstance(conv, nn.Conv1d)
            conv.reset_parameters()
        self.downconv.reset_parameters()


class COPEFT(BasePEFTModel):
    """COPEFT module for running PEFT on input point cloud features.

    .. note::
        This module applies Inception-like parallel convolutional layers to the input
        point cloud feature on its point dimension to extract and aggregate correlation
        information among local point clouds. The up-projection and down-projection act
        as feature collation and point cloud reconstruction, respectively.

    Attribtues:
        config (COPEFTConfig): The configuration of the COPEFT module.
        branches (nn.ModuleList): The parallel convolutional branch for point clouds.
    """

    branches: nn.ModuleList
    """nn.ModuleList: The parallel convolutional branch for point clouds."""

    def __init__(self, config: COPEFTConfig) -> None:
        assert isinstance(config, COPEFTConfig), TypeError(
            "'config' must be an instance of <COPEFTConfig>, "
            f'but got <{config}: {type(config)}>.')
        super().__init__(config=config)

        self.branches = nn.ModuleList([
            COPEFTBlock(
                in_features=self.config.in_features,
                out_features=self.config.out_features,
                kernel_sizes=self.config.kernel_sizes,
                aggregation=self.config.aggregation,
            ) for _ in range(config.num_branches)
        ])
        self.reset_parameters()

    def forward(self, x: Tensor | list[Tensor]) -> FloatTensor:
        """Forward pass of the COPEFT module.

        Args:
            x (Tensor | list[Tensor]): The input point cloud features.

        Returns:
            FloatTensor: The reconstructed point cloud features.
        """
        if isinstance(x, list):
            # handle batches of point clouds
            return [self._single_batch_forward(x_i) for x_i in x]
        else:
            # handle a single batch of point clouds
            return self._single_batch_forward(x)

    def reset_parameters(self) -> None:
        """Reset the parameters of the COPEFT module."""
        for branch in self.branches:
            assert isinstance(branch, COPEFTBlock)
            branch.reset_parameters()

    @property
    def num_branches(self) -> int:
        """int: The number of parallel convolutional branches."""
        return len(self.branches)

    def _single_batch_forward(self, x: Tensor) -> FloatTensor:
        x = x.float()

        labels = x[..., -1].long().unique()
        # assert labels.size(0) <= self.num_branches, RuntimeError(
        #     f"Number of labels <{labels.size(0)}> does not match "
        #     f"number of branches <{self.num_branches}>."
        # )

        output = torch.zeros_like(x)
        for idx, label in enumerate(labels):
            branch = self.branches[idx]
            assert isinstance(branch, COPEFTBlock)
            mask = x[..., -1] == label
            output[mask] = branch(x[mask])
        output = x + output

        return output
