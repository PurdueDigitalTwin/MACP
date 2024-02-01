from dataclasses import dataclass

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from torch import FloatTensor, Tensor

from projects.Coperception.coperception.peft import (BasePEFTConfig,
                                                     BasePEFTModel)


@dataclass
class SSFConfig(BasePEFTConfig):
    """Configuration for PEFT SSF module."""

    pass


class SSF(BasePEFTModel):
    """PEFT SSF module."""

    scale: nn.Parameter
    shift: nn.Parameter

    def __init__(self, config: SSFConfig) -> None:
        """Initialize the PEFT SSF module."""
        super().__init__(config)
        self.scale = nn.Parameter(torch.ones(config.in_features))
        self.shift = nn.Parameter(torch.zeros(config.in_features))

    def __repr__(self) -> str:
        """Return the string representation of the PEFT SSF module."""
        return f'SSF({self.config.in_features})'

    def forward(self, x: Tensor) -> FloatTensor:
        """Forward pass of the PEFT SSF module."""
        if x.shape[-1] == self.scale.shape[0]:
            return self.scale * x + self.shift
        elif x.shape[1] == self.scale.shape[0]:
            if len(x.shape) == 3:
                return x * self.scale.view(1, -1, 1) + self.shift.view(
                    1, -1, 1)
            else:
                return x * self.scale.view(1, -1, 1, 1) + self.shift.view(
                    1, -1, 1, 1)
        else:
            raise ValueError(
                f'The input tensor shape {list(x.size())} does not match '
                f'the shape {list(self.scale.size())} of scale and shift factors.'
            )

    def reset_parameters(self) -> None:
        """Reset the parameters of the PEFT SSF module."""
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, mean=0.0, std=0.02)


@dataclass
class SPSSFConfig(BasePEFTConfig):
    """Configuration for PEFT SSF module."""

    pass


class SPSSF(BasePEFTModel):
    """PEFT Sparse SSF module."""

    def __init__(self, config: SSFConfig) -> None:
        """Initialize the PEFT SSF module."""
        super().__init__(config)
        self.conv = spconv.SparseConv3d(
            config.in_features,
            config.in_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def __repr__(self) -> str:
        """Return the string representation of the PEFT SSF module."""
        return f'SPSSF({self.config.in_features})'

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return self.conv(x)
