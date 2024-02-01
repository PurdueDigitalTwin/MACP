"""Base classes for PEFT modules."""
from __future__ import annotations
import abc
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

from torch import nn

from .typing import PEFTType


@dataclass
class BasePEFTConfig:
    """Base configuration for PEFT modules."""

    type: str
    """str: Type of the PEFT module. Example: `Adapter`."""
    in_features: int
    """Optional[int]: Dimensionality of the upstream layer/module output features."""
    upstream_name: str
    """str: Name of the upstream layer or module."""
    out_features: Optional[int] = None
    """Optional[int]: Dimensionality of the downstream layer/module input features."""
    downstream_name: Optional[str] = None
    """Optional[str]: Name of the downstream layer or module."""
    name: Optional[str] = None
    """Optional[str]: Name of the PEFT module."""
    peft_type: PEFTType = PEFTType.LAYER
    """PEFTTpe: Type of PEFT to apply."""

    def __post_init__(self):
        assert (isinstance(self.upstream_name, str)
                and '.' not in self.upstream_name
                ), f'Invalid upstream layer/module name {self.upstream_name}.'
        if self.downstream_name is not None:
            assert (
                isinstance(self.downstream_name, str)
                and '.' not in self.downstream_name
            ), f'Invalid downstream layer/module name {self.downstream_name}.'
        if self.out_features is None:
            self.out_features = self.in_features

        if self.name is None:
            # Use the default name.
            self.name = f'{self.upstream_name}_{self.downstream_name}_peft'

        if isinstance(self.peft_type, str):
            self.peft_type = PEFTType.deserialize(self.peft_type)

    @classmethod
    def deserialize(cls, cfg: Dict[str, Any]) -> BasePEFTConfig:
        """Deserialize a configuration from a dictionary.

        Args:
            cfg (Dict[str, Any]): Dictionary containing the configuration.

        Returns:
            BasePEFTConfig: The deserialized configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        assert isinstance(cfg, dict), f'Invalid PEFT configuration {cfg}.'

        cfg_ = {}
        for field_ in fields(cls):
            cfg_[field_.name] = cfg.get(field_.name, field_.default)

        return cls(**cfg_)

    def serialize(self) -> Dict[str, Any]:
        """Serialize a PEFT type to a dict.

        Returns:
            Dict[str, Any]: The serialized PEFT type.
        """
        return {
            field_.name: getattr(self, field_.name)
            for field_ in fields(self)
        }


class BasePEFTModel(nn.Module, abc.ABC):
    """Base class for PEFT models."""

    config: BasePEFTConfig
    """BasePEFTConfig: The configuration of the PEFT model.""" ''

    def __init__(self, config: BasePEFTConfig) -> None:
        """Initialize the PEFT model."""
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, x) -> None:
        """Forward pass of the PEFT model."""
        raise NotImplementedError
