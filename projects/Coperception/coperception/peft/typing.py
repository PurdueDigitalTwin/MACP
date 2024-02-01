"""Typing for PEFT module classes."""
from __future__ import annotations
from enum import Enum


class PEFTType(str, Enum):
    """Enum class for available PEFT types."""

    LAYER = 'layer'
    """Applying PEFT layers in between layers."""
    MODULE = 'module'
    """Applying PEFT layers in between modules.""" ''

    @classmethod
    def deserialize(cls, name: str) -> PEFTType:
        """Deserialize a PEFT type from a string.

        Args:
            name (str): Name of the PEFT type.

        Returns:
            PEFTType: The deserialized PEFT type.

        Raises:
            ValueError: If the PEFT type is invalid.
        """
        name = name.strip().upper()
        if name not in PEFTType._member_names_:
            raise ValueError(f'Invalid PEFT type {name}.')

        return PEFTType[name]
