"""Generic configuration for PEFT."""
# Copyright 2023 Purdue Digital Twin Lab. All rights reserved.
from __future__ import annotations
import importlib
import os
import sys
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, Iterable, List, Union

from torch import nn

from .base import BasePEFTConfig
from .typing import PEFTType

# Add the peft module to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PEFTConfigCollection(object):
    """A container class representing a collection of PEFT configurations."""

    _configs: Dict[str, Dict[str, Any]] = {}
    """Internal mapping from PEFT module names to PEFT configurations."""
    _upstream_mapping: Dict[str, List[str]] = {}
    """Internal mapping from upstream layers/modules to PEFT module names."""
    _downstream_mapping: Dict[str, List[str]] = {}
    """Internal mapping from downstream layers/modules to PEFT module names."""

    def __init__(self, configs: Iterable[Union[BasePEFTConfig, Dict[str,
                                                                    Any]]]):
        """Initialize a PEFT configuration collection.

        Args:
            configs (Iterable[BasePEFTConfig]): An iterable of PEFT configurations.
        """
        self._configs = {
            cfg.name:
            cfg.serialize() if isinstance(cfg, BasePEFTConfig) else cfg
            for cfg in configs
        }

        # create index table for fast lookup
        self._upstream_mapping = defaultdict(list)
        self._downstream_mapping = defaultdict(list)
        for cfg in configs:
            self._upstream_mapping[cfg['upstream_name']].append(cfg['name'])
            if hasattr(cfg, 'downstream_name'):
                self._downstream_mapping[cfg['downstream_name']].append(
                    cfg['name'])

    def __len__(self) -> int:
        """int: The number of PEFT configurations."""
        return len(self._configs)

    def __iter__(self) -> Dict[str, Any]:
        """Iterator[Dict[str, Any]]: An iterator over the PEFT
        configurations."""
        return iter(self._configs.values())

    @cached_property
    def upstream_layers(self) -> List[str]:
        """List[str]: The names of all upstream layers/modules."""
        return [cfg['upstream_name'] for cfg in self._configs.values()]

    @cached_property
    def downstream_layers(self) -> List[str]:
        """List[str]: The names of all downstream layers/modules."""
        return [cfg['downstream_name'] for cfg in self._configs.values()]

    def get_upstream_peft_modules(self, layer_name: str) -> List[str]:
        """Get all PEFT modules associated with a downstream layer/module.

        Args:
            layer_name (str): Name of the layer/module.

        Returns:
            List[str]: A list of PEFT modules.
        """
        return [name for name in self._downstream_mapping[layer_name]]

    def get_downstream_peft_modules(self, layer_name: str) -> List[str]:
        """Get all PEFT modules associated with an upstream layer/module.

        Args:
            layer_name (str): Name of the layer/module.

        Returns:
            List[str]: A list of PEFT modules.
        """
        return [name for name in self._upstream_mapping[layer_name]]


def build_layers_from_configs(
    configs: PEFTConfigCollection,
    repeats: int = 1,
) -> Dict[str, nn.Module, nn.ModuleDict]:
    """Build PEFT layers from configurations.

    Args:
        configs (PEFTConfigCollection): A PEFT configuration collection.
        repeats (int, optional): Number of times to repeat each PEFT layer.
        Defaults to 1.

    Returns:
        Dict[str, nn.Module]: A dictionary of PEFT layers.
    """
    module = importlib.import_module('peft')

    peft_layers = {}
    for cfg in configs:
        try:
            layer_class = getattr(module, str(cfg['type']))
            config_class = getattr(module, str(cfg['type']) + 'Config')
        except AttributeError:
            raise ValueError(f'Invalid PEFT layer type {cfg.type}.')
        cfg: BasePEFTConfig = config_class.deserialize(cfg)
        if repeats > 1 and cfg.peft_type == PEFTType.LAYER:
            for i in range(repeats):
                name = cfg.name + f'_{i}'
                peft_layers[name] = layer_class(config=cfg)
        else:
            name = cfg.name
            peft_layers[name] = layer_class(config=cfg)
    peft_layers = nn.ModuleDict(peft_layers)

    return peft_layers
