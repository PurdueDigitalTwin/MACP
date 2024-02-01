"""Helper functions for PEFT."""
# Copyright 2023 Purdue Digital Twin Lab. All rights reserved.
# Author: Juanwu Lu and Yunsheng Ma
from __future__ import annotations

from torch import nn

from projects.Coperception.coperception.peft import PEFTConfigCollection


def freeze_module(module: nn.Module) -> None:
    """Freeze a module.

    Args:
        module (nn.Module): The module to be frozen.
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze a module.

    Args:
        module (nn.Module): The module to be unfrozen.
    """
    for param in module.parameters():
        param.requires_grad = True


def unfreeze_module_with_name(module: nn.Module, key='peft_layers') -> None:
    """Recursively unfreeze a module with a specific name."""
    for name, child in module.named_children():
        unfreeze_module_with_name(child, key)
        if key in name:
            unfreeze_module(child)


def get_peft_layer(peft_cfg: PEFTConfigCollection,
                   peft_layers,
                   name,
                   branch_idx=-1):
    peft_module_names = peft_cfg.get_downstream_peft_modules(name)
    # assert len(peft_module_names) == 1
    if branch_idx == -1:
        return peft_layers[peft_module_names[0]] if len(
            peft_module_names) == 1 else None
    else:
        return peft_layers[f'{peft_module_names[0]}_{branch_idx}']
