"""Typing for Coperception project."""
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Union  # noqa: F401

from torch import Tensor

OptDict = Optional[Dict[Any, Any]]
OptPEFTConfig = Optional[Union[Iterable[Dict[str, Any]], Dict[str, Any]]]
OptTensor = Optional[Tensor]
