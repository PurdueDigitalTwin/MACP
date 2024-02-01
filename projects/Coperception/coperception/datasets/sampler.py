from typing import Iterator, List, Sequence, Sized

from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler


# noinspection PyMissingConstructor
@DATA_SAMPLERS.register_module()
class SubsetSampler(Sampler):
    indices: Sequence[int]
    """Sequence[int]: Indices of the subset."""

    def __init__(
        self,
        dataset: Sized,
        indices: List[int],
        seed: int = 0,
    ):
        self.indices = indices
        assert max(self.indices) < len(dataset)

    def __iter__(self) -> Iterator[int]:
        """Iterate all indices in the sampler."""
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
