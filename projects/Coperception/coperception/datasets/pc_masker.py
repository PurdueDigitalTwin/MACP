"""Point cloud masking module."""
# Copyright 2023 Purdue Digital Twin Lab. All rights reserved.
from __future__ import annotations
import math
from typing import Optional, Sequence

import torch
from mmcv.transforms.base import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import LiDARPoints


@TRANSFORMS.register_module()
class PCMasker2D(BaseTransform):
    """Mask a specific part of the source point cloud on 2D plane.

    The module is used to mask a specific part of the source point cloud within
    the area spcified by the area index and the resolution. The area index
    grows from top-left to bottom-right, and the resolution is a float or a
    tuple of floats that specifies the strides horizontally and vertically.
    """

    # ----------- public attributes ------------ #
    index: Optional[int]
    """Optional[int]: The index of the area to be masked. If `None`, skip masking."""
    resolution: tuple[float, float]
    """tuple[float]: The resolution of the area to be masked."""
    point_cloud_range: list[float]
    """list[float]: The range of the point cloud."""

    def __init__(
        self,
        resolution: float | Sequence[float],
        point_cloud_range: Sequence[float],
        index: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert isinstance(resolution, float) or (
            isinstance(resolution, Sequence) and len(resolution) == 2
        ), ValueError(
            "Resolution must be a float or a tuple of two floats."
            f" Got {resolution} instead."
        )
        self.resolution = (
            tuple(resolution[0:2])
            if isinstance(resolution, Sequence)
            else (resolution, resolution)
        )

        assert (
            isinstance(point_cloud_range, Sequence) and len(point_cloud_range) == 6
        ), ValueError("Invalid point cloud range.")
        self.point_cloud_range = list(point_cloud_range)

        if isinstance(index, int):
            assert isinstance(index, int) and index >= 0, ValueError("Invalid index.")
            assert index < self._num_grids, ValueError(
                f"Invalid index {index}."
                f" There are only {self._num_grids} areas available."
            )
        self.index = index

    def transform(self, results: dict) -> dict:
        """Method to mask points in the source point cloud.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the masked point cloud.
        """
        if self.index is None:
            # skip masking if no index is specified
            return results

        points = results["points"]
        assert isinstance(points, LiDARPoints), TypeError(
            "Only supports 'LiDARPoints' point cloud container."
        )

        # get the mask of points to be masked
        x_min, y_min = self.point_cloud_range[0:2]
        mask = (
            (points.tensor[:, 0] >= x_min + self.resolution[0] * self._col_idx)
            * (points.tensor[:, 0] <= x_min + self.resolution[0] * (self._col_idx + 1))
            * (points.tensor[:, 1] >= y_min + self.resolution[1] * self._row_idx)
            * (points.tensor[:, 1] <= y_min + self.resolution[1] * (self._row_idx + 1))
        )
        new_points = points.tensor[~mask].clone()

        results["points"] = LiDARPoints(
            tensor=new_points,
            points_dim=points.points_dim,
            attribute_dims=points.attribute_dims,
        )

        return results

    # ----------- private methods ------------ #
    @property
    def _num_cols(self) -> int:
        """int: The number of columns inside the area."""
        x_min = self.point_cloud_range[0]
        x_max = self.point_cloud_range[3]
        return math.ceil((x_max - x_min) / self.resolution[0])

    @property
    def _num_rows(self) -> int:
        """int: The number of rows inside the area."""
        y_min = self.point_cloud_range[1]
        y_max = self.point_cloud_range[4]
        return math.ceil((y_max - y_min) / self.resolution[1])

    @property
    def _num_grids(self) -> int:
        """int: The total number of grids."""
        return self._num_rows * self._num_cols

    @property
    def _row_idx(self) -> int:
        """int: The row index of the specified grid."""
        return self.index // self._num_cols

    @property
    def _col_idx(self) -> int:
        """int: The column index of the specified grid."""
        return self.index % self._num_cols

    def __str__(self) -> str:
        attr_str = ",".join(
            [
                f"index={self.index}",
                f"resolution={self.resolution}",
                f"point_cloud_range={self.point_cloud_range}",
            ]
        )
        return f"<{self.__class__.__name__}({attr_str}) at {hex(id(self))}>"


@TRANSFORMS.register_module()
class PCEgoMasker2D(PCMasker2D):
    """Mask a specific part of the source ego vehicle point cloud on 2D plane.

    The module extends the :class:`PCMasker2D` to mask only points collected by the
    ego vehicle. The ego vehicle is assumed to be the one whose point cloud is more
    concentrated around the origin.
    """

    def transform(self, results: dict) -> dict:
        """Override the method to mask points in the source point cloud for ego vehicle.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the masked point cloud.
        """
        if self.index is None:
            # skip masking if no index is specified
            return results

        points = results["points"]
        assert isinstance(points, LiDARPoints), TypeError(
            "Only supports 'LiDARPoints' point cloud container."
        )

        # identify the ego vehicle
        # the ego vehicle is assumed to be the one whose point cloud is more
        # concentrated around the origin. The groupby mean is adapted from
        # https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
        locs = points.tensor[:, 0:2]
        labels = points.tensor[:, -1].long()

        uniques = labels.unique().tolist()
        labels = labels.tolist()
        key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}

        labels = torch.LongTensor(list(map(key_val.get, labels)))
        labels = labels.view(labels.size(0), 1).expand(-1, locs.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        mean_locs = torch.zeros_like(unique_labels, dtype=torch.float)
        mean_locs.scatter_add_(0, labels, locs)
        mean_locs = mean_locs / labels_count.float().unsqueeze(1)

        ego_label = uniques[torch.argmin(torch.norm(mean_locs, p=2, dim=-1)).item()]

        # get the mask of points to be masked
        x_min, y_min = self.point_cloud_range[0:2]
        valid = (
            (points.tensor[:, 0] <= x_min + self.resolution[0] * self._col_idx)
            + (points.tensor[:, 0] >= x_min + self.resolution[0] * (self._col_idx + 1))
            + (points.tensor[:, 1] <= y_min + self.resolution[1] * self._row_idx)
            + (points.tensor[:, 1] >= y_min + self.resolution[1] * (self._row_idx + 1))
        ) * (points.tensor[:, -1] == ego_label) + points.tensor[:, -1] != ego_label
        new_points = points.tensor[valid].clone()

        results["points"] = LiDARPoints(
            tensor=new_points,
            points_dim=points.points_dim,
            attribute_dims=points.attribute_dims,
        )

        return results
