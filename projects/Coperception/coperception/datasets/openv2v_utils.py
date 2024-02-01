"""Helper functions related to the standard data format in OpenMMLab V2.0.

Copyright (c) Purdue Digital Twin Lab. All rights reserved.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

import mmengine
import numpy as np
from PIL import Image


@dataclass
class ImageInfo:
    """Image information data container for a single image."""

    img_path: str
    """Path to the image file."""
    height: int = field(default=None)
    """The height of the image."""
    width: int = field(default=None)
    """The width of the image."""
    depth_map: str = field(default=None)
    """Path of the depth map file."""
    cam2img: list[list[float]] | None = field(default=None)
    """Camera intrinsic matrix projecting 3D points onto the image."""
    lidar2img: list[list[float]] | None = field(default=None)
    """A 4x4 transformation matrix from lidar or depth to image."""
    lidar2cam: list[list[float]] | None = field(default=None)
    """A 4x4 transformation matrix from lidar to camera."""
    cam2ego: list[list[float]] | None = field(default=None)
    """A 4x4 transformation matrix from camera to ego-vehicle."""

    def __post_init__(self) -> None:
        if self.height is None or self.width is None:
            with Image.open(self.img_path) as img:
                self.width, self.height = img.size

        if self.cam2img is not None:
            assert len(self.cam2img) in (3, 4) and all(
                len(row) in (3, 4) for row in self.cam2img
            ), "The given 'cam2img' transformation matrix is not valid."

        if self.lidar2img is not None:
            assert len(self.lidar2img) == 4 and all(
                len(row) == 4 for row in self.lidar2img
            ), "The given 'lidar2img' transformation matrix is not 4x4."

        if self.cam2ego is not None:
            assert len(self.cam2ego) == 4 and all(
                len(row) == 4 for row in self.cam2ego
            ), "The given 'cam2ego' transformation matrix is not 4x4."


@dataclass
class Instance:
    """Annotation data container for a single instance."""

    bbox: list[int] = field(default=None)
    """List of 4 numbers representing the bounding box of the instance."""
    bbox_label: int = field(default=None)
    """An integer in the range [0, num_categories - 1] the category label."""
    bbox_3d: list[int] = field(default=None)
    """List of 7 (or 9) numbers representing the 3D bounding box of the
    instance."""
    bbox3d_isvalid: bool = field(default=None)
    """If to use the 3D bounding box during training."""
    bbox_label_3d: int = field(default=None)
    """3D category label, typically the same as label."""
    depth: float = field(default=None)
    """Projected center depth of the 3D bounding box compared to the image
    plane."""
    center_2d: list[float] = field(default=None)
    """Projected 2D center of the 3D bounding box."""
    attr_label: int = field(default=None)
    """Attribute labels (fine-grained labels such as stopping, moving, ignore,
    etc.)."""
    num_lidar_pts: int = field(default=None)
    """The number of LiDAR points in the 3D bounding box."""
    num_radar_pts: int = field(default=None)
    """The number of Radar points in the 3D bounding box."""
    difficulty: int = field(default=None)
    """The difficulty level of detecting the 3D bounding box."""
    unaligned_bbox_3d: bool = field(default=None)
    """If the 3D bounding box is unaligned."""

    def __post_init__(self) -> None:
        """Sanity checks for the instance data."""

        # align bbox_label and bbox_label_3d
        if self.bbox_label is None:
            assert self.bbox_label_3d is not None, 'Empty label!'
            self.bbox_label = self.bbox_label_3d
        if self.bbox_label_3d is None:
            assert self.bbox_label is not None, 'Empty label!'
            self.bbox_label_3d = self.bbox_label
        assert self.bbox_label == self.bbox_label_3d, 'Inconsistant labels!'

        if self.bbox is not None:
            assert (
                len(self.bbox) == 4
            ), f"Expect 'bbox' to have 4 numbers, but got {len(self.bbox)}."
        if self.bbox_label is not None:
            assert isinstance(
                self.bbox_label,
                int), ("Expect 'bbox_label' to be 'int', "
                       f'but got {type(self.bbox_label).__name__}.')


@dataclass
class LidarPoints:
    """Data container for lidar points."""

    num_pts_feats: int = field(default=None)
    """Number of features for each LiDAR point."""
    lidar_path: str = field(default=None)
    """Path of LiDAR data file."""
    lidar2ego: list[list[float]] = field(default=None)
    """A 4x4 transformation matrix from lidar to ego-vehicle."""

    def __post_init__(self) -> None:
        """Sanity checks for the `RadarPoints` object."""
        if self.lidar2ego is not None:
            assert len(self.lidar2ego) == 4 and all(
                len(row) == 4 for row in self.lidar2ego
            ), "The given 'lidar2ego' transformation matrix is not 4x4."


class RadarPoints:
    """Data container for radar points."""

    num_pts_feats: int = field(default=None)
    """Number of features for each Radar point."""
    radar_path: str = field(default=None)
    """Path of Radar data file."""
    radar2ego: list[list[float]] = field(default=None)
    """A 4x4 transformation matrix from radar to ego-vehicle."""

    def __post_init__(self) -> None:
        """Sanity checks for the `RadarPoints` object."""
        if self.radar2ego is not None:
            assert len(self.radar2ego) == 4 and all(
                len(row) == 4 for row in self.radar2ego
            ), "The given 'radar2ego' transformation matrix is not 4x4."


@dataclass
class DataInfo:
    """Standard OpenMMLab V2.0 data information container."""

    sample_idx: str
    """Sample id of the frame."""
    token: str = field(default=None)
    """Token of the frame."""
    timestamp: float = field(default=None)
    """timestamp of the current frame."""
    ego2global: list[list[float]] = field(default=None)
    """A transformation matrix from ego-vehicle to the global coordinate
    system."""
    images: ImageInfo | dict[str, ImageInfo] = field(
        default_factory=lambda: {})
    """A single image (represented by `ImageInfo`) or a dict for multi-view
    images."""
    lidar_points: LidarPoints | None = field(default=None)
    """Lidar point information container."""
    radar_points: RadarPoints | None = field(default=None)
    """Radar point information container."""
    instances: list[Instance] = field(default_factory=lambda: [])
    """Instances to be detected for object detection tasks."""
    instances_ignore: list[Instance] = field(default_factory=lambda: [])
    """Instances to be ignored during training for object detection tasks."""
    pts_instance_mask_path: str = field(default=None)
    """Path of instance labels for each point."""
    pts_semantic_mask_path: str = field(default=None)
    """Path of semantic labels for each point."""

    def __post_init__(self) -> None:
        assert isinstance(
            self.sample_idx, str
        ), f"Expect 'sample_idx' to be 'str', but got {type(self.sample_idx).__name__}"
        if self.ego2global is not None:
            assert len(self.ego2global) in (3, 4) and all(
                len(row) in (3, 4) for row in self.ego2global
            ), "The given 'ego2global' transformation matrix is not valid."
        assert (all(
            isinstance(img, ImageInfo) for img in self.images.values())
                if isinstance(self.images, dict) else isinstance(
                    self.images, ImageInfo)), "Invalid attribute: 'images'."
        assert isinstance(
            self.lidar_points,
            (LidarPoints, type(None))), "Invalid attribute: 'lidar_points'."
        assert isinstance(
            self.radar_points,
            (RadarPoints, type(None))), "Invalid attribute: 'radar_points'."
        assert (all(isinstance(ins, Instance) for ins in self.instances)
                or len(self.instances) == 0), "Invalid attribute: 'instances'"
        assert (all(
            isinstance(ins, Instance) for ins in self.instances_ignore)
                or len(self.instances_ignore)
                == 0), "Invalid attribute: 'instances_ignore'"
        if self.pts_instance_mask_path is not None:
            mmengine.check_file_exist(self.pts_instance_mask_path)
        if self.pts_semantic_mask_path is not None:
            mmengine.check_file_exist(self.pts_semantic_mask_path)

        # create cam2lidar matrix
        if self.lidar_points is not None:
            lidar2ego = np.asarray(
                self.lidar_points.lidar2ego, dtype=np.float64)
            for img in self.images.values():
                cam2ego = np.asarray(img.cam2ego, dtype=np.float64)
                img.lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
