from os import path as osp
from typing import Callable, List, Union

import mmengine
import numpy as np
from mmengine.registry import DATASETS

from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes


@DATASETS.register_module()
class V2V4RealDataset(Det3DDataset):
    r"""V2V4Real Dataset.

    This class serves as the API for experiments on the V2V4Real Dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:
            - 'LiDAR': Box in LiDAR coordinates.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
    """
    METAINFO = {
        'classes': ['car'],
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ],
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = None,
                 box_type_3d: str = 'LiDAR',
                 modality: dict = None,
                 **kwargs) -> None:
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
            )
        if pipeline is None:
            pipeline = []

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            modality=modality,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            **kwargs)

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.
        Args:
            info (dict): Data information of single data sample.
        Returns:
            dict: Annotation information consists of the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            # empty instance
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            origin=(0.5, 0.5, 0.5),
        )
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
