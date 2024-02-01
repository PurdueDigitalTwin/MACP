"""OpenV2V dataset in MMDet3D standard format.

Copyright(c) Purdue Digital Twin.
"""
from __future__ import annotations
import os

import mmengine
import numpy as np

from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes


@DATASETS.register_module()
class OpenV2VDataset(Det3DDataset):
    r"""OpenV2V Dataset.

    This class serves as the API for experiments on the OpenV2V Dataset.

    Please refer to the `OpenV2V Dataset <https://mobility-lab.seas.ucla.edu/opv2v/>`
    for data downloading and preparation.
    """

    METAINFO = {
        'classes': ('car', ),
        'version': 'train',
    }

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consisting of the following keys:

                1. `gt_bboxes_3d` (:obj:`LiDARInstances3DBoxes`): Ground-truth bboxes.
                2. `gt_labels_3d` (:obj:`NDArray`): Ground-truth labels of the bboxes.
        """
        ann_info = super().parse_ann_info(info)

        if ann_info is None:
            # cases with empty instance set
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros(shape=(0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            tensor=ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            origin=(0.5, 0.5, 0.5),
        ).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to the absolute path.
        And process the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And all path are absolute path.
        """
        token = info['token']
        scenario_token, agent_id, _ = token.split('.')[:3]

        if self.modality['use_lidar']:
            # info['lidar_points']['lidar_path'] = os.path.join(
            #     self.data_prefix.get('pts', ''),
            #     scenario_token,
            #     agent_id,
            #     info['lidar_points']['lidar_path'],
            # )
            info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']

        if self.modality['use_camera']:
            for img_info in info['images'].values():
                if 'img_path' in img_info:
                    img_info['img_path'] = os.path.join(
                        self.data_prefix.get('img', ''),
                        scenario_token,
                        agent_id,
                        img_info['img_path'],
                    )
                    mmengine.check_file_exist(img_info['img_path'])

            if self.default_cam_key is not None:
                info['img_path'] = info['images'][
                    self.default_cam_key]['img_path']
                if 'lidar2cam' in info['images'][self.default_cam_key]:
                    info['lidar2cam'] = np.array(
                        info['images'][self.default_cam_key]['lidar2cam'])
                if 'cam2img' in info['images'][self.default_cam_key]:
                    info['cam2img'] = np.array(
                        info['images'][self.default_cam_key]['cam2img'])
                if 'lidar2img' in info['images'][self.default_cam_key]:
                    info['lidar2img'] = np.array(
                        info['images'][self.default_cam_key]['lidar2img'])
                else:
                    info['lidar2img'] = info['cam2img'] @ info['lidar2cam']

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info
