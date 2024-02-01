"""PEFT adaptation of BEVFusion Model."""
# Copyright 2023 Purdue Digital Twin Lab. All rights reserved.
# Codebase adapted from BEVFusion implementation in MMDetection3D.
from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .._typing import OptDict, OptPEFTConfig, OptTensor
from ..peft.config import PEFTConfigCollection, build_layers_from_configs
from ..peft.tools import freeze_module, unfreeze_module
from .depth_lss import BaseDepthTransform
from .ops import Voxelization


@MODELS.register_module()
class BEVFusionPEFT(Base3DDetector):
    """The BEVFusion model adapted with Parameter-Efficient Fine-Tuning (PEFT)
    supports.

    This module overrides the existing BEVFusion implementation supporting flexible PEFT
    adaptor layers for fine-tuning. The PEFT adaptor layers are added in between the
    component modules, or the "blocks", of the BEVFusion model.

    Attributes:
        voxelize_reduce (bool): Whether to reduce the voxelized point-cloud data.
        pts_voxel_layer (nn.Module | None): Point-cloud voxelize transformation layer.
        pts_voxel_encoder (nn.Module | None): Point-cloud voxel encoder.
        pts_middle_encoder (nn.Module | None): Point-cloud middle encoder.
        image_backbone (nn.Module | None): Image backbone encoder.
        image_neck (nn.Module | None): Image neck module.
        view_transform (nn.Module | None): Image view transformation module.
        fusion_layer (nn.Module | None): Fusion layer for image and point-cloud data.
        pts_backbone (nn.Module | None): Point-cloud-based predictor backbone.
        pts_neck (nn.Module | None): Point-cloud-based predictor neck.
        bbox_head (nn.Module | None): Bounding-box head for prediction.
    """

    voxelize_reduce: bool
    """bool: Whether to reduce the voxelized point-cloud data."""
    pts_voxel_layer: Optional[nn.Module]
    """nn.Module | None: Point-cloud voxelize transformation layer."""
    pts_voxel_encoder: Optional[nn.Module]
    """nn.Module | None: Point-cloud voxel encoder."""
    pts_middle_encoder: Optional[nn.Module]
    """nn.Module | None: Point-cloud middle encoder."""
    image_backbone: Optional[nn.Module]
    """nn.Module | None: Image backbone encoder."""
    image_neck: Optional[nn.Module]
    """nn.Module | None: Image neck module."""
    view_transform: Optional[nn.Module]
    """nn.Module | None: Image view transformation module."""
    fusion_layer: Optional[nn.Module]
    """nn.Module | None: Fusion layer for image and point-cloud data."""
    pts_backbone: Optional[nn.Module]
    """nn.Module | None: Point-cloud-based predictor backbone."""
    pts_neck: Optional[nn.Module]
    """nn.Module | None: Point-cloud-based predictor neck."""
    bbox_head: Optional[nn.Module]
    """nn.Module | None: Bounding-box head for prediction."""
    peft_cfg: Optional[PEFTConfigCollection]
    """PEFTConfigCollection | None: PEFT configuration collection."""
    peft_layers: Optional[nn.ModuleDict]
    """nn.ModuleDict | None: PEFT layers."""

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: OptDict = None,
        pts_middle_encoder: OptDict = None,
        fusion_layer: OptDict = None,
        img_backbone: OptDict = None,
        img_neck: OptDict = None,
        view_transform: OptDict = None,
        pts_backbone: OptDict = None,
        pts_neck: OptDict = None,
        bbox_head: OptDict = None,
        init_cfg: OptMultiConfig = None,
        peft_cfg: OptPEFTConfig = None,
        headonly: bool = True,
        *args,
        **kwargs,
    ) -> None:
        # point-cloud modules
        if 'voxelize_cfg' in data_preprocessor:
            voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        else:
            voxelize_cfg = None
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # point-cloud modules
        if voxelize_cfg is not None:
            self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
            self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        # point-cloud modules
        self.pts_voxel_encoder = (
            MODELS.build(pts_voxel_encoder)
            if pts_voxel_encoder is not None else None)
        self.pts_middle_encoder = (
            MODELS.build(pts_middle_encoder)
            if pts_middle_encoder is not None else None)

        # image modules
        self.image_backbone = (
            MODELS.build(img_backbone) if img_backbone is not None else None)
        self.image_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = (
            MODELS.build(view_transform)
            if view_transform is not None else None)

        # fusion modules
        self.fusion_layer = (
            MODELS.build(fusion_layer) if fusion_layer is not None else None)

        # predictor modules
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.bbox_head = MODELS.build(bbox_head)

        # PEFT layers
        if peft_cfg is not None:
            self.peft_cfg = PEFTConfigCollection(peft_cfg)
            self.peft_layers = build_layers_from_configs(self.peft_cfg)

        self.reset_parameters()

        if headonly:
            freeze_module(self)
            unfreeze_module(self.fusion_layer)
            unfreeze_module(self.pts_neck)
            unfreeze_module(self.bbox_head)
        else:
            pass

        self.show_trainable_parameters()

    @property
    def with_bbox_head(self) -> bool:
        """bool: Whether the detector has a bbox head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self) -> bool:
        """bool: Whether the detector has a segmentation head."""
        return hasattr(self, 'seg_head') and self.seg_head is not None

    # noinspection PyMethodOverriding
    def extract_feat(
        self,
        batch_inputs_dict: Dict[str, OptTensor],
        batch_input_metas: List[Dict[str, Any]],
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        images = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if images is not None:
            images = images.contiguous()
            lidar2image, cam_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for meta in batch_input_metas:
                lidar2image.append(meta['lidar2image'])
                cam_intrinsics.append(meta['camera_intrinsic'])
                camera2lidar.append(meta['camera2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = images.new_tensor(np.asarray(lidar2image))
            cam_intrinsics = images.new_tensor(np.asarray(cam_intrinsics))
            camera2lidar = images.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = images.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = images.new_tensor(np.asarray(lidar_aug_matrix))
            # noinspection PyTypeChecker
            image_feature = self._extract_img_feat(
                x=images,
                points=deepcopy(points),
                lidar2image=lidar2image,
                cam_intrinsics=cam_intrinsics,
                camera2lidar=camera2lidar,
                img_aug_matrix=img_aug_matrix,
                lidar_aug_matrix=lidar_aug_matrix,
                metas=batch_input_metas,
            )
            features.append(image_feature)
        pts_feature = self._extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            feature = self.fusion_layer.forward(features)
        else:
            assert len(
                features
            ) == 1, f'No fusion layer, but got {len(features)} feats'
            feature = features[0]

        feature = self.pts_backbone.forward(feature)
        feature = self.pts_neck.forward(feature)

        return feature

    def loss(
        self,
        batch_inputs_dict: Dict[str, OptTensor],
        batch_data_samples: List[Det3DDataSample],
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (Dict[str, Tensor]): Batch of input tensors as dictionary.
            batch_data_samples (List[Det3DDataSample]): Batch of data samples.

        Returns:
            Dict[str, Tensor]: Losses of heatmap and bounding box of each task.
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        losses = {}

        feature = self.extract_feat(
            batch_inputs_dict=batch_inputs_dict,
            batch_input_metas=batch_input_metas)
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feature, batch_data_samples)
            losses.update(bbox_loss)

        return losses

    def predict(
        self,
        batch_inputs_dict: Dict[str, OptTensor],
        batch_data_samples: List[Det3DDataSample],
    ) -> List[Det3DDataSample]:
        """Predict and post-process results from a batch of inputs and data
        samples.

        Args:
            batch_inputs_dict (Dict[str, Tensor]): Batch of input tensors as dictionary.
            batch_data_samples (List[Det3DDataSample]): Batch of data samples.

        Returns:
            List[Det3DDataSample]: Prediction results.
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feature = self.extract_feat(
            batch_inputs_dict=batch_inputs_dict,
            batch_input_metas=batch_input_metas)

        if self.with_bbox_head:
            output = self.bbox_head.predict(feature, batch_input_metas)

        # add predictions back to the source data sample
        output = self.add_pred_to_datasample(batch_data_samples, output)

        return output

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        if self.image_backbone is not None:
            self.image_backbone.init_weights()

    def show_trainable_parameters(self) -> None:
        print('==================== Parameters ====================')
        total_size = 0
        tuned_from_original_size = 0
        peft_size = 0
        for name, parameter in self.named_parameters():
            param_size = parameter.numel()
            if parameter.requires_grad and 'peft' not in name:
                tuned_from_original_size += param_size
            elif 'peft' in name:
                assert parameter.requires_grad, f'PEFT layer {name} should be trainable'
                peft_size += param_size
            total_size += param_size
            print(
                f"{'Freeze' if not parameter.requires_grad else 'Trainable'}: {name} {tuple(parameter.shape)} "
                f'({param_size / 1e6:.4f}M)')
        print(
            f'Total Trainable Parameters: {(tuned_from_original_size + peft_size) / 1e6:.2f}M'
        )
        print(f'Total Tuned Parameters: {tuned_from_original_size / 1e6:.2f}M')
        print(f'Total PEFT Parameters: {peft_size / 1e6:.2f}M')
        print(f'Total Parameters: {total_size / 1e6:.2f}M')
        print('======================= End ============================')

        # ----------- private methods -------------

    # noinspection PyUnresolvedReferences
    def _extract_img_feat(
        self,
        x: Tensor,
        points: Tensor,
        lidar2image: Tensor,
        cam_intrinsics: Tensor,
        camera2lidar: Tensor,
        img_aug_matrix: Tensor,
        lidar_aug_matrix: Tensor,
        metas: Dict[str, Any],
    ) -> Tensor:
        """Encode and extract features from image data.

        Args:
            x (Tensor): Image data.
            points (Tensor): Associated point-cloud data.
            lidar2image (Tensor): Transformation matrix from lidar to image.
            cam_intrinsics (Tensor): Camera intrinsic matrix.
            camera2lidar (Tensor): Transformation matrix from camera to lidar.
            img_aug_matrix (Tensor): Image augmentation matrix.
            lidar_aug_matrix (Tensor): Point-cloud augmentation matrix.
            metas (Dict[str, Any]): Image meta information.

        Returns:
            Tensor: Extracted image features.
        """
        n_batch, n_sample, n_channels, height, width = x.shape
        x = x.view(n_batch * n_sample, n_channels, height, width).contiguous()

        # forward with peft layers
        x = self.image_backbone(x)
        if 'image_backbone' in self.peft_cfg.upstream_layers:
            ret = []
            for name in self.peft_cfg.get_downstream_peft_modules(
                    'image_backbone'):
                ret.append(self.peft_layers[name](x)).unsqueeze(1)
            ret = torch.cat(ret, dim=1).mean(dim=1, keepdim=False)
        x = self.image_neck(x)
        if 'image_neck' in self.peft_cfg.upstream_layers:
            ret = []
            for name in self.peft_cfg.get_downstream_peft_modules(
                    'image_neck'):
                ret.append(self.peft_layers[name](x)).unsqueeze(1)
            ret = torch.cat(ret, dim=1).mean(dim=1, keepdim=False)

        if not isinstance(x, Tensor):
            # NOTE: in case of an iterable of feature maps, use the first one
            x = x[0]
        x = x.view(n_batch, int(x.size(0) / n_batch), x.size(1), x.size(2),
                   x.size(3))

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            assert isinstance(self.view_transform, BaseDepthTransform)
            x = self.view_transform.forward(
                img=x,
                points=points,
                lidar2image=lidar2image,
                cam_intrinsic=cam_intrinsics,
                camera2lidar=camera2lidar,
                img_aug_matrix=img_aug_matrix,
                lidar_aug_matrix=lidar_aug_matrix,
                metas=metas,
            )

        return x

    def _extract_pts_feat(self, batch_inputs_dict: Dict[str, Any]) -> Tensor:
        """Extract features from point-cloud data.

        Args:
            batch_inputs_dict (Dict[str, Any]): Batch of input tensors as dictionary.

        Returns:
            Tensor: Extracted point-cloud features.
        """
        points = batch_inputs_dict['points']
        with torch.autocast(device_type='cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, _ = self._voxelize(points)
            batch_size = coords[-1, 0] + 1

        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    # noinspection PyMethodOverriding
    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: OptSampleList) -> None:
        # ignore implementation
        pass

    @torch.no_grad()
    def _voxelize(self,
                  points: Iterable[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Voxelize the input point-cloud data.

        Args:
            points (Iterable[Tensor]): Input point-cloud data.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Voxelized features, coordinates and sizes.
        """
        feats, coords, sizes = [], [], []
        for k, pnt in enumerate(points):
            voxelized_pnt = self.pts_voxel_layer(pnt)
            if len(voxelized_pnt) == 3:
                # hard voxelize
                feat, coord, num = voxelized_pnt
            else:
                assert len(voxelized_pnt) == 2
                feat, coord = voxelized_pnt
                num = None

            feats.append(feat)
            coords.append(
                nn.functional.pad(coord, (1, 0), mode='constant', value=k))
            if num is not None:
                sizes.append(num)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats: Tensor = feats.sum(dim=1, keepdim=False)
                feats: Tensor = feats / sizes.type_as(feats).view(-1, 1)
                feats: Tensor = feats.contiguous()

        return feats, coords, sizes


@MODELS.register_module()
class BEVFusionPEFTMid(BEVFusionPEFT):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: OptDict = None,
        pts_middle_encoder: OptDict = None,
        fusion_layer: OptDict = None,
        img_backbone: OptDict = None,
        img_neck: OptDict = None,
        view_transform: OptDict = None,
        pts_backbone: OptDict = None,
        pts_neck: OptDict = None,
        bbox_head: OptDict = None,
        init_cfg: OptMultiConfig = None,
        peft_cfg: OptPEFTConfig = None,
        headonly: bool = True,
        num_branches=1,
        max_num_cav=100,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            data_preprocessor,
            pts_voxel_encoder,
            pts_middle_encoder,
            fusion_layer,
            img_backbone,
            img_neck,
            view_transform,
            pts_backbone,
            pts_neck,
            bbox_head,
            init_cfg,
            peft_cfg,
            headonly,
            *args,
            **kwargs,
        )
        self.num_branches = num_branches
        self.max_num_cav = max_num_cav
        assert self.num_branches == 1 or 2, 'Only support 1 or 2 branches'

    @staticmethod
    def split_points_tensor(points: Tensor,
                            threshold: int = 1000) -> List[Tensor]:
        """Split the input point-cloud data according to the CAV sources.
        Args:
            points (Tensor): Input point-cloud data.
            threshold (int): The minimum number of points in a CAV.

        Returns:
            List[Tensor]: List of point-cloud data.
        """
        points = points.float()
        cav_ids = points[:, -1].long().unique()
        points_list = []
        for idx, label in enumerate(cav_ids):
            mask = points[:, -1] == label
            cav_points = points[mask, :-1]  # remove CAV ID
            if cav_points.shape[0] > threshold:
                points_list.append(cav_points)
        # sort by the mean of x and y
        points_list.sort(
            key=lambda x: torch.abs(torch.mean(x[:, :2])).item(),
            reverse=False)
        # points_list.sort(key=lambda x: x.shape[0], reverse=True)
        return points_list

    @staticmethod
    def pad_cavs(points_list: List[Tensor], pad_len: int) -> List[Tensor]:
        """Pad the input point-cloud data to the same length."""
        while len(points_list) < pad_len:
            points_list.append(
                torch.zeros((1, points_list[0].shape[1]),
                            device=points_list[0].device))  # dummy tensor
        return points_list

    def extract_feat(
        self,
        batch_inputs_dict: Dict[str, OptTensor],
        batch_input_metas: List[Dict[str, Any]],
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        images = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        batch_size = len(points)

        if images is not None:
            raise NotImplementedError(
                'Image features are not supported in this model.')
        batch_split_list = []
        for batch in points:
            splits = self.split_points_tensor(batch)[0:self.max_num_cav]
            batch_split_list.append(splits)

        # pad the splits to the same length
        num_splits = max(len(splits) for splits in batch_split_list)
        for idx, splits in enumerate(batch_split_list):
            batch_split_list[idx] = self.pad_cavs(splits, num_splits)
        assert all(len(splits) == num_splits for splits in batch_split_list
                   )  # all batches should have the same number of cavs

        split_inputs = [{'points': []} for _ in range(num_splits)]
        for splits in batch_split_list:
            for split_idx, split in enumerate(splits):
                split_inputs[split_idx]['points'].append(split)
        # All branches should have the same batch size
        assert all(
            len(split_input['points']) == batch_size
            for split_input in split_inputs)

        features = []

        for branch_idx, branch_inputs in enumerate(split_inputs):
            branch_idx = (-1 if self.num_branches == 1 else min(
                branch_idx, self.num_branches - 1))
            pts_feature = self._extract_pts_feat(
                branch_inputs, branch_idx=branch_idx)
            features.append(pts_feature)

        feature = self.fusion_layer.forward(features)
        feature = self.pts_backbone.forward(feature)
        feature = self.pts_neck.forward(feature)
        return feature

    def _extract_pts_feat(self,
                          batch_inputs_dict: Dict[str, Any],
                          branch_idx: int = -1) -> Tensor:
        """Extract features from point-cloud data.
        Args:
            batch_inputs_dict (Dict[str, Any]): Batch of input tensors as dictionary.

        Returns:
            Tensor: Extracted point-cloud features.
        """
        points = batch_inputs_dict['points']
        with torch.autocast(device_type='cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, _ = self._voxelize(points)
            batch_size = coords[-1, 0] + 1

        x = self.pts_middle_encoder(feats, coords, batch_size, branch_idx)
        return x


@MODELS.register_module()
class BEVFusionPEFTNo(BEVFusionPEFT):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: OptDict = None,
        pts_middle_encoder: OptDict = None,
        fusion_layer: OptDict = None,
        img_backbone: OptDict = None,
        img_neck: OptDict = None,
        view_transform: OptDict = None,
        pts_backbone: OptDict = None,
        pts_neck: OptDict = None,
        bbox_head: OptDict = None,
        init_cfg: OptMultiConfig = None,
        peft_cfg: OptPEFTConfig = None,
        headonly: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            data_preprocessor,
            pts_voxel_encoder,
            pts_middle_encoder,
            fusion_layer,
            img_backbone,
            img_neck,
            view_transform,
            pts_backbone,
            pts_neck,
            bbox_head,
            init_cfg,
            peft_cfg,
            headonly,
            *args,
            **kwargs,
        )

    def extract_feat(
        self,
        batch_inputs_dict: Dict[str, OptTensor],
        batch_input_metas: List[Dict[str, Any]],
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        images = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        batch_size = len(points)

        if images is not None:
            raise NotImplementedError(
                'Image features are not supported in this model.')

        ego_inputs = {'points': []}
        for batch in points:
            splits = BEVFusionPEFTMid.split_points_tensor(batch)
            ego_inputs['points'].append(splits[0])

        pts_feature = self._extract_pts_feat(ego_inputs)
        feature = self.pts_backbone.forward(pts_feature)
        feature = self.pts_neck.forward(feature)
        return feature
