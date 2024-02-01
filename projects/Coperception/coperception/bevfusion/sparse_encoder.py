# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch.nn as nn

from mmdet3d.models.layers import make_sparse_convmodule
from mmdet3d.models.layers.sparse_block import (SparseBasicBlock,
                                                replace_feature)
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.middle_encoders import SparseEncoder
from mmdet3d.registry import MODELS
from projects.Coperception.coperception._typing import OptPEFTConfig
from projects.Coperception.coperception.peft import PEFTConfigCollection
from projects.Coperception.coperception.peft.config import \
    build_layers_from_configs
from projects.Coperception.coperception.peft.tools import (
    freeze_module, get_peft_layer, unfreeze_module, unfreeze_module_with_name)

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor  # type: ignore
else:
    from mmcv.ops import SparseConvTensor

import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor
from spconv.pytorch import functional as FSP
from spconv.pytorch.modules import is_spconv_module


@MODELS.register_module()
class BEVFusionSparseEncoder(SparseEncoder):
    r"""Sparse encoder for BEVFusion. The difference between this implementation
    and that of ``SparseEncoder`` is that the shape order of 3D conv is (H, W,
    D) in ``BEVFusionSparseEncoder`` rather than (D, H, W) in
    ``SparseEncoder``. This difference comes from the implementation of
    ``voxelization``.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """

    def __init__(
        self,
        in_channels,
        sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='conv_module',
        return_middle_feats=False,
    ):
        super(SparseEncoder, self).__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        self.return_middle_feats = return_middle_feats
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ),
            )
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
            )

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type,
        )

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d',
        )

    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
                include:

            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_middle_feats is True, the
                module returns middle features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features


@MODELS.register_module()
class BEVFusionSparseEncoderPEFT(BEVFusionSparseEncoder):
    r"""Sparse encoder for BEVFusion. The difference between this implementation
    and that of ``SparseEncoder`` is that the shape order of 3D conv is (H, W,
    D) in ``BEVFusionSparseEncoder`` rather than (D, H, W) in
    ``SparseEncoder``. This difference comes from the implementation of
    ``voxelization``.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """

    def __init__(
        self,
        in_channels: int,
        sparse_shape: List[int],
        order: List[str] = ('conv', 'norm', 'act'),
        norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels: int = 16,
        output_channels: int = 128,
        encoder_channels: Tuple[Tuple[int]] = ((16, ), (32, 32, 32),
                                               (64, 64, 64), (64, 64, 64)),
        encoder_paddings: Tuple[Tuple[int]] = ((1, ), (1, 1, 1), (1, 1, 1),
                                               ((0, 1, 1), 1, 1)),
        block_type: str = 'conv_module',
        return_middle_feats: bool = False,
        peft_cfg: OptPEFTConfig = None,
        num_peft_branches: int = 1,
    ):
        super().__init__(in_channels, sparse_shape, order, norm_cfg,
                         base_channels, output_channels, encoder_channels,
                         encoder_paddings, block_type, return_middle_feats)

        if peft_cfg is not None:
            if isinstance(peft_cfg, dict):
                peft_cfg = [peft_cfg]
            self.num_peft_branches = num_peft_branches
            self.peft_cfg = PEFTConfigCollection(peft_cfg)
            self.peft_layers = nn.ModuleDict(
                build_layers_from_configs(
                    self.peft_cfg, repeats=self.num_peft_branches))
            freeze_module(self)
            if len(self.peft_layers) > 0:
                unfreeze_module(self.conv_input)
                unfreeze_module_with_name(self, 'peft_layers')

    def forward(self, voxel_features, coors, batch_size, peft_branch_idx=-1):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.
            peft_branch_idx (int): PEFT branch index.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
                include:

            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_middle_feats is True, the
                module returns middle features.
        """
        if not hasattr(self, 'peft_cfg'):
            return super().forward(voxel_features, coors, batch_size)
        else:
            coors = coors.int()
            input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                               self.sparse_shape, batch_size)
            x: SparseConvTensor = self.conv_input(input_sp_tensor)

            prefix = '_'.join([self.__class__.__name__, 'encoder_layers'])
            for i in range(len(self.encoder_layers)):
                encoder_layer = self.encoder_layers[i]
                for j in range(len(encoder_layer)):
                    peft_idx = 0
                    block = encoder_layer[j]
                    if isinstance(block, spconv.SparseSequential):
                        for f in block:
                            assert isinstance(
                                x, SparseConvTensor
                            ), f'x should be SparseConvTensor, got {type(x)}'
                            if is_spconv_module(f):
                                x = f(x)
                            else:
                                if isinstance(x, SparseConvTensor):
                                    if x.indices.shape[0] != 0:
                                        x = x.replace_feature(f(x.features))
                                else:
                                    x = f(x)

                            if isinstance(
                                    f,
                                (spconv.SparseConv3d, spconv.SubMConv3d)):
                                peft_layer = get_peft_layer(
                                    self.peft_cfg, self.peft_layers,
                                    f'{prefix}_{i}_{j}_{peft_idx}',
                                    peft_branch_idx)
                                if peft_layer is not None:
                                    x = peft_layer(x)
                                    peft_idx += 1
                    elif isinstance(block, SparseBasicBlock):
                        identity = x.features

                        assert x.features.dim(
                        ) == 2, f'x.features.dim()={x.features.dim()}'

                        xo = block.conv1(x)
                        peft_layer = get_peft_layer(self.peft_cfg,
                                                    self.peft_layers,
                                                    f'{prefix}_{i}_{j}_0',
                                                    peft_branch_idx)
                        if peft_layer is not None:
                            xo = peft_layer(xo)
                        xo = replace_feature(xo, block.norm1(xo.features))
                        xo = replace_feature(xo, block.relu(xo.features))

                        xo = block.conv2(xo)
                        peft_layer = get_peft_layer(self.peft_cfg,
                                                    self.peft_layers,
                                                    f'{prefix}_{i}_{j}_1',
                                                    peft_branch_idx)
                        if peft_layer is not None:
                            xo = peft_layer(xo)
                        xo = replace_feature(xo, block.norm2(xo.features))

                        assert block.downsample is None

                        xo = replace_feature(xo, xo.features + identity)
                        xo = replace_feature(xo, block.relu(xo.features))
                        x = xo

                    else:
                        raise NotImplementedError

            out = self.conv_out(x)
            spatial_features = out.dense()

            N, C, H, W, D = spatial_features.shape
            spatial_features = spatial_features.permute(0, 1, 4, 2,
                                                        3).contiguous()
            spatial_features = spatial_features.view(N, C * D, H, W)

            return spatial_features
