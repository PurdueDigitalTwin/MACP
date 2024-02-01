_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.Coperception.coperception'], allow_failed_imports=False)

# wild settings
is_debug = False
is_peft = True

# model settings
# Voxel size for voxel encoder.
# Usually voxel size is changed consistently with the point cloud range.
# If point cloud range is modified,
# do remember to change all related keys in the config.
# * indicates hparam used in OpenCOOD
voxel_size = [0.20, 0.075, 0.2]
point_cloud_range = [-140., -40., -3., 140., 40., 1.]
grid_size = [1440, 1440, 41]
post_center_range = [-160., -50., -10.0, 160., 50., 10.0]

# dataset settings
dataset_type = 'OpenV2VDataset'
data_root = 'data/openv2v/'
class_names = ['car']
num_classes = len(class_names)
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
num_pts_feats = 5
pts_use_dim = [0, 1, 2, 3, 4]

backend_args = None

model = dict(
    type='BEVFusionPEFT',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,  # *max_points_per_voxel=32
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[120000, 160000],  # *max_voxel=32000
            voxelize_reduce=True,
        ),
    ),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE', num_features=len(pts_use_dim)),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoderPEFT',
        in_channels=len(pts_use_dim),
        sparse_shape=grid_size,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=(
            (16, 16, 32),
            (32, 32, 64),
            (64, 64, 128),
            (128, 128),
        ),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock',
        peft_cfg=([
                      dict(
                          name=f'0_0_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_0_0_{k}',
                          in_features=16,
                          hidden_size=4,
                      ) for k in range(2)
                  ] + [
                      dict(
                          name=f'0_1_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_0_1_{k}',
                          in_features=16,
                          hidden_size=4,
                      ) for k in range(2)
                  ] + [
                      dict(
                          name=f'0_2_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_0_2_{k}',
                          in_features=32,
                          hidden_size=8,
                      ) for k in range(1)
                  ] + [
                      dict(
                          name=f'1_0_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_1_0_{k}',
                          in_features=32,
                          hidden_size=8,
                      ) for k in range(2)
                  ] + [
                      dict(
                          name=f'1_1_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_1_1_{k}',
                          in_features=32,
                          hidden_size=8,
                      ) for k in range(2)
                  ] + [
                      dict(
                          name=f'1_2_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_1_2_{k}',
                          in_features=64,
                          hidden_size=16,
                      ) for k in range(1)
                  ] + [
                      dict(
                          name=f'2_0_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_2_0_{k}',
                          in_features=64,
                          hidden_size=16,
                      ) for k in range(2)
                  ] + [
                      dict(
                          name=f'2_1_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_2_1_{k}',
                          in_features=64,
                          hidden_size=16,
                      ) for k in range(2)
                  ] + [
                      dict(
                          name=f'2_2_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_2_2_{k}',
                          in_features=128,
                          hidden_size=32,
                      ) for k in range(1)
                  ] + [
                      dict(
                          name=f'3_0_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_3_0_{k}',
                          in_features=128,
                          hidden_size=32,
                      ) for k in range(2)
                  ] + [
                      dict(
                          name=f'3_1_{k}_adapter',
                          type='SPAdapter',
                          upstream_name=
                          f'BEVFusionSparseEncoderPEFT_encoder_layers_3_1_{k}',
                          in_features=128,
                          hidden_size=32,
                      ) for k in range(2)
                  ]) if is_peft else None,
    ),
    pts_backbone=dict(
        type='SECONDPEFT',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        peft_cfg=([
                      dict(
                          name=f'block_0_{j}_ssf',
                          type='SSF',
                          upstream_name=f'SECONDPEFT_blocks_0_{j}',
                          in_features=128) for j in range(6)
                  ] + [
                      dict(
                          name=f'block_1_{j}_ssf',
                          type='SSF',
                          upstream_name=f'SECONDPEFT_blocks_1_{j}',
                          in_features=256) for j in range(6)
                  ]) if is_peft else None,
    ),
    pts_neck=dict(
        type='SECONDFPNPEFT',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True,
    ),
    bbox_head=dict(
        type='TransFusionHeadPEFT',
        num_proposals=100,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=num_classes,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayerPEFT',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(
                type='PositionEncodingLearnedPEFT',
                input_channel=2,
                num_pos_feats=128,
            ),
        ),
        train_cfg=dict(
            dataset='OpenV2V',
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15,
                ),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25),
            ),
        ),
        test_cfg=dict(
            dataset='OpenV2V',
            grid_size=grid_size,
            out_size_factor=8,
            voxel_size=voxel_size,
            pc_range=point_cloud_range,
            nms_type=None,
        ),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range,
            post_center_range=post_center_range,
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size,
            # code_size=10
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0,
        ),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
    ),
    peft_cfg=None,
    headonly=False,
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=pts_use_dim,
        backend_args=backend_args,
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(
        type=
        'GlobalRotScaleTrans',  # *random_world_rotation, *random_world_scaling
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.78539816, 0.78539816],
    ),
    dict(type='BEVFusionRandomFlip3D'),  # *random_world_flip
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=['car']),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=pts_use_dim,
        backend_args=backend_args,
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='openv2v_infos_train_x.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR',
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='openv2v_infos_test_x.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
    ),
)
test_dataloader = val_dataloader if not is_debug else dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='SubsetSampler', indices=list(range(0, 12000, 100))),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='openv2v_infos_test_x.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
    ),
)

val_evaluator = dict(
    type='V2V4RealMetric',
    ann_file=data_root + 'openv2v_infos_test_x.pkl',
    pcd_limit_range=point_cloud_range,  # the final range for evaluation
    score_threshold=0.2,
    nms_iou_threshold=0.15,
)
test_evaluator = val_evaluator

total_epochs: int = 20

# learning rate
lr = 2e-5
param_scheduler = [
    # learning rate scheduler
    # During the first 2/5 epochs, learning rate increases from 0 to lr * 10
    # during the next 3/5 epochs, learning rate decreases from lr * 10 to lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=int(0.4 * total_epochs),
        eta_min=lr * 10,
        begin=0,
        end=int(0.4 * total_epochs),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=int(0.6 * total_epochs),
        eta_min=lr * 1e-4,
        begin=int(0.4 * total_epochs),
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first 2/5 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 3/5 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=int(0.4 * total_epochs),
        eta_min=0.85 / 0.95,
        begin=0,
        end=int(0.4 * total_epochs),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingMomentum',
        T_max=int(0.6 * total_epochs),
        eta_min=1,
        begin=int(0.4 * total_epochs),
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=total_epochs, val_interval=4)
val_cfg = dict()
test_cfg = dict()

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    checkpoint=dict(type='CheckpointHook', interval=2),
    visualization=dict(
        type='V2V4RealVisualizationHook', draw=True, show=True, wait_time=2))
load_from = 'data/models/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth'
