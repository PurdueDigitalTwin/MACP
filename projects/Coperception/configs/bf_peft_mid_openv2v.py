_base_ = [
    './bf_peft_openv2v.py',
]
num_peft_branches = 1
pts_use_dim = 4

model = dict(
    type='BEVFusionPEFTMid',
    num_branches=num_peft_branches,
    pts_voxel_encoder=dict(num_features=pts_use_dim),
    pts_middle_encoder=dict(
        in_channels=pts_use_dim,
        num_peft_branches=num_peft_branches,
    ),
    fusion_layer=dict(
        type='ConAda',
        aggregation='weighted_sum',
        fusion='conv',
        compress_ratio=4,
    ))
