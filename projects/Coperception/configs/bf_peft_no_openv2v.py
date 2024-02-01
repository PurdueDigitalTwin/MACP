_base_ = [
    './bf_peft_openv2v.py',
]
pts_use_dim = 4

model = dict(
    type='BEVFusionPEFTNo',
    pts_voxel_encoder=dict(num_features=pts_use_dim),
    pts_middle_encoder=dict(in_channels=pts_use_dim),
)
