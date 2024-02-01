"""See issue: https://github.com/open-mmlab/mmdetection3d/issues/2584 Fix the
issue of pts_middle_encoder."""
import glob
import os

import torch

dirs = [
    '/media/mayson/SamsungSSD/github/mmdet3d1.x/data/models/MACP/macp_v2v4real_c256',
]
path = ''
# path = '/work_dirs/bf_peft_mid_openv2v/20230806_123337/epoch_2.pth'
mismatch_list = [
    'pts_middle_encoder.conv_input.0.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer1.0.conv1.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer1.0.conv2.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer1.1.conv1.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer1.1.conv2.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer1.2.0.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer2.0.conv1.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer2.0.conv2.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer2.1.conv1.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer2.1.conv2.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer2.2.0.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer3.0.conv1.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer3.0.conv2.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer3.1.conv1.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer3.1.conv2.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer3.2.0.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer4.0.conv1.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer4.0.conv2.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer4.1.conv1.weight',
    'pts_middle_encoder.encoder_layers.encoder_layer4.1.conv2.weight',
    'pts_middle_encoder.conv_out.0.weight',
]

mismatch_list += [
    'pts_middle_encoder.peft_layers.0_0_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.0_0_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.0_0_1_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.0_0_1_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.0_1_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.0_1_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.0_1_1_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.0_1_1_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.0_2_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.0_2_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.1_0_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.1_0_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.1_0_1_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.1_0_1_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.1_1_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.1_1_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.1_1_1_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.1_1_1_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.1_2_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.1_2_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.2_0_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.2_0_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.2_0_1_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.2_0_1_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.2_1_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.2_1_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.2_1_1_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.2_1_1_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.2_2_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.2_2_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.3_0_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.3_0_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.3_0_1_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.3_0_1_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.3_1_0_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.3_1_0_adapter.adapter.2.weight',
    'pts_middle_encoder.peft_layers.3_1_1_adapter.adapter.0.weight',
    'pts_middle_encoder.peft_layers.3_1_1_adapter.adapter.2.weight',
]


def fix_ckpt(p):
    model = torch.load(p)
    for key in model['state_dict'].keys():
        if key in mismatch_list:
            print('fixing', key)
            model['state_dict'][key] = torch.transpose(
                model['state_dict'][key], 0, 1)
            model['state_dict'][key] = torch.transpose(
                model['state_dict'][key], 1, 2)
            model['state_dict'][key] = torch.transpose(
                model['state_dict'][key], 2, 3)
            model['state_dict'][key] = torch.transpose(
                model['state_dict'][key], 3, 4)
    torch.save(model, p.replace('.pth', '_fixed.pth'))


if __name__ == '__main__':
    if path:
        fix_ckpt(path)
    else:
        for d in dirs:
            for p in glob.glob(os.path.join(d, '*.pth')):
                fix_ckpt(p)
