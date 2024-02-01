import pickle
import random

import mmcv
import numpy as np
import torch
from mmengine.fileio import get
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.structures import LiDARInstance3DBoxes
from projects.Coperception.coperception.visualization import V2V4RealVisualizer

pkl_path = 'data/openv2v/openv2v_infos_validate.pkl'
load_dim = 5
use_dim = 4


def load_points(lidar_points: dict, load_dim=5, use_dim=4):
    pts_bytes = get(lidar_points['lidar_path'])
    points = np.frombuffer(pts_bytes, dtype=np.float32)
    points = points.reshape(-1, load_dim)
    return dict(points=points[:, :use_dim])


def load_multi_view_image(img_infos: dict):
    filename, lidar2img = [], []
    for _, cam_item in img_infos.items():
        filename.append(cam_item['img_path'])
        lidar2img.append(np.asarray(cam_item['lidar2img'], dtype=np.float64))

    img_bytes = [get(name) for name in filename]
    imgs = [mmcv.imfrombytes(img_byte) for img_byte in img_bytes]
    img = np.stack(imgs, axis=-1)
    results = dict(
        filename=filename,
        img=[img[..., i] for i in range(img.shape[-1])],
        lidar2img=lidar2img,
        img_shape=img.shape[:2],
    )
    return results


def visualize_sample(sample, visualizer: V2V4RealVisualizer):
    data_input = dict()
    input_meta = dict()
    instance_data = InstanceData()
    instance_data.bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor([instance['bbox_3d']
                      for instance in sample['instances']]),
        origin=(0.5, 0.5, 0.5))
    instance_data.labels_3d = torch.tensor(
        [instance['bbox_label_3d'] for instance in sample['instances']])

    points_dict = load_points(sample['lidar_points'])

    # image_dict = load_multi_view_image(sample['images'])

    data_input['points'] = points_dict['points']

    # data_input['img'] = image_dict['img']
    # input_meta['lidar2img'] = image_dict['lidar2img']

    data_3d = visualizer.draw_instances_3d(
        data_input,
        instance_data,
        input_meta,
        vis_task=
        'lidar_det'  # choices: 'multi-modality_det', 'mono_det', 'lidar_det'
    )
    visualizer.show(
        drawn_img_3d=data_3d['img'] if 'img' in data_3d else None, wait_time=0)
    visualizer.clear()

    # points = load_points(sample['lidar_points'])
    # # set point cloud in visualizer
    # visualizer.set_points(points)
    # # Draw 3D bboxes
    # for instance in sample['instances']:
    #     bbox_3d = instance['bbox_3d']
    #     bboxes_3d = LiDARInstance3DBoxes(
    #         torch.tensor([bbox_3d]),
    #         origin=(0.5, 0.5, 0.5))
    #     visualizer.draw_bboxes_3d(bboxes_3d, bbox_color=(0., 1., 0.), show_direction=True, vis_mode='add')
    # visualizer.show(wait_time=0)
    # visualizer.clear()


if __name__ == '__main__':
    visualizer: V2V4RealVisualizer = V2V4RealVisualizer()
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)['data_list']
    while True:
        i = random.randint(0, len(data_list) - 1)
        print(f'visualizing sample {i}')
        sample = data_list[i]
        visualize_sample(sample, visualizer)
