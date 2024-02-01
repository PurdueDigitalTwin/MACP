import copy

import numpy as np
import torch

from mmdet3d.structures import LiDARInstance3DBoxes


def proj_lidar_bbox3d_to_img(bboxes_3d: LiDARInstance3DBoxes,
                             lidar2img: np.ndarray) -> np.ndarray:
    """
    Project 3D bbox to 2D image.
    Args:
        bboxes_3d: 3D bbox in LiDAR coordinate system to visualize.
        lidar2img: Transform matrix from LiDAR to image.

    Returns:

    """
    corners_3d = bboxes_3d.corners.cpu().numpy()
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=1)
    lidar2img = copy.deepcopy(lidar2img).reshape(4, 4)
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()

    pts_2d = pts_4d @ lidar2img.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    # pts_2d[:, 0] /= pts_2d[:, 2]
    # pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
    return imgfov_pts_2d
