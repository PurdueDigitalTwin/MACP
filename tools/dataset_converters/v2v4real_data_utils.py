import math
import os
import re
import sys

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import yaml


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), radians, angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = (
        torch.stack(
            (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones),
            dim=1).view(-1, 3, 3).float())
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


################################################### PCD ###################################################
def pcd_to_np(pcd_file):
    """Read  pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd : o3d.PointCloud
        PointCloud object, used for visualization
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)
    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def shuffle_points(points):
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]
    return points


def mask_ego_points(points):
    """Remove the lidar points of the ego vehicle itself.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """
    mask = ((points[:, 0] >= -1.95)
            & (points[:, 0] <= 2.95)
            & (points[:, 1] >= -1.1)
            & (points[:, 1] <= 1.1))
    points = points[np.logical_not(mask)]

    return points


def mask_points_by_range(points, limit_range):
    """Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """

    mask = ((points[:, 0] > limit_range[0])
            & (points[:, 0] < limit_range[3])
            & (points[:, 1] > limit_range[1])
            & (points[:, 1] < limit_range[4])
            & (points[:, 2] > limit_range[2])
            & (points[:, 2] < limit_range[5]))

    points = points[mask]
    return points


################################################### YAML ###################################################
def load_yaml(file, opt=None):
    """Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file)
    loader = yaml.Loader
    loader.add_implicit_resolver(
        'tag:yaml.org,2002:float',
        re.compile(
            """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list('-+0123456789.'),
    )
    param = yaml.load(stream, Loader=loader)
    if 'yaml_parser' in param:
        param = eval(param['yaml_parser'])(param)

    return param


################################################### BBOX ###################################################
def create_bbx(extent):
    """Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        Width, height, length of the bbx.

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """

    bbx = np.array([
        [extent[0], -extent[1], -extent[2]],
        [extent[0], extent[1], -extent[2]],
        [-extent[0], extent[1], -extent[2]],
        [-extent[0], -extent[1], -extent[2]],
        [extent[0], -extent[1], extent[2]],
        [extent[0], extent[1], extent[2]],
        [-extent[0], extent[1], extent[2]],
        [-extent[0], -extent[1], extent[2]],
    ])

    return bbx


def boxes_to_corners_3d(boxes3d, order):
    """4 -------- 5.

       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center.
    order : str
        'lwh' or 'hwl'
    Returns:
        corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.
    """
    # ^ z
    # |
    # |
    # | . x
    # |/
    # +-------> y

    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    if order == 'hwl':
        boxes3d[:, 3:6] = boxes3d[:, [5, 4, 3]]

    template = (
        boxes3d.new_tensor((
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
        )) / 2)

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3),
                                      boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d.numpy() if is_numpy else corners3d


def mask_boxes_outside_range_numpy(boxes,
                                   limit_range,
                                   order,
                                   min_num_corners=2):
    """
    Parameters
    ----------
    boxes: np.ndarray
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    limit_range: list
        [minx, miny, minz, maxx, maxy, maxz]
    min_num_corners: int
        The required minimum number of corners to be considered as in range.
    order : str
        'lwh' or 'hwl'
    Returns
    -------
    boxes: np.ndarray
        The filtered boxes.
    """
    assert boxes.shape[1] == 8 or boxes.shape[1] == 7

    new_boxes = boxes.copy()
    if boxes.shape[1] == 7:
        new_boxes = boxes_to_corners_3d(new_boxes, order)

    mask = ((new_boxes >= limit_range[0:3]) &
            (new_boxes <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return boxes[mask], mask


def corner_to_center(corner3d, order='lwh'):
    """Convert 8 corners to x, y, z, dx, dy, dz, yaw.

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)
    order : str
        'lwh' or 'hwl'

    Returns
    -------
    box3d : np.ndarray
        (N, 7)
    """
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)
    h = abs(
        np.mean(
            corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1, keepdims=True))
    l = (np.sqrt(
        np.sum(
            (corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]])**2,
            axis=1,
            keepdims=True,
        )) + np.sqrt(
            np.sum(
                (corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]])**2,
                axis=1,
                keepdims=True,
            )) + np.sqrt(
                np.sum(
                    (corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]])**2,
                    axis=1,
                    keepdims=True,
                )) + np.sqrt(
                    np.sum(
                        (corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]])**2,
                        axis=1,
                        keepdims=True,
                    ))) / 4

    w = (np.sqrt(
        np.sum(
            (corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]])**2,
            axis=1,
            keepdims=True,
        )) + np.sqrt(
            np.sum(
                (corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]])**2,
                axis=1,
                keepdims=True,
            )) + np.sqrt(
                np.sum(
                    (corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]])**2,
                    axis=1,
                    keepdims=True,
                )) + np.sqrt(
                    np.sum(
                        (corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]])**2,
                        axis=1,
                        keepdims=True,
                    ))) / 4

    theta = (np.arctan2(
        corner3d[:, 1, 1] - corner3d[:, 2, 1],
        corner3d[:, 1, 0] - corner3d[:, 2, 0],
    ) + np.arctan2(
        corner3d[:, 0, 1] - corner3d[:, 3, 1],
        corner3d[:, 0, 0] - corner3d[:, 3, 0],
    ) + np.arctan2(
        corner3d[:, 5, 1] - corner3d[:, 6, 1],
        corner3d[:, 5, 0] - corner3d[:, 6, 0],
    ) + np.arctan2(
        corner3d[:, 4, 1] - corner3d[:, 7, 1],
        corner3d[:, 4, 0] - corner3d[:, 7, 0],
    ))[:, np.newaxis] / 4

    if order == 'lwh':
        return np.concatenate([xyz, l, w, h, theta],
                              axis=1).reshape(batch_size, 7)
    elif order == 'hwl':
        return np.concatenate([xyz, h, w, l, theta],
                              axis=1).reshape(batch_size, 7)
    else:
        sys.exit('Unknown order')


def project_world_objects(object_dict, output_dict, transformation_matrix,
                          lidar_range, order):
    """Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    transformation_matrix : np.ndarray
        From current object to ego.

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'
    """
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']
        if 'ass_id' not in object_content:
            ass_id = object_id
        else:
            ass_id = object_content['ass_id']
        if 'obj_type' not in object_content:
            obj_type = 'Car'
        else:
            obj_type = object_content['obj_type']

        # Pedestrians are not considered yet (Only single class now)
        if obj_type == 'Pedestrian':
            continue

        object_pose = [
            location[0] + center[0],
            location[1] + center[1],
            location[2] + center[2],
            rotation[0],
            rotation[1],
            rotation[2],
        ]
        object2lidar = x1_to_x2(object_pose, transformation_matrix)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        bbx_lidar, _ = mask_boxes_outside_range_numpy(bbx_lidar, lidar_range,
                                                      order, 2)

        if bbx_lidar.shape[0] > 0:
            output_dict.update(
                {object_id: {
                    'coord': bbx_lidar,
                    'ass_id': ass_id
                }})


def project_points_by_matrix_torch(points, transformation_matrix):
    """Project the points to another coordinate system based on the
    transformation matrix.

    Parameters
    ----------
    points : torch.Tensor
        3D points, (N, 3)
    transformation_matrix : torch.Tensor
        Transformation matrix, (4, 4)
    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 3)
    """
    points, is_numpy = check_numpy_to_torch(points)
    transformation_matrix, _ = check_numpy_to_torch(transformation_matrix)

    # convert to homogeneous coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = F.pad(points, (0, 1), mode='constant', value=1)
    # (N, 4)
    projected_points = torch.einsum('ik, jk->ij', points_homogeneous,
                                    transformation_matrix)

    return (projected_points[:, :3]
            if not is_numpy else projected_points[:, :3].numpy())


################################################### Transformation ###################################################


def x1_to_x2(x1, x2):
    """Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list or np.ndarray
        The pose of x1 under world coordinates or transformation matrix x1->world
    x2 : list or np.ndarray
        The pose of x2 under world coordinates or transformation matrix x2->world
    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.
    """
    if isinstance(x1, list) and isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        x2_to_world = x_to_world(x2)
        world_to_x2 = np.linalg.inv(x2_to_world)
        transformation_matrix = np.dot(world_to_x2, x1_to_world)

    # object pose is list while lidar pose is transformation matrix
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        world_to_x2 = x2
        transformation_matrix = np.dot(world_to_x2, x1_to_world)
    # both are numpy matrix
    else:
        world_to_x2 = np.linalg.inv(x2)
        transformation_matrix = np.dot(world_to_x2, x1)

    return transformation_matrix


def x_to_world(pose):
    """The transformation matrix from x-coordinate system to carla world
    system.

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def dist_two_pose(cav_pose, ego_pose):
    """Calculate the distance between agent by given their pose."""
    if isinstance(cav_pose, list):
        distance = math.sqrt((cav_pose[0] - ego_pose[0])**2 +
                             (cav_pose[1] - ego_pose[1])**2)
    else:
        distance = math.sqrt((cav_pose[0, -1] - ego_pose[0, -1])**2 +
                             (cav_pose[1, -1] - ego_pose[1, -1])**2)
    return distance


######################################################################################################
