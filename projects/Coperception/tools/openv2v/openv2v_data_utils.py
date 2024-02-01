"""Helper function for processing opencood data.

Copyright (c) Purdue Digital Twin Lab. All rights reserved.
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

# Constants
AVAILABLE_VERS = [
    'train',
    'validate',
    'test',
    'test_culver_city',
    'test_additional',
]
"""A list of all available data versions."""


def load_yaml(filepath: str) -> dict:
    """Load OpenV2V dataset yaml files.

    Args:
        filepath (str): directory to the `.yaml` file.

    Returns:
        dict: data annotations or configurations.
    """
    with open(filepath) as file:
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
        param = yaml.load(file, Loader=loader)
        if 'yaml_parser' in param:
            param = eval(param['yaml_parser'])(param)

    return param


# def pose3d_to_transform(pose: Iterable[float],
#                         in_degree: bool = True) -> npt.NDArray[np.float64]:
#     """Compute and returns a 4x4 transformation matrix given a 3D pose.
#
#     Args:
#         pose (Iterable[float]): A 3D pose in Euler angles `[x, y, z, roll, yaw, pitch]`.
#         in_degree (bool, optional): If angles are in degrees. Defaults to True.
#
#     Returns:
#         npt.NDArray[np.float64]: The 4x4 transformation matrix projecting global 3D
#         points into the coordinate system about the given pose.
#     """
#     assert len(
#         pose
#     ) == 6, f'Invalid input size. Expect 6 floats, but got {len(pose)}.'
#     x, y, z, roll, yaw, pitch = pose
#
#     if in_degree:
#         roll = np.deg2rad(roll)
#         yaw = np.deg2rad(yaw)
#         pitch = np.deg2rad(pitch)
#
#     rot_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
#                       [0, np.sin(roll), np.cos(roll)]])
#     rot_y = np.array([
#         [np.cos(pitch), 0, np.sin(pitch)],
#         [0, 1, 0],
#         [-np.sin(pitch), 0, np.cos(pitch)],
#     ])
#     rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                       [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
#     rotation = np.dot(rot_z, np.dot(rot_y, rot_x))
#
#     transform = np.array([
#         [rotation[0, 0], rotation[0, 1], rotation[0, 2], x],
#         [rotation[1, 0], rotation[1, 1], rotation[1, 2], y],
#         [rotation[2, 0], rotation[2, 1], rotation[2, 2], z],
#         [0, 0, 0, 1],
#     ])
#
#     return transform
#
# def _flatten_dict(data: dict) -> list:
#     ret = []
#     for key, values in data.items():
#         if isinstance(values, list):
#             ret.extend(['.'.join([key, val]) for val in values])
#         if isinstance(values, dict):
#             ret.extend(
#                 ['.'.join([key, val]) for val in _flatten_dict(values)])
#
#     return ret


def _build_df(data: dict) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(columns=['scenario_token', 'agent_id', 'timestamp'])
    for scenario_token, agents in data.items():
        for agent_id, timestamps in agents.items():
            df = pd.concat(
                [
                    df,
                    pd.DataFrame({
                        'scenario_token': scenario_token,
                        'agent_id': agent_id,
                        'timestamp': timestamps,
                    }),
                ],
                ignore_index=True,
            )
    grouped_df = df.groupby(['scenario_token', 'timestamp'])
    # Counting the number of cavs in each scenario and timestamp
    count_df = grouped_df.size().reset_index(name='Count')
    counts = count_df['Count'].tolist()

    print(f'{min(counts)} cavs in the least crowded scenario.')
    print(f'{max(counts)} cavs in the most crowded scenario.')

    result_dict = {}
    # Iterate over the grouped dataframe
    for group, data in grouped_df:
        # Concatenate `scenario_token` and `timestamp` to form a unique key
        key = '.'.join([str(x) for x in group])
        # Get the list of agent_ids in the group
        agent_ids = data['agent_id'].tolist()
        # Add the key-value pair to the result dictionary
        result_dict[key] = agent_ids

    return df, result_dict


def parse_all_files_and_tokens(
    root_path: str,
    version: str,
) -> Any:
    assert (os.path.isdir(root_path) and len(os.listdir(root_path)) > 0
            ), f'Invalid data root_path directory: {root_path}.'
    assert version in AVAILABLE_VERS, f'Invalid data version {version}.'

    _db_struct = {
        scenario_token: {
            agent_id: sorted(
                list({
                    str(filename.name).split('.')[0].split('_')[0]
                    for filename in Path(root_path, version, scenario_token,
                                         agent_id).glob('*')
                }))
            for agent_id in os.listdir(
                os.path.join(root_path, version, scenario_token))
            if os.path.isdir(
                os.path.join(root_path, version, scenario_token, agent_id))
        }
        for scenario_token in os.listdir(os.path.join(root_path, version))
    }

    return _build_df(_db_struct)
    # return _db_struct, flatten_dict(_db_struct)


def clear_instance_unused_keys(instance):
    keys = list(instance.keys())
    for k in keys:
        if instance[k] is None:
            del instance[k]
    return instance


def clear_data_info_unused_keys(data_info):
    keys = list(data_info.keys())
    empty_flag = True
    for key in keys:
        # we allow no annotations in datainfo
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag is False:
                empty_flag = False
            else:
                # sub field is empty
                del data_info[key]
        else:
            empty_flag = False

    return data_info, empty_flag


def x_to_world(pose: list) -> np.ndarray:
    """The transformation matrix from x-coordinate system to carla world
    system.

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The 4x4 transformation matrix
    """
    if not isinstance(pose, list):
        return pose

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
        The 4x4 transformation matrix
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


def project_world_objects(object_dict,
                          output_dict,
                          transformation_matrix,
                          order='lwh'):
    """Project the objects under world coordinates into another coordinate
    based on the provided transformation matrix.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.
    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).
    transformation_matrix : np.ndarray
        From current object to ego.
    order : str
        'lwh'
    """
    assert order == 'lwh', 'Only support lwh now.'
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
        # bbx_lidar, _ = mask_boxes_outside_range_numpy(bbx_lidar, lidar_range, order, 2)

        if bbx_lidar.shape[0] > 0:
            output_dict.update(
                {object_id: {
                    'coord': bbx_lidar,
                    'ass_id': ass_id
                }})


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


def corner_to_center(corner3d, order='lwh'):
    """Convert 8 corners to x, y, z, dx, dy, dz, yaw.

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)
    order : str
        'lwh'

    Returns
    -------
    box3d : np.ndarray (N, 7)
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
    # elif order == 'hwl':
    #     return np.concatenate([xyz, h, w, l, theta],
    #                           axis=1).reshape(batch_size, 7)
    else:
        sys.exit('Unsupported order')


def pcd_to_np(pcd_file):
    """Read pcd and return numpy array.

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


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


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
    points_homogeneous = F.pad(
        points, (0, 1), mode='constant', value=1)  # (N, 4)
    projected_points = torch.einsum('ik, jk->ij', points_homogeneous,
                                    transformation_matrix)  # (N, 4)

    return (projected_points[:, :3]
            if not is_numpy else projected_points[:, :3].numpy())


def split_list(data: list[Any], n: int) -> list[list[Any]]:
    """Split the list into `n` sublists.

    Args:
        data (list[Any]): The input list to split.
        n (int): The number of sublists to generate.

    Returns:
        list[list[Any]]: A list of generated sublists.
    """
    ret = []
    k, m = divmod(len(data), n)
    for i in range(n):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        ret.append(data[start:end])

    return ret


def split_dict(data: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """Split the dict into `n` subdicts.

    Args:
        data (dict[str, Any]): The input dict to split.
        n (int): The number of subdicts to generate.

    Returns:
        list[dict[str, Any]]: A list of generated subdicts.
    """
    ret = []
    k, m = divmod(len(data), n)
    for i in range(n):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        ret.append({k: data[k] for k in list(data.keys())[start:end]})

    return ret


if __name__ == '__main__':
    pass
