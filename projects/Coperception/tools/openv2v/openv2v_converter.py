"""Helper functions for converting OpenV2V data files to MMDet3D dataset.

Copyright (c) Purdue Digital Twin Lab. All rights reserved.
"""
from __future__ import annotations, print_function
import os
import os.path as osp
from dataclasses import asdict
from itertools import combinations
from multiprocessing import Manager, Process, set_start_method

import mmengine
import numpy as np
from PIL import Image
from tqdm import tqdm

from projects.Coperception.coperception.datasets.openv2v_utils import (
    DataInfo, ImageInfo, Instance, LidarPoints)
from projects.Coperception.coperception.visualization import V2V4RealVisualizer
from projects.Coperception.tools.openv2v.openv2v_data_utils import (
    clear_data_info_unused_keys, load_yaml, parse_all_files_and_tokens,
    pcd_to_np, project_points_by_matrix_torch, project_world_objects,
    split_dict, x1_to_x2, x_to_world)
from projects.Coperception.tools.visualize_data import visualize_sample

CATEGORIES: tuple[str] = ('car', )
MAX_AGENT_NUM: int = 0


def generate_object_centers(agent_info, reference_lidar_pose):
    """Retrieve all objects in a format of (n, 7) where n is the number of
    objects.

    Each object is represented by (x, y, z, l, w, h, yaw), i.e., (x, y, z, dx,
    dy, dz, yaw).
    """
    output_dict = {}
    project_world_objects(agent_info['vehicles'], output_dict,
                          reference_lidar_pose)
    object_bbxs = []
    object_ids = []
    for idx, (object_id, object_info) in enumerate(output_dict.items()):
        object_bbx = object_info['coord']
        object_bbxs.append(object_bbx[0, :])
        if object_info['ass_id'] != -1:
            object_ids.append(object_info['ass_id'])
        else:
            object_ids.append(object_id + 100 * int(agent_info['agent_id']))
    if len(object_bbxs) >= 1:
        object_bbxs = np.vstack(object_bbxs)
    else:
        object_bbxs = np.zeros((0, 7), dtype=np.float64)
    return object_bbxs, object_ids


def _get_item_single_agent(agent_info: dict, is_sim: bool):
    """Project the LiDAR points the bbxs to ego."""
    object_bbxs, object_ids = generate_object_centers(
        agent_info,
        reference_lidar_pose=agent_info['agent_lidar_to_ego_lidar']
        if not is_sim else agent_info['ego_lidar_pose'])

    points = agent_info['points']
    # TODO: remove points that hit itself
    points[:, :3] = project_points_by_matrix_torch(
        points[:, :3], agent_info['agent_lidar_to_ego_lidar'])
    # Add agent_id as an additional dim in points
    points = np.hstack([
        points,
        np.full(
            shape=(points.shape[0], 1),
            fill_value=float(agent_info['agent_id']),
            dtype=points.dtype)
    ])
    agent_info.update({
        'object_bbxs': object_bbxs,
        'object_ids': object_ids,
        'projected_points': points,
    })
    return agent_info


def _create_single_sample(
    sample_idx: int,
    root_path: str,
    cache_path: str,
    scenario_token: str,
    timestamp: str,
    ego_agent_id: str,
    agent_ids: list[str],
    early_fusion: bool = True,
    is_simulation: bool = True,
) -> dict:
    assert early_fusion, 'Only support early fusion now.'
    agent_annotations = {
        agent_id: load_yaml(
            os.path.join(root_path, scenario_token, agent_id,
                         timestamp + '.yaml'))
        for agent_id in agent_ids
    }
    # print(*agent_annotations[ego_agent_id].items(), sep='\n')

    ego_pose = agent_annotations[ego_agent_id]['true_ego_pos']
    ego_lidar_pose = agent_annotations[ego_agent_id]['lidar_pose']
    lidar2global = x_to_world(ego_lidar_pose)
    lidar2ego = x1_to_x2(ego_lidar_pose, ego_pose)
    ego2global: np.ndarray = x_to_world(ego_pose)

    projected_points_stack = []
    object_stack = []
    object_ids = []
    images = {}
    agent_idx = 0
    # loop over all CAVs to collect their projected points and objects
    for agent_id in agent_ids:
        agent_annotation = agent_annotations[agent_id]
        agent_lidar_pose = agent_annotation['lidar_pose']
        agent_info = dict()
        agent_info['agent_id'] = agent_id
        agent_info['ego_lidar_pose'] = ego_lidar_pose
        agent_info['agent_lidar_to_ego_lidar'] = x1_to_x2(
            agent_lidar_pose, ego_lidar_pose)
        agent_info['vehicles'] = agent_annotation['vehicles']
        agent_info['lidar_path'] = osp.join(root_path, scenario_token,
                                            agent_id, timestamp + '.pcd')
        agent_info['points'] = pcd_to_np(agent_info['lidar_path'])
        _get_item_single_agent(agent_info, is_simulation)

        # all these lidar and object coordinates are projected to ego already.
        projected_points_stack.append(agent_info['projected_points'])
        object_stack.append(agent_info['object_bbxs'])
        object_ids += agent_info['object_ids']

        # Currently only simulation data has images
        if is_simulation:
            # process camera data
            for cam_id in [
                    key for key in agent_annotation.keys()
                    if key.startswith('camera')
            ]:
                camera_props = agent_annotation[cam_id]
                cam_pose = camera_props['cords']
                cam2global = x_to_world(cam_pose)
                img_path = osp.join(root_path, scenario_token, agent_id,
                                    '_'.join([timestamp, cam_id]) + '.png')
                mmengine.check_file_exist(img_path)
                with Image.open(img_path) as img:
                    width, height = img.size

                cam2img = np.asarray(
                    camera_props['intrinsic'], dtype=np.float64)
                cam_extrinsic = np.asarray(
                    camera_props['extrinsic'], dtype=np.float64)
                cam2img_ext = np.eye(4, dtype=np.float64)
                cam2img_ext[0:3, 0:3] = cam2img
                elidar2cam = x1_to_x2(ego_lidar_pose, cam_pose)
                lidar2img = cam2img_ext @ elidar2cam
                # noinspection PyTypeChecker
                images['camera' + str(agent_idx) + cam_id[-1]] = ImageInfo(
                    img_path=img_path,
                    height=height,
                    width=width,
                    cam2img=cam2img_ext.tolist(),
                    lidar2img=lidar2img.tolist(),
                    lidar2cam=elidar2cam.tolist(),
                    cam2ego=x1_to_x2(cam_pose, ego_pose).tolist(),
                )
        agent_idx += 1

    # exclude all repetitive objects
    unique_indices = [object_ids.index(x) for x in set(object_ids)]
    object_stack = np.vstack(object_stack)[unique_indices]
    projected_points_stack = np.vstack(
        projected_points_stack)  # list to numpy array

    # convert to right hand coordinate
    object_stack[:, 1] = -object_stack[:, 1]  # left -> right hand coordinate
    object_stack[:, 6] = -object_stack[:, 6]  # left -> right hand coordinate
    projected_points_stack[:,
                           1] = -projected_points_stack[:,
                                                        1]  # left -> right hand coordinate

    lidar_path = osp.join(cache_path, scenario_token, ego_agent_id)
    os.makedirs(lidar_path, exist_ok=True)
    lidar_file_name = timestamp + '.' + '.'.join(sorted(agent_ids)) + '.bin'
    lidar_path = osp.join(lidar_path, lidar_file_name)
    assert not osp.exists(
        lidar_path), f'lidar_path {lidar_path} already exists.'
    with open(lidar_path, 'wb') as f:
        f.write(projected_points_stack.tobytes())

    instances = []
    for i in range(len(object_stack)):
        instance = Instance(
            bbox_3d=object_stack[i].tolist(),
            bbox_label_3d=0,  # Car only
            bbox_label=0,  # Car only
        )
        instances.append(instance)

    # noinspection PyTypeChecker
    lidar_points = LidarPoints(
        num_pts_feats=5,
        lidar2ego=lidar2ego.tolist(),
        lidar_path=lidar_path,
    )

    # noinspection PyTypeChecker
    data_info, is_empty = clear_data_info_unused_keys(
        asdict(
            DataInfo(
                sample_idx=str(sample_idx),
                token='.'.join([
                    scenario_token, ego_agent_id, timestamp,
                    '.'.join(sorted(agent_ids))
                ]),
                timestamp=float(timestamp),
                ego2global=ego2global.tolist(),
                images=images,
                lidar_points=lidar_points,
                instances=instances,
            ),
            # filter None attributes
            dict_factory=lambda d: {k: v
                                    for k, v in d if v is not None},
        ))
    assert not is_empty, f'DataInfo {data_info} is empty.'
    return data_info


def parser_worker(
    nproc: int,
    root_path: str,
    cache_path: str,
    token_dict: dict[str, list],
    out: list,
    is_simulation: bool = True,
    show: bool = False,
) -> None:
    """Single-process data info parser function.

    Args:
        nproc (int): The order of current process.
        root_path (str): The root directory of all data files.
        cache_path (str): The path to store the cached points.
        token_dict (dict[str, list]): The tokens of scenario_id and timestamp.
        out (list): The target list to store the results.
        show (bool): Whether to show the visualization of the data info.
        is_simulation (bool): Whether the data is from simulation or real world.
    """
    infos = []
    # CAV Data Augmentation
    num_samples = sum([
        len(list(combinations(agent_ids, MAX_AGENT_NUM))) * MAX_AGENT_NUM
        for agent_ids in token_dict.values()
    ]) if MAX_AGENT_NUM >= 1 else sum(
        [len(agent_ids) for agent_ids in token_dict.values()])
    pbar = tqdm(
        total=num_samples,
        desc=f'creating {cache_path} # {nproc}',
        leave=False,
        position=nproc,
        ascii=' >=',
        colour='green',
        unit=' case(s)',
    )
    visualizer = V2V4RealVisualizer() if show else None

    sample_idx = 0
    for token, agent_ids in token_dict.items():
        scenario_token, timestamp = token.split('.')
        if MAX_AGENT_NUM >= 1:
            agent_combinations = list(combinations(agent_ids, MAX_AGENT_NUM))
        else:
            agent_combinations = [agent_ids]
        for agent_subset in agent_combinations:
            for ego_agent_id in agent_subset:
                sample = _create_single_sample(
                    sample_idx,
                    root_path,
                    cache_path,
                    scenario_token,
                    timestamp,
                    ego_agent_id,
                    agent_subset,
                    is_simulation=is_simulation)

                if visualizer is not None and sample_idx % 100 == 0:
                    visualize_sample(sample, visualizer)
                infos.append(sample)
                sample_idx += 1
                pbar.update(1)

    assert len(
        infos
    ) == num_samples, f'len(infos) {len(infos)} != num_samples {num_samples}'
    pbar.close()
    out.append(infos)


def create_openv2v_infos(
    root_path: str,
    cache_path: str,
    info_prefix: str = 'openv2v',
    version: str = 'train',
    num_workers: int = 8,
    is_simulation: bool = True,
) -> None:
    """Create info files of OpenV2V dataset.

    Given the raw data, generate its related info file in '.pkl' format.

    Args:
        root_path (str): Root directory of all the data files.
        cache_path (str): Cache directory to store the cached files.
        info_prefix (str, optional): Prefix of the info files. Defaults to “openv2v”.
        version (str, optional): Version of the data. Defaults to “train”.
        num_workers (int, optional): Number of workers. Defaults to 8.
        is_simulation (bool, optional): Whether the data is from simulation or real world. Defaults to True.
    """
    data_list = []
    _, token_dict = parse_all_files_and_tokens(root_path, version)
    base_dir = os.path.join(root_path, version)
    suffix = str(MAX_AGENT_NUM) if MAX_AGENT_NUM >= 1 else 'x'
    info_path = os.path.join(root_path,
                             f'{info_prefix}_infos_{version}_{suffix}.pkl')
    out_dir = os.path.join(cache_path, version)
    if num_workers == 1:  # FOR DEBUG
        parser_worker(
            0,
            base_dir,
            out_dir,
            token_dict,
            data_list,
            is_simulation,
            show=True)

    if not os.path.exists(info_path):
        set_start_method('spawn')
        manager = Manager()
        return_list = manager.list()
        jobs: list[Process] = []
        sub_tokens = split_dict(token_dict, min(num_workers, os.cpu_count()))
        # sub_tokens = split_list(tokens, min(num_workers, cpu_count()))
        for idx, sub_token in enumerate(sub_tokens):
            p = Process(
                target=parser_worker,
                args=(idx, base_dir, out_dir, sub_token, return_list,
                      is_simulation))
            p.start()
            jobs.append(p)

        for proc in jobs:
            proc.join()

        for ret in return_list:
            data_list.extend(ret)
        assert len(data_list) > 0, 'Empty parsed annotations!'

    # Align sample index
    data_list.sort(key=lambda x: x['token'])
    for sample_idx, data in enumerate(data_list):
        assert 'sample_idx' in data
        data['sample_idx'] = str(sample_idx)

    print(f'There are {len(data_list)} samples in {version} split.')

    data = dict(
        data_list=data_list,
        metainfo={
            'categories': {
                'Car': 0
            },
            'dataset': 'openv2v',
            'version': version,
        },
    )
    mmengine.dump(data, info_path)
