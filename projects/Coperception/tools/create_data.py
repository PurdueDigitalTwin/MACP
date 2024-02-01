"""Entrypoint for OpenV2V data preparation.

Copyright (c) Purdue Digital Twin Lab. All rights reserved.
"""
from __future__ import annotations, print_function
import os
from typing import Optional, Union

import mmengine
import numpy.typing as npt

from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops
from projects.Coperception.tools.openv2v.openv2v_converter import \
    create_openv2v_infos

# Type aliases
_PathLike = Union[str, 'os.PathLike[str]']


def create_groundtruth_database(
    dataset_class_name: str,
    data_path: str,
    info_prefix: str,
    info_path: Optional[_PathLike] = None,
    database_path: Optional[_PathLike] = None,
    db_info_path: Optional[_PathLike] = None,
) -> None:
    """Given the preprocessed raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info files.
        info_path (Optional[_PathLike], optional): Path to info files. Defaults to None.
        database_path (Optional[_PathLike], optional): Path to save db. Defaults to None
        db_info_path (Optional[_PathLike], optional): Path to db_info. Defaults to None.
    """
    print(f'Creating ground-truth database of {dataset_class_name}...')

    dataset_cfg = {
        'type': dataset_class_name,
        'data_root': data_path,
        'ann_file': info_path,
    }
    if dataset_class_name == 'OpenV2VDataset':
        # TODO: create OpenV2VDataset
        dataset_cfg.update(
            data_prefix={
                'pts': 'train',
                'img': 'train'
            },
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=3,
                    use_dim=3),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
            ],
        )
    else:
        raise NotImplementedError
    dataset: Det3DDataset = DATASETS.build(dataset_cfg)

    if database_path is None:
        database_path = os.path.join(data_path, f'{info_prefix}_gt_database')
    if db_info_path is None:
        db_info_path = os.path.join(data_path,
                                    f'{info_prefix}_db_infos_train.pkl')
    mmengine.mkdir_or_exist(database_path)
    all_db_infos = {}

    for j in mmengine.track_iter_progress(range(len(dataset))):
        data_info = dataset.get_data_info(j)
        sample = dataset.pipeline(data_info)
        annos = sample['ann_info']
        image_idx = sample['sample_idx']
        points: npt.NDArray = sample['points'].numpy()
        gt_boxes_3d: npt.NDArray = annos['gt_bboxes_3d'].numpy()
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = os.path.join(database_path, filename)
            rel_filepath = os.path.join(f'{info_prefix}_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, 0:3] -= gt_boxes_3d[i, 0:3]

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            db_info = {
                'name': names[i],
                'path': rel_filepath,
                'image_idx': image_idx,
                'gt_idx': i,
                'box3d_lidar': gt_boxes_3d[i],
                'num_points_in_gt': gt_points.shape[0],
            }

            if names[i] in all_db_infos:
                all_db_infos[names[i]].append(db_info)
            else:
                all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'Loaded {len(v)} {k} database infos')

    with open(db_info_path, 'wb') as f:
        mmengine.dump(all_db_infos, f)

    print(f'Creating ground-truth database of {dataset_class_name}...DONE!')


def openv2v_data_prep(
    root_path: _PathLike,
    cache_path: _PathLike,
    info_prefix: str,
    version: str,
    num_workers: int = 1,
    is_simulation: bool = True,
) -> None:
    """Prepare data related to the OpenV2V/V2V4Real dataset.

    Args:
        root_path (_PathLike): Path to the dataset root directory.
        cache_path (_PathLike): Path to the cache directory.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version string.
        num_workers (int, optional): Number of workers to create data.
        is_simulation (bool, optional): Whether the data is from simulation.
    """
    create_openv2v_infos(
        root_path=root_path,
        cache_path=cache_path,
        info_prefix=info_prefix,
        version=version,
        num_workers=num_workers,
        is_simulation=is_simulation)

    # TODO: create ground-truth database function implementation
    # NOTE: failed to load point cloud data ->
    # ValueError: buffer size must be a multiple of element size
    # at mmdet3d.datasets.transforms.loading:619
    # create_groundtruth_database(
    #     dataset_class_name="OpenV2VDataset",
    #     data_path=root_path,
    #     info_prefix=info_prefix,
    #     info_path=f"{info_prefix}_infos_{version}.pkl",
    # )


if __name__ == '__main__':
    import argparse

    from mmdet3d.utils import register_all_modules
    register_all_modules()

    parser = argparse.ArgumentParser(description='Dataset preparation.')
    parser.add_argument(
        '--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument(
        '--root-path',
        type=str,
        required=True,
        help='Path to the dataset root.')
    parser.add_argument(
        '--cache-path',
        type=str,
        default=None,
        help='Path to the cached data.')
    parser.add_argument(
        '--info-prefix',
        type=str,
        default='openv2v',
        help='Prefix tags of info files.')
    parser.add_argument(
        '--version', type=str, default='train', help='Dataset version.')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of workers to create infos.')

    args = parser.parse_args()
    if args.dataset == 'openv2v' or 'v2v4real':
        openv2v_data_prep(
            root_path=args.root_path,
            cache_path=args.cache_path,
            info_prefix=args.info_prefix,
            version=args.version,
            num_workers=args.num_workers,
            is_simulation=True if args.dataset == 'openv2v' else False,
        )
    else:
        raise NotImplementedError(
            f'Dataset named {args.dataset} is not supported yet.')
