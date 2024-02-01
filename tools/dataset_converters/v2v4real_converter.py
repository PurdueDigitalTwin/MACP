import itertools
import os
import os.path as osp
from collections import OrderedDict
from pathlib import Path

import mmengine
import numpy as np

from tools.dataset_converters.update_infos_to_v2 import (
    clear_data_info_unused_keys, clear_instance_unused_keys,
    get_empty_instance, get_empty_standard_data_info)
from tools.dataset_converters.v2v4real_data_utils import (
    corner_to_center, create_bbx, dist_two_pose, load_yaml,
    mask_boxes_outside_range_numpy, mask_ego_points, mask_points_by_range,
    pcd_to_np, project_points_by_matrix_torch, project_world_objects,
    shuffle_points, x1_to_x2)

cav_list = ('0', '1')
cav_permutations = tuple(itertools.permutations(cav_list))


class V2VSetting:
    dataset: str = 'v2v4real'
    seed: int = 20
    async_flag: bool = False
    async_overhead: float = 60.0
    async_model: str = 'sim'  # choices = ['sim', 'real']
    data_size: float = 0.0  # Mb
    transmission_speed: float = 27.0  # Mbps
    backbone_delay: float = 0.0  # ms
    loc_err_flag: bool = False
    xyz_std: float = 0.2
    ryp_std: float = 0.2
    cur_ego_pose_flag: bool = True  # whether to use current ego pose to calculate transformation matrix.

    gt_range = [-100, -40, -5, 100, 40, 3]  # the final range for evaluation
    order: str = 'lwh'
    max_num: int = 100  # maximum number of objects in a single frame, make sure different frames has the same dimension in the same batch
    is_sim: bool = False
    cav_lidar_range = [
        -140.8,
        -40,
        -5,
        140.8,
        40,
        3,
    ]  # lidar range for each individual cav.


def create_v2v4real_infos(data_path: str, info_prefix):
    """Create info file of v2v4real dataset.

    Given the raw data, generate its related info file in pkl format

    Args:
        data_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.

    Returns:
    """
    data_path = Path(data_path)
    train_scenes = sorted([str(p) for p in (data_path / 'train').iterdir()])
    test_scenes = sorted([str(p) for p in (data_path / 'test').iterdir()])
    v2v4real_test, test_len_list = _build_database(test_scenes)
    v2v4real_train, train_len_list = _build_database(train_scenes)
    test_infos = _fill_infos(v2v4real_test, test_len_list)
    train_infos = _fill_infos(v2v4real_train, train_len_list)

    metadata = {k: v for k, v in vars(V2VSetting).items() if '__' not in k}

    print('train sample: {}, test sample: {}'.format(
        len(train_infos), len(test_infos)))
    data = dict(data_list=train_infos, metainfo=metadata)
    info_path = osp.join(data_path, f'{info_prefix}_infos_train.pkl')
    mmengine.dump(data, info_path)
    data['data_list'] = test_infos
    info_test_path = osp.join(data_path, f'{info_prefix}_infos_test.pkl')
    mmengine.dump(data, info_test_path)


def _get_timestamp_key(scene_dict: OrderedDict, timestamp_index):
    """
    Given the timestamp index, return the correct timestamp key, e.g. 2 --> '000078'.
    Parameters
    ----------
    scene_dict : OrderedDict
        The dictionary contains all contents in the current scene.
    timestamp_index : int
        The index for timestamp.
    Returns
    -------
    timestamp_key : str
        The timestamp key saved in the cav dictionary.
    """
    # get all timestamp keys
    timestamp_keys = list(scene_dict.items())
    timestamp_keys = timestamp_keys[0][1]
    # retrieve the correct index
    timestamp_key = list(timestamp_keys.items())[timestamp_index][0]
    return timestamp_key


def _calc_dist_to_ego(scene_dict, timestamp_key):
    """Calculate the distance to ego for each cav."""
    ego_lidar_pose = None
    ego_cav_content = None
    # Find ego pose first
    for cav_id, cav_content in scene_dict.items():
        if cav_content['ego']:
            ego_cav_content = cav_content
            ego_lidar_pose = load_yaml(
                cav_content[timestamp_key]['yaml'])['lidar_pose']
            break
    assert ego_lidar_pose is not None
    for cav_id, cav_content in scene_dict.items():
        cur_lidar_pose = load_yaml(
            cav_content[timestamp_key]['yaml'])['lidar_pose']
        distance = dist_two_pose(cur_lidar_pose, ego_lidar_pose)
        cav_content['distance_to_ego'] = distance
        scene_dict.update({cav_id: cav_content})

    return ego_cav_content


def _time_delay_calculation(ego_flag):
    """Calculate the time delay for a certain vehicle.

    Parameters
    ----------
    ego_flag : boolean
        Whether the current cav is ego.

    Return
    ------
    time_delay : int
        The time delay quantization.
    """
    # there is no time delay for ego vehicle
    if ego_flag or not V2VSetting.async_flag:
        return 0
    if V2VSetting.async_model == 'real':
        # noise/time is in ms unit
        overhead_noise = np.random.uniform(0, V2VSetting.async_overhead)
        tc = V2VSetting.data_size / V2VSetting.transmission_speed * 1000
        time_delay = int(overhead_noise + tc + V2VSetting.backbone_delay)
    elif V2VSetting.async_model == 'sim':
        time_delay = int(np.abs(V2VSetting.async_overhead))
    else:
        raise NotImplementedError

    time_delay = (time_delay // 100
                  )  # current 10hz, we may consider 20hz in the future
    return time_delay


def _add_loc_noise(pose):
    # TODO: move to V2V4RealDataset
    np.random.seed(V2VSetting.seed)
    xyz_noise = np.random.normal(0, V2VSetting.xyz_std, 3)
    ryp_std = np.random.normal(0, V2VSetting.ryp_std, 3)
    noise_pose = [
        pose[0] + xyz_noise[0],
        pose[1] + xyz_noise[1],
        pose[2] + xyz_noise[2],
        pose[3],
        pose[4] + ryp_std[1],
        pose[5],
    ]
    return noise_pose


def _generate_object_center(cav_contents, reference_lidar_pose):
    """Retrieve all objects in a format of (n, 7), where 7 represents x, y, z,
    l, w, h, yaw or x, y, z, h, w, l, yaw.

    Parameters
    ----------
    cav_contents : list
        List of dictionary, save all cavs' information.
    reference_lidar_pose : np.ndarray
        The final target lidar pose with length 6.

    Returns
    -------
    object_np : np.ndarray
        Shape is (max_num, 7).
    mask : np.ndarray
        Shape is (max_num,).
    object_ids : list
        Length is number of bbx in current sample.
    """
    tmp_object_dict = {}
    for cav_content in cav_contents:
        tmp_object_dict.update(cav_content['params']['vehicles'])
        cav_id = cav_content['cav_id']

    output_dict = {}
    filter_range = V2VSetting.gt_range
    project_world_objects(
        tmp_object_dict,
        output_dict,
        reference_lidar_pose,
        filter_range,
        V2VSetting.order,
    )
    object_np = np.zeros((V2VSetting.max_num, 7))
    mask = np.zeros(V2VSetting.max_num)
    object_ids = []

    for i, (object_id, object_content) in enumerate(output_dict.items()):
        object_bbx = object_content['coord']
        object_np[i] = object_bbx[0, :]
        mask[i] = 1
        if object_content['ass_id'] != -1:
            object_ids.append(object_content['ass_id'])
        else:
            object_ids.append(object_id + 100 * cav_id)

    return object_np, mask, object_ids


def _get_item_single_car(selected_cav_base, ego_pose):
    """
    Project the lidar and bbx to ego space first, and then do clipping.
    Parameters
    ----------
    selected_cav_base : dict
        The dictionary contains a single CAV's raw information.
    ego_pose : list
        The ego vehicle lidar pose under world coordinate.
    Returns
    -------
    selected_cav_processed : dict
        The dictionary contains the CAV's processed information.
    """
    selected_cav_processed = {}
    # calculate the transformation matrix
    transformation_matrix = selected_cav_base['params'][
        'transformation_matrix']
    object_bbx_center, object_bbx_mask, object_ids = _generate_object_center(
        [selected_cav_base], transformation_matrix)

    # filter lidar
    lidar_np = selected_cav_base['lidar_np']
    lidar_np = shuffle_points(lidar_np)
    # remove points that hit itself
    lidar_np = mask_ego_points(lidar_np)
    # project the lidar to ego space
    lidar_np[:, :3] = project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)

    selected_cav_processed.update({
        'object_bbx_center':
        object_bbx_center[object_bbx_mask == 1],
        'object_ids':
        object_ids,
        'projected_lidar':
        lidar_np,
    })

    return selected_cav_processed


def _early_fusion(info, sample_idx):
    processed_data_dict = OrderedDict()
    processed_data_dict['ego'] = {}

    ego_id = -1
    ego_lidar_pose = []

    # first find the ego vehicle's lidar pose
    for cav_id, cav_content in info.items():
        if cav_content['ego']:
            ego_id = cav_id
            ego_lidar_pose = cav_content['params']['lidar_pose']
            break

    assert ego_id != -1
    assert len(ego_lidar_pose) > 0

    projected_lidar_stack = []
    object_stack = []
    object_id_stack = []

    # loop over all CAVs to process information
    for cav_id, selected_cav_base in info.items():
        selected_cav_processed = _get_item_single_car(selected_cav_base,
                                                      ego_lidar_pose)
        # all these lidar and object coordinates are projected to ego already.
        projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
        object_stack.append(selected_cav_processed['object_bbx_center'])
        object_id_stack += selected_cav_processed['object_ids']

    # exclude all repetitive objects
    unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
    object_stack = np.vstack(object_stack)
    object_stack = object_stack[unique_indices]

    # make sure bounding boxes across all frames have the same number
    object_bbx_center = np.zeros((V2VSetting.max_num, 7))
    mask = np.zeros(V2VSetting.max_num)
    object_bbx_center[:object_stack.shape[0], :] = object_stack
    mask[:object_stack.shape[0]] = 1

    # convert list to numpy array, (N, 4)
    projected_lidar_stack = np.vstack(projected_lidar_stack)

    # We do lidar filtering in the stacked lidar
    projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                 V2VSetting.cav_lidar_range)

    # augmentation may remove some of the bbx out of range
    object_bbx_center_valid: np.ndarray = object_bbx_center[mask == 1]
    object_bbx_center_valid, valid_mask = mask_boxes_outside_range_numpy(
        object_bbx_center_valid, V2VSetting.cav_lidar_range, V2VSetting.order)
    mask[object_bbx_center_valid.shape[0]:] = 0
    object_bbx_center[:object_bbx_center_valid.
                      shape[0]] = object_bbx_center_valid
    object_bbx_center[object_bbx_center_valid.shape[0]:] = 0

    # update unique indices
    unique_indices = [
        unique_indices[i] for i, n in enumerate(list(valid_mask)) if n
    ]

    ############################################ Standard data info ############################################
    std_data_info = get_empty_standard_data_info(camera_types=[])
    std_data_info['sample_idx'] = sample_idx

    projected_lidar_stack[:,
                          1] = -projected_lidar_stack[:,
                                                      1]  # left -> right hand
    std_data_info['points'] = projected_lidar_stack

    std_data_info['timestamp'] = info[ego_id]['timestamp']
    std_data_info['lidar_points']['lidar_path'] = ''
    std_data_info['lidar_points']['num_pts_feats'] = 4

    object_bbx_center_valid[:,
                            1] = -object_bbx_center_valid[:,
                                                          1]  # left -> right hand
    instance_list = []
    for i in range(object_bbx_center_valid.shape[0]):
        empty_instance = get_empty_instance()
        bbox_3d = object_bbx_center_valid[i, :].tolist()
        empty_instance['bbox_3d'] = bbox_3d
        empty_instance['bbox_label'] = 0  # Car only
        empty_instance['bbox_label_3d'] = 0
        empty_instance = clear_instance_unused_keys(empty_instance)
        instance_list.append(empty_instance)
    std_data_info['instances'] = instance_list
    std_data_info, _ = clear_data_info_unused_keys(std_data_info)
    return std_data_info


def _fill_infos(v2v4real, len_record):
    data_list = []
    sample_idx = 0
    for idx in mmengine.track_iter_progress(list(range(
            len_record[0]))):  # TODO: For Debug Use
        # for idx in mmengine.track_iter_progress(list(range(len_record[-1]))):
        scene_idx = 0
        for i, ele in enumerate(len_record):
            if idx < ele:
                scene_idx = i
                break
        scene_dict = v2v4real[scene_idx]
        timestamp_index = (
            idx if scene_idx == 0 else idx - len_record[scene_idx - 1])
        timestamp_key = _get_timestamp_key(scene_dict, timestamp_index)
        ego_cav_content = _calc_dist_to_ego(scene_dict, timestamp_key)

        info = OrderedDict()
        for cav_id, cav_content in scene_dict.items():
            info[cav_id] = dict()
            info[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            timestamp_delay = _time_delay_calculation(cav_content['ego'])
            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = timestamp_index - timestamp_delay
            timestamp_key_delay = _get_timestamp_key(scene_dict,
                                                     timestamp_index_delay)

            # add time delay to vehicle parameters
            info[cav_id]['time_delay'] = timestamp_delay
            info[cav_id]['timestamp'] = timestamp_index

            #### Reform the data params with current timestamp object groundtruth and delay timestamp LiDAR pose. ####
            cur_params = load_yaml(cav_content[timestamp_key]['yaml'])
            delay_params = load_yaml(cav_content[timestamp_key_delay]['yaml'])
            cur_ego_params = load_yaml(ego_cav_content[timestamp_key]['yaml'])
            delay_ego_params = load_yaml(
                ego_cav_content[timestamp_key_delay]['yaml'])
            # We need to calculate the transformation matrix from cav to ego at the delayed timestamp
            delay_cav_lidar_pose = delay_params['lidar_pose']
            delay_ego_lidar_pose = delay_ego_params['lidar_pose']
            cur_ego_lidar_pose = cur_ego_params['lidar_pose']
            cur_cav_lidar_pose = cur_params['lidar_pose']

            if not cav_content['ego'] and V2VSetting.loc_err_flag:
                delay_cav_lidar_pose = _add_loc_noise(delay_cav_lidar_pose)
                cur_cav_lidar_pose = _add_loc_noise(cur_cav_lidar_pose)

            if V2VSetting.cur_ego_pose_flag:
                transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                                 cur_ego_lidar_pose)
                spatial_correction_matrix = np.eye(4)
            else:
                transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                                 delay_ego_lidar_pose)
                spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                     cur_ego_lidar_pose)

            # This is only used for late fusion, as it did the transformation in the postprocess, so we want the gt
            # object transformation use the correct one
            gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                                cur_ego_lidar_pose)

            info[cav_id]['params'] = delay_params
            delay_params['vehicles'] = cur_params['vehicles']
            delay_params['transformation_matrix'] = transformation_matrix
            delay_params['gt_transformation_matrix'] = gt_transformation_matrix
            delay_params[
                'spatial_correction_matrix'] = spatial_correction_matrix

            info[cav_id]['lidar_np'] = pcd_to_np(
                cav_content[timestamp_key_delay]['lidar'])
            info[cav_id]['lidar_path'] = cav_content[timestamp_key_delay][
                'lidar']
            # info[cav_id]['folder_name'] = cav_content[timestamp_key_delay]['lidar'].split('/')[-3]
            info[cav_id]['index'] = timestamp_index
            info[cav_id]['cav_id'] = int(cav_id)

        temp_data_info = _early_fusion(info, sample_idx)
        data_list.append(temp_data_info)
        sample_idx += 1
    return data_list


def _build_database(scenes):
    scene_database = OrderedDict()
    len_record = []

    for i, scene in enumerate(scenes):
        scene_database.update({i: OrderedDict()})
        # TODO: use CAV '0' as ego only in test, should use permutation instead in train
        for j, cav_id in enumerate(cav_list):
            scene_database[i][cav_id] = OrderedDict()
            cav_path = osp.join(scene, cav_id)
            # noinspection PyTypeChecker
            yaml_files = sorted([
                osp.join(cav_path, x) for x in os.listdir(cav_path)
                if x.endswith('.yaml') and 'additional' not in x
            ])
            timestamps = _extract_timestamps(yaml_files)

            for timestamp in timestamps:
                scene_database[i][cav_id][timestamp] = OrderedDict()
                yaml_file = os.path.join(cav_path, timestamp + '.yaml')
                point_file = os.path.join(cav_path, timestamp + '.pcd')
                scene_database[i][cav_id][timestamp]['yaml'] = yaml_file
                scene_database[i][cav_id][timestamp]['lidar'] = point_file

            if j == 0:
                scene_database[i][cav_id]['ego'] = True
                if not len_record:
                    len_record.append(len(timestamps))
                else:
                    prev_last = len_record[-1]
                    len_record.append(prev_last + len(timestamps))
            else:
                scene_database[i][cav_id]['ego'] = False

    return scene_database, len_record


def _extract_timestamps(yaml_files):
    timestamps = []

    for file in yaml_files:
        res = file.split('/')[-1]
        timestamp = res.replace('.yaml', '')
        timestamps.append(timestamp)

    return timestamps


if __name__ == '__main__':
    create_v2v4real_infos('data/v2v4real', 'v2v4real')
