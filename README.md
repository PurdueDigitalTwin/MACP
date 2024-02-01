# MACP: Efficient Model Adaptation for Cooperative Perception

[![python](https://img.shields.io/badge/-Python_3.8-306998?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3817/)
[![BSD 3-Clause License](https://img.shields.io/badge/license-MIT-750014.svg)](https://github.com/PurdueDigitalTwin/MACP/blob/master/LICENSE)

---

The official repository for the WACV 2024
paper [MACP: Efficient Model Adaptation for Cooperative Perception](https://openaccess.thecvf.com/content/WACV2024/html/Ma_MACP_Efficient_Model_Adaptation_for_Cooperative_Perception_WACV_2024_paper.html).
This work proposes a novel method to adapt a single-agent pretrained model to a V2V cooperative perception setting. It
achieves state-of-the-art performance on both the [V2V4Real](https://mobility-lab.seas.ucla.edu/v2v4real/) and
the [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) datasets.

## Setup

Our project is based on [MMDetection3D v1.1.0](https://github.com/open-mmlab/mmdetection3d/releases/tag/v1.1.0). Please
refer to the [official documentation](https://mmdetection3d.readthedocs.io/en/v1.1.0/get_started.html) to set up the
environment.

### Data Preparation

Download the [V2V4Real](https://mobility-lab.seas.ucla.edu/v2v4real/)
and [OPV2V](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu) datasets.

Once the data is downloaded, it's necessary organize the data in the following structure:

```plain
├── $REPO_ROOT
│   ├── data
│   │   ├── v2v4real
│   │   │   ├── train
│   │   │   │   ├── testoutput_CAV_data_2022-03-15-09-54-40_0 # data folder
│   │   │   ├── test
|   |   ├── openv2v
│   │   │   ├── train
│   │   │   │   ├── 2021_08_16_22_26_54 # data folder
│   │   │   ├── test
|   |   |   ├── validate
|   |   |   ├── test_culver_city
```

Then, run the script files `scripts/create_v2v4real.sh` and `scripts/create_openv2v.sh` to prepare the cached data.

### Notes

- The core code of our project is in the `projects/Coperception` folder.
- The voxelization OP in the original implementation of `BEVFusion` is different from the implementation in MMCV. Please
  refer [here](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion#compiling-operations-on-cuda) to
  compile the OP on CUDA.

## MACP Weights

If you are interested in including any other pretrained weights or details, please open an issue or
contact [us](mailto:yunsheng@purdue.edu).

|     Model     |    Backbone     |                                              Checkpoint                                               |                                                Config                                                 | AP@50 | AP@70 |                                                  Log                                                  |
|:-------------:|:---------------:|:-----------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|:-----:|:-----:|:-----------------------------------------------------------------------------------------------------:|
| MACP-V2V4Real | BEVFusion-LiDAR | [Google Drive](https://drive.google.com/file/d/1SVaMekq_hpnZ_dUb0dvD7tVYXNbTiSfj/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1K2IGPhxr2JWH20MbNlbi3hNiOdHk_LGw/view?usp=drive_link) | 67.6  | 47.9  | [Google Drive](https://drive.google.com/file/d/1SVaMekq_hpnZ_dUb0dvD7tVYXNbTiSfj/view?usp=drive_link) |
|  MACP-OPV2V   | BEVFusion-LiDAR | [Google Drive](https://drive.google.com/file/d/1fWULVO-3vGQlQ_Hmqq9dcZ5SUSwYicD1/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1_dtbrYahK1zf_-fp4IYocIuMVbx01fNc/view?usp=drive_link) | 93.7  | 90.3  | [Google Drive](https://drive.google.com/file/d/1fzHDJdsNzmsZQ59zt0_FRzoUhC0i1Ufu/view?usp=drive_link) |

## Training

We train our model on one NVIDIA RTX 4090 GPU with 24GB memory. The training command is as follows:

```bash
cd /path/to/repo
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/train.py path/to/config
```

## Evaluation

The evaluation command is as follows:

```bash
cd /path/to/repo
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/test.py path/to/config path/to/checkpoint
```

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{ma2024macp,
  title={MACP: Efficient Model Adaptation for Cooperative Perception},
  author={Ma, Yunsheng and Lu, Juanwu and Cui, Can and Zhao, Sicheng and Cao, Xu and Ye, Wenqian and Wang, Ziran},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3373--3382},
  year={2024}
}
```

## Acknowledgement

This project is based on code from several open-source projects. We would like to thank the authors for their great
work:

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [V2V4Real](https://github.com/ucla-mobility/V2V4Real)
- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
