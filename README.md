<!-- # OmniTrack
The official implementation of OmniTrack: Omnidirectional Multi-Object Tracking (CVPR 2025) -->

<p align="center">
<h1 align="center"><strong>OmniTrack: Omnidirectional Multi-Object Tracking</strong></h1>
<h3 align="center">CVPR 2025</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2503.04565-<COLOR>.svg)](https://arxiv.org/abs/2503.04565) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnidirectional-multi-object-tracking/multi-object-tracking-on-jrdb)](https://paperswithcode.com/sota/multi-object-tracking-on-jrdb?p=omnidirectional-multi-object-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnidirectional-multi-object-tracking/object-tracking-on-quadtrack)](https://paperswithcode.com/sota/object-tracking-on-quadtrack?p=omnidirectional-multi-object-tracking)

More details can be found in our paper [PDF](https://arxiv.org/pdf/2503.04565)

## News
- [2025/02]: 🔥 OmniTrack is accepted by [CVPR 2025](https://cvpr2025.thecvf.com/).
- [2024/11]: OmniTrack achieves the SOTA on [JRDB MOT (Multi-Object Tracking) Task](http://www.semantic-kitti.org/tasks.html#ssc) with **26.92% HOTA** (Track-by-Detection)!
</br>

## Abstract
Panoramic imagery, with its 360° field of view, offers comprehensive information to support Multi-Object Tracking (MOT) in capturing spatial and temporal relationships of surrounding objects. However, most MOT algorithms are tailored for pinhole images with limited views, impairing their effectiveness in panoramic settings. Additionally, panoramic image distortions, such as resolution loss, geometric deformation, and uneven lighting, hinder direct adaptation of existing MOT methods, leading to significant performance degradation. To address these challenges, we propose OmniTrack, an omnidirectional MOT framework that incorporates Tracklet Management to introduce temporal cues, FlexiTrack Instances for object localization and association, and the CircularStatE Module to alleviate image and geometric distortions. This integration enables tracking in large field-of-view scenarios, even under rapid sensor motion. To mitigate the lack of panoramic MOT datasets, we introduce the QuadTrack dataset—a comprehensive panoramic dataset collected by a quadruped robot, featuring diverse challenges such as wide fields of view, intense motion, and complex environments. Extensive experiments on the public JRDB dataset and the newly introduced QuadTrack benchmark demonstrate the state-of-the-art performance of the proposed framework. OmniTrack achieves a HOTA score of 26.92\% on JRDB, representing an improvement of 3.43\%, and further achieves 23.45\% on QuadTrack, surpassing the baseline by 6.81\%. 
The dataset and code will be made publicly available.

## Demo

https://github.com/user-attachments/assets/ff612706-8c87-418e-9a8f-d1f63a251ec0

## Method

| ![space-1.jpg](teaser/arch.png) | 
|:--:| 
| ***Figure 1. The proposed OmniTrack pipeline**. CSEM refers to the CircularStatE Module, DA stands for data association, E2E denotes the End-to-End tracking paradigm, TBD refers to the Track-By-Detection tracking paradigm, Upd refers to updating tracks, Init to initializing tracks, and Del to deleting tracks.* |

## Installation
Requirements are not in principle hard requirements, but there might be some differences (not tested):
- Linux
- Python 3.10
- CUDA 11.8
- torch 2.1.1

[Quick Start](docs/quick_start.md) \
[Dataset preparation](docs/dataset_preparation.md)

## Dataset

- [x] JRDB
- [x] QuadTrack

## Publication
If you find this repo useful, please consider referencing the following paper:
```
@inproceedings{luo2025omniTrack,
  title={Omnidirectional Multi-Object Tracking},
  author={Kai Luo, Hao Shi, Sheng Wu, Fei Teng, Mengfei Duan, Chang Huang, Yuhang Wang, Kaiwei Wang, Kailun Yang},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## Acknowledgement

Our code is heavily based on [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)、 [ultralytics](https://github.com/ultralytics/ultralytics) and [HybridSORT](https://github.com/ymzis69/HybridSORT) thanks for their excellent work!
