import random
import math
import os
import time
import os.path as osp
import cv2
import tempfile
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
import pyquaternion


import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from .utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
    Box as JRDBBox,
)

import iou3d_nms_cuda
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


@DATASETS.register_module()
class JRDB3DDetTrackDataset(Dataset):
    DefaultAttribute = {
        "pedestrian": "pedestrian.moving",
    }
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "pedestrian",
    )
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_frame_num=None,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        tracking=False,
        tracking_threshold=0.2,
    ):
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0

        if classes is not None:
            self.CLASSES = classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.with_velocity = with_velocity

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.tracking = tracking
        self.tracking_threshold = tracking_threshold
        self.sequences_frame_num = sequences_frame_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.current_aug = None
        self.last_id = None
        if with_seq_flag:
            self._set_sequence_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = -1
        name = ''
        for idx in range(len(self.data_infos)):
            cur_name = self.data_infos[idx]['lidar_path'][0].split('/')[-2]
            if cur_name != name:
                # the interval is too large, -> new sequence
                curr_sequence += 1
            name = cur_name
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_frame_num != None:
            if self.sequences_frame_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                sequences_num = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    self.sequences_frame_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )
                    sequences_num += (len(curr_sequence_length) - 1)
                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == sequences_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata)
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        input_dict["lidar2global"] = ego2global @ lidar2ego

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            cam_intrinsic = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = (
                    cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                )
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsic,
                )
            )

        # add JRDB specific attributes
        lower2upper = np.array(info["lower2upper"]),
        upper2ego = np.array(info["upper2ego"]),
        input_dict.update(
            dict(   
                lower2upper=lower2upper,
                upper2ego=upper2ego,
            )
        )
        
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict.update(annos)
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int)[mask]
            anns_results["instance_inds"] = instance_inds
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        jrdb_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_jrdb_box(det)
            sample_token = self.data_infos[sample_id]["token"]

            # need to convert to KITTI format
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                jrdb_anno = dict(
                    sample_token=sample_token,
                    center=box.center.tolist(),
                    size=box.wlh.tolist(),
                    yaw=box.yaw,
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    cls_score=box.cls_score,
                    tracking_name=name,
                    tracking_id=str(box.token),
                )
                
                annos.append(jrdb_anno)
            jrdb_annos[sample_token] = annos
        nusc_submissions = {"results": jrdb_annos}

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_jrdb.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return nusc_submissions


    def format_results(self, results, **kwargs):
        assert isinstance(results, list), "results must be a list"
        
        if "jsonfile_prefix" in kwargs:
            if not isinstance(kwargs["jsonfile_prefix"], str):
                jsonfile_prefix = "./results/submission"
            else:
                jsonfile_prefix = kwargs["jsonfile_prefix"]
        else:
            jsonfile_prefix = "./results/submission"

        if "img_bbox" not in results[0]:
            print("No bboxes found in results, format_results skipped")
            return None
        else:
            # result_files = dict()
            print(f"\nFormating bboxes of img_bbox")
            results_ = [out["img_bbox"] for out in results]
            # result_files.update({"img_bbox": self._format_bbox(results_)})
            KITTI_results = self._format_bbox(results_, jsonfile_prefix)

            return KITTI_results


    def evaluate(
        self,
        results,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):  
        self.eval_track = True
        self.eval_det = True

        # if "tracking" in metric:
        #     self.eval_track = True
        # if "detection" in metric:
        #     self.eval_det = True
        
        # assert self.eval_track or self.eval_det, "metric should be tracking or detection"
            
        result_dict = self.format_results(results,jsonfile_prefix=jsonfile_prefix)
        if self.eval_det and "gt_boxes" in self.data_infos[0]:
            assert len(result_dict['results']) == len(self.data_infos), "result_dict and data_infos should have the same length"
            print("Evaluating detection...")
            ospa_dict = defaultdict(list)
            for sample_id, (sample_token, result) in enumerate(mmcv.track_iter_progress(result_dict['results'].items())):
                pred_bboxes_3d = np.array([[*i['center'], *i['size'], i['yaw']] for i in result if i['detection_score'] > 0.4])
                pred_scores_3d = np.array([i['detection_score'] for i in result if i['detection_score'] > 0.4])
                # pred_bboxes_3d = np.array([[*i['center'], *i['size'], i['yaw']] for i in result])
                # pred_scores_3d = np.array([i['detection_score'] for i in result])
                gt_boxes = self.data_infos[sample_id]["gt_boxes"]

                ospa = calculate_ospa_single_frame(pred_bboxes_3d, gt_boxes, pred_scores_3d, c=1)
                seq_name, seq_frame = self.data_infos[sample_id]['token'].rsplit('_', 1)
                ospa_dict[seq_name].append(ospa)

            
            ospa_dict = {k: np.mean(v, axis=0) for k, v in ospa_dict.items()}

            # calculate overall ospa
            _sum = np.sum(list(ospa_dict.values()), axis=0)

            # calculate average ospa
            ospa_dict['overall'] = _sum / len(ospa_dict)

            # save ospa to file
            if jsonfile_prefix is None:
                jsonfile_prefix = "./results/submission"
            with open(os.path.join(jsonfile_prefix, 'ospa.txt'), 'w') as f:
                for key, value in ospa_dict.items():
                    value_str = ','.join(map(str, value))
                    f.write(f'{key},{value_str}\n')
            print(f"OSPA: saved to {os.path.join(jsonfile_prefix, 'ospa.txt')}")        
        return {"data_time": 0.0, "metric": 0.0, 'ospa': ospa_dict['overall'][0]}







def output_to_jrdb_box(detection, threshold=None):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"].cpu().numpy()  # .numpy()
    if "cls_scores" in detection:
        cls_scores = detection["cls_scores"].numpy() #.numpy()


    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        box = JRDBBox(
            box_gravity_center[i],
            box_dims[i],
            box_yaw[i],
            label=labels[i],
            score=scores[i],
            velocity=velocity,
            cls_score=cls_scores[i]
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list



def calculate_ospa_single_frame(box_3d_a, box_3d_b, score_a, c=1):
    """
    this function calculate the ospa between two 3d boxes
    Args:
        box_3d_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        box_3d_b: (N, 7) [x, y, z, dx, dy, dz, heading]
        score_a: (N)
    Returns:
        ospa: [ospa, term1, term2]
    """
    sub_a = box_3d_a.shape[0]
    sub_b = box_3d_b.shape[0]
    if sub_a == 0 and sub_b != 0:
        return [c, c, 0]
    if sub_a != 0 and sub_b == 0:
        return [c, c, 0]
    if sub_a == 0 and sub_b == 0:
        return [0, 0, 0]
    iou = iou_matrix_3d_gpu(box_3d_a, box_3d_b, score_a)
    cost_matrix = 1 - iou
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind].sum()
    term1 = 1 / max(sub_a, sub_b) * (c * abs(sub_a - sub_b))
    term2 = 1 / max(sub_a, sub_b) * cost
    ospa = 1 / max(sub_a, sub_b) * (c * abs(sub_a - sub_b) + cost)
    return [ospa, term1, term2]


def iou_matrix_3d_gpu(boxes_a, boxes_b, score):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    
    # convert to tensor
    if isinstance(boxes_a, np.ndarray):
        boxes_a = np.array(boxes_a.tolist(), dtype=float)
        boxes_a = torch.from_numpy(boxes_a).float().cuda()
    if isinstance(boxes_b, np.ndarray):
        boxes_b = np.array(boxes_b.tolist(), dtype=float)
        boxes_b = torch.from_numpy(boxes_b).float().cuda()
    if isinstance(score, np.ndarray):
        score = np.array(score.tolist(), dtype=float)
        score = torch.from_numpy(score).float().cuda()

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h * score.view(-1, 1)

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    # convert to numpy
    iou3d = iou3d.cpu().numpy()
    return iou3d