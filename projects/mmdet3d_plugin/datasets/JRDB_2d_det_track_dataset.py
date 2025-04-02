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
    Box2D as JRDBBox2D,
)

import iou3d_nms_cuda
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


@DATASETS.register_module()
class JRDB2DDetTrackDataset(Dataset):
    DefaultAttribute = {
        "pedestrian": "pedestrian.moving",
    }
    CLASSES = (
        "pedestrian",
    )
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
            cur_name = self.data_infos[idx]['token'].rsplit('_', 1)[0]
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
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
        )

        if self.modality["use_camera"]:
            image_paths = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])

            input_dict.update(
                dict(
                    img_filename=image_paths,
                )
            )
      
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict.update(annos)
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]

        gt_bboxes_2d = info["gt_boxes"]
        gt_names_2d = info["gt_names"]
        gt_labels_2d = []

        class_indices = {cls: index for index, cls in enumerate(self.CLASSES)}
        gt_labels_2d = [class_indices.get(cat, -1) for cat in gt_names_2d]

        gt_labels_2d = np.array(gt_labels_2d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_2d = np.concatenate([gt_bboxes_2d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_2d=gt_bboxes_2d,
            gt_labels_2d=gt_labels_2d,
            gt_names=gt_names_2d,
        )
        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int)
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
                    x1y1=box.center.tolist(),
                    size=box.wlh.tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    cls_score=box.cls_score,
                    tracking_id=str(box.token),
                )
                
                annos.append(jrdb_anno)
            jrdb_annos[sample_token] = annos
        nusc_submissions = {"results": jrdb_annos}

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_jrdb2d.json")
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
                pred_bboxes_2d = np.array([[*i['x1y1'], *i['size']] for i in result if i['detection_score'] > 0.4])
                pred_scores_2d = np.array([i['detection_score'] for i in result if i['detection_score'] > 0.4])
                # pred_bboxes_3d = np.array([[*i['center'], *i['size'], i['yaw']] for i in result])
                # pred_scores_3d = np.array([i['detection_score'] for i in result])
                gt_boxes = self.data_infos[sample_id]["gt_boxes"]

                ospa = calculate_ospa_single_frame(pred_bboxes_2d, gt_boxes, pred_scores_2d, c=1)
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
    box2d = detection["boxes_2d"]
    box2d = np.array(box2d) if isinstance(box2d, list) else box2d.numpy()

    scores = detection["scores_2d"]
    scores = np.array(scores) if isinstance(scores, list) else scores.numpy()

    labels = detection["labels_2d"] 
    labels = np.array(labels) if isinstance(labels, list) else labels.numpy()
        
    if "instance_ids" in detection:
        ids = detection['instance_ids']
        ids = np.array(ids) if isinstance(ids, list) else ids.numpy()
    if "cls_scores" in detection:
        cls_scores = detection["cls_scores"] #.numpy()
        cls_scores = np.array(cls_scores) if isinstance(cls_scores, list) else cls_scores.numpy()
    else:
        cls_scores = None

    if 'quality' in detection:
        qt = detection['quality']
        qt = np.array(qt) if isinstance(qt, list) else qt.numpy()
    else:
        qt = None



    
    box_gravity_center = box2d[..., :2].copy()
    box_dims = box2d[..., 2:].copy()


    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box2d)):
        box = JRDBBox2D(
            box_gravity_center[i],
            box_dims[i],
            label=labels[i],
            score=scores[i],
            cls_score=qt[i] if qt is not None else np.nan,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)

    if "det_boxes_2d" in detection:
        det_boxes_2d = detection["det_boxes_2d"]
        det_boxes_2d = np.array(det_boxes_2d) if isinstance(det_boxes_2d, list) else det_boxes_2d.numpy()
        det_scores_2d = detection["det_scores_2d"]
        det_scores_2d = np.array(det_scores_2d) if isinstance(det_scores_2d, list) else det_scores_2d.numpy()
        det_bbox = det_boxes_2d[..., :2].copy()
        det_dims = det_boxes_2d[..., 2:].copy()
        for i in range(len(det_boxes_2d)):
            box = JRDBBox2D(
                det_bbox[i],
                det_dims[i],
                label=0,
            )
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
    if box_3d_a.shape[1] == 4:
        iou = bbox_iou(box_3d_a, box_3d_b, score=score_a)
    else:
        iou = iou_matrix_3d_gpu(box_3d_a, box_3d_b, score_a)
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


import numpy as np

def bbox_iou(box1, box2, xywh=True, eps=1e-7, score=None):
    """
    Calculate Intersection over Union (IoU) of box1(m, 4) to box2(n, 4).

    Args:
        box1 (numpy.ndarray): A numpy array representing m bounding boxes with shape (m, 4).
        box2 (numpy.ndarray): A numpy array representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (numpy.ndarray): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    if xywh:  # transform from xywh to xyxy
        box1 = np.concatenate([box1[:, :2] - box1[:, 2:] / 2, box1[:, :2] + box1[:, 2:] / 2], axis=1)
        box2 = np.concatenate([box2[:, :2] - box2[:, 2:] / 2, box2[:, :2] + box2[:, 2:] / 2], axis=1)

    # Expand dimensions to allow broadcasting
    box1 = box1[:, np.newaxis, :]  # shape (m, 1, 4)
    box2 = box2[np.newaxis, :, :]  # shape (1, n, 4)

    # Extract the coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Calculate intersection area
    inter = np.maximum(0, np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)) * \
            np.maximum(0, np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1))

    # Calculate union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps

    # Apply score if provided
    if score is not None:
        score = score[:, np.newaxis]  # shape (m, 1)
        inter = inter * score
        v1 = w1 * h1 * score
        v2 = w2 * h2
        union = v1 + v2 - inter + eps

    # Calculate IoU
    iou = inter / union

    return iou



def ospa_iou(X, Y, score_X):
    '''
    Inputs:
    X (prediction), Y (truth) - matrices of column vector representing the
    rectangles with two corners' representation, e.g. X=(x1,y1,x2,y2), x2>x1,
    y2>y1
    score_X - categories of X (note scores of Y are implicitly ones)
    c - cut-off cost (for OSPA)
    p - norm order (for OSPA, Wasserstein)

    Outputs:
    dist_opa: OSPA distance
    '''


    # Calculate sizes of the input point patterns
    n = X.shape[1]
    m = Y.shape[1]

    # Calculate IoU matrix for pairings - fast method with vectorization
    XX = np.tile(X, [1, m])
    YY = np.reshape(np.tile(Y,[n, 1]),(Y.shape[0], n*m), order="F")
    AX = np.prod(XX[2:4,:] - XX[0:2,:], axis=0)
    AY = np.prod(YY[2:4,:] - YY[0:2,:], axis=0)
    score_XX = np.tile(score_X, [1,m])
    VX = np.multiply(AX, score_XX)
    VY = AY # as detection score = 1


    XYm = np.minimum(XX, YY)
    XYM = np.maximum(XX, YY)
    Int = np.zeros((1, XX.shape[1]))
    V_Int = np.zeros((1, XX.shape[1]))
    ind = np.all(np.less(XYM[0:2,:],XYm[2:4,:]), axis=0)
    Int[0,ind] = np.prod(XYm[2:4,ind]-XYM[0:2,ind], axis=0)
    V_Int[0,ind] = np.multiply(Int[0,ind], score_XX[0,ind])
    V_Unn = VX + VY - V_Int
    V_IoU = np.divide(V_Int, V_Unn)


    return V_IoU