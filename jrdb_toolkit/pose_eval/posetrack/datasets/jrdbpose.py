import os
import csv
import configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException, count_valid_joints
import json
from shapely import geometry
from pathlib import Path
import cv2

UNLABELED_VS_PREDS_IOU_THRESHOLD = 0.8
UNLABELED_VS_LABELED_IOU_THRESHOLD = 0.3

SIGMAS = np.array([
    0.079, 0.025, 0.025, 0.079, 0.026, 0.079, 0.072, 0.072, 0.107,
    0.062, 0.107, 0.107, 0.062, 0.087, 0.087, 0.089, 0.089
])


def xywh2xyxy(boxes):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    # boxes [num_boxes, 4]
    r = np.zeros(boxes.shape)
    r[:, 0] = boxes[:, 0]
    r[:, 1] = boxes[:, 1]
    r[:, 2] = boxes[:, 0] + boxes[:, 2]
    r[:, 3] = boxes[:, 1] + boxes[:, 3]
    return r


def matrix_iou(X, Y):
    """Calculates IoU between each box in X and each box in Y.
    """
    if not np.any(np.any(X[2:4, :] > X[0:2, :])):
        # print(
        #     "The coordinates of the second corner of a rectangle should be bigger than the coordinates of the first corner")
        return np.zeros((Y.shape[1], X.shape[1]))

    # Calculate sizes of the input point patterns
    n = X.shape[1]
    m = Y.shape[1]

    score_X = np.ones((len(X[0]),))

    XX = np.tile(X, [1, m])
    YY = np.reshape(np.tile(Y, [n, 1]), (Y.shape[0], n * m), order="F")
    AX = np.prod(XX[2:4, :] - XX[0:2, :], axis=0)
    AY = np.prod(YY[2:4, :] - YY[0:2, :], axis=0)
    score_XX = np.tile(score_X, [1, m])
    VX = np.multiply(AX, score_XX)
    VY = AY  # as detection score = 1

    XYm = np.minimum(XX, YY)
    XYM = np.maximum(XX, YY)
    Int = np.zeros((1, XX.shape[1]))
    V_Int = np.zeros((1, XX.shape[1]))
    ind = np.all(np.less(XYM[0:2, :], XYm[2:4, :]), axis=0)
    Int[0, ind] = np.prod(XYm[2:4, ind] - XYM[0:2, ind], axis=0)
    V_Int[0, ind] = np.multiply(Int[0, ind], score_XX[0, ind])
    V_Unn = VX + VY - V_Int
    V_IoU = np.divide(V_Int, V_Unn)
    return V_IoU.reshape((-1, X.shape[1]))


class JRDBPose(_BaseDataset):
    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()

        self.n_joints = 17
        self.n_raw_joints = 17
        self.joint_names = ['head',
                            'right eye',
                            'left eye',
                            'right shoulder',
                            'neck',
                            'left shoulder',
                            'right elbow',
                            'left elbow',
                            'tailbone',
                            'right hand',
                            'right hip',
                            'left hip',
                            'left hand',
                            'right knee',
                            'left knee',
                            'right foot',
                            'left foot']
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.gt_fol = self.config['GT_FOLDER']
        self.all_labels_fol = os.path.abspath(os.path.join(self.gt_fol, "..", ))
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.output_sub_fol = ''
        self.should_classes_combine = False
        self.use_super_categories = False
        # self.labels_2d = None

        self.output_fol = None
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        tracker_fol_json_files = [file for file in sorted(os.listdir(self.tracker_fol)) if '.json' in file]
        curr_path = Path(self.tracker_fol)
        parent_path = curr_path.parent.absolute()
        self.tracker_fol = parent_path
        self.tracker_list = [os.path.basename(curr_path)]
        self.tracker_to_disp = {folder: '' for folder in self.tracker_list}
        self.class_list = ["pedestrian"]
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')
        for tracker in self.tracker_list:
            for seq in self.seq_list:

                det_file = os.path.join(self.tracker_fol, tracker, seq)

                if not os.path.isfile(det_file):
                    print(f"DET file {det_file} not found for tracker {tracker}")
                    raise TrackEvalException(f"DET file {det_file} not found for tracker {tracker}")

    def _get_seq_info(self):
        sequence_files = [file for file in sorted(os.listdir(self.gt_fol)) if '.json' in file]
        seq_lengths = dict()

        # reading sequence lengths
        for seq in sequence_files:
            seq_path = os.path.join(self.gt_fol, seq)

            with open(seq_path, 'r') as f:
                seq_data = json.load(f)

            annotated_images = [img for img in seq_data['images']]
            seq_lengths[seq] = len(annotated_images)

        return sequence_files, seq_lengths

    @staticmethod
    def get_default_dataset_config():
        default_config = {
            'GT_FOLDER': "/path/to/JRDB2022/train_dataset_with_activity/labels/labels_2d_pose_stitched_coco",
            'TRACKERS_FOLDER': "/path/to/tracker/folder",
            # Trackers location
            "PRINT_CONFIG": True,
            "ASSURE_SINGLE_TRACKER": False
        }
        return default_config

    def get_preprocessed_seq_data(self, raw_data, cls):
        self._check_unique_ids(raw_data)

        cls_id = 1  # we only have class person

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores',
                     'keypoint_visibilities', 'oks_kpts_sims',
                     'keypoint_matches', 'original_gt_ids', 'original_tracker_ids']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        num_gt_joints = np.zeros([self.n_joints], dtype=int)
        num_tracker_joints = np.zeros([self.n_joints], dtype=int)

        for t in range(raw_data['num_timesteps']):
            tracker_classes = raw_data['tracker_classes'][t]

            # Evaluation is ONLY valid for pedestrian class
            if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
                raise TrackEvalException(
                    'Evaluation is only valid for persons class. Non person class (%i) found in sequence %s at '
                    'timestep %i.' % (np.max(tracker_classes), raw_data['seq'], t))

            # for now, do not perform pre-processing and copy data!
            for k in data_keys:
                if k in raw_data:
                    data[k][t] = raw_data[k][t]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['original_gt_ids'][t] = data['gt_ids'][t].copy()
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)

                    gt_dets = data['gt_dets'][t]
                    num_gt_joints += count_valid_joints(gt_dets)

        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['original_tracker_ids'][t] = data['tracker_ids'][t].copy()
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)
                    tracker_dets = data['tracker_dets'][t]
                    num_tracker_joints += count_valid_joints(tracker_dets)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']
        data['num_gt_joints'] = num_gt_joints
        data['num_tracker_joints'] = num_tracker_joints

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def get_box_from_kpts(self, keypoints):
        keypoints = keypoints[:, :, :2]
        x0 = keypoints[:, :, 0].min(axis=1)
        y0 = keypoints[:, :, 1].min(axis=1)
        x1 = keypoints[:, :, 0].max(axis=1)
        y1 = keypoints[:, :, 1].max(axis=1)

        for i in range(len(x0)):
            if abs(x1[i] - x0[i]) > 400:
                x0[i] += WIDTH
                x0[i], x1[i] = x1[i], x0[i]
        w = x1 - x0
        h = y1 - y0
        boxes_2d = np.stack([x0, y0, w, h])
        return boxes_2d

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        global WIDTH
        if is_gt:
            file_path = os.path.join(self.gt_fol, seq)

            if 'image' not in seq:
                label_ver = 'labels_2d_stitched'
                WIDTH = 3760
            else:
                label_ver = 'labels_2d'
                WIDTH = 752


            with open(os.path.join(self.all_labels_fol, label_ver, seq), "r") as f:
                labels_2d = json.load(f)['labels']
            # with open(os.path.join(self.all_labels_fol, "labels_2d_head_stitched", seq), "r") as f:
            #     labels_heads = json.load(f)

        else:
            file_path = os.path.join(self.tracker_fol, tracker, seq)
        with open(file_path, 'r') as f:
            read_data = json.load(f)
        if 'images' in read_data.keys():
            image_data = {img['id']: {**img, 'annotations': []} for img in read_data['images']}
        else:
            image_data = {i:{'annotations': []} for i in range(1,self.seq_lengths[seq]+1)}


        for ann in read_data['annotations']:
            im_id = ann['image_id']
            # if im_id not in image_data.keys():
            #     image_data[im_id] = {}
            #     image_data[im_id]['annotations'] = []
            image_data[im_id]['annotations'].append(ann)
        image_ids = list(image_data.keys())
        if is_gt:
            # for head_ann in labels_heads['annotations']:
            #     im_id = head_ann['image_id']
            #     for ped in image_data[im_id]['annotations']:
            #         if ped['track_id'] == head_ann['track_id']:
            #             ped['bbox_head'] = head_ann['bbox']

            for im_id, v in image_data.items():
                im_name = v['file_name'][-10:]
                ped_boxes_2d = {}
                # temp
                if im_name not in labels_2d.keys():
                    continue
                for ped in labels_2d[im_name]:
                    track_id = int(ped['label_id'][ped['label_id'].find(':') + 1:])
                    ped_boxes_2d[track_id] = ped['box']
                image_data[im_id]['annotations_bbox'] = ped_boxes_2d
        else:
            for im_id, v in image_data.items():
                keypoints = []
                track_ids = []
                for ped in image_data[im_id]['annotations']:
                    keypoints.append(ped['keypoints'])
                    track_ids.append(ped['track_id'])
                keypoints = np.array(keypoints).reshape([-1, self.n_raw_joints, 3])
                ped_boxes_2d = self.get_box_from_kpts(keypoints).transpose()
                ped_boxes_2d = {track_ids[i]: ped_boxes_2d[i] for i in range(len(track_ids))}
                # for ped in image_data[im_id]['annotations']:
                #     ped['box_2d'] = ped_boxes_2d[ped['track_id']]
                image_data[im_id]['annotations_bbox'] = ped_boxes_2d

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets', 'image_ids',
                     'boxes_2d', 'box_wo_keypoints']
        # noneed
        if is_gt:
            data_keys += [
                # 'gt_crowd_ignore_regions',
                'gt_extras',
                'head_sizes',
                'body_sizes',
                'ignore_regions',
                'is_labeled']
        else:
            data_keys += ['tracker_confidences', 'keypoint_detected']

        raw_data = {key: [None] * num_timesteps for key in data_keys}

        for t in range(num_timesteps):
            im_id = image_ids[t]
            frame_data = image_data[im_id]

            raw_data['image_ids'][t] = im_id
            frame_annotations = frame_data['annotations']
            # temp
            if "annotations_bbox" not in frame_data:
                continue
            frame_box_annotations = frame_data['annotations_bbox']

            track_ids = []
            keypoints = []
            bboxes_2d = []
            scores = []
            head_sizes = []
            for p in frame_annotations:
                track_ids.append(p['track_id'])
                keypoints.append(p['keypoints'])

                keys = np.array(p['keypoints']).reshape([-1, self.n_raw_joints, 3])
                box_2d = self.get_box_from_kpts(keys).transpose()[0]
                bboxes_2d.append(box_2d)

                # if p['track_id'] not in frame_box_annotations.keys():
                #     keys = np.array(p['keypoints']).reshape([-1, self.n_raw_joints, 3])
                #     box_2d = self.get_box_from_kpts(keys).transpose()[0]
                #     bboxes_2d.append(box_2d)
                # else:
                #     keys = np.array(p['keypoints']).reshape([-1, self.n_raw_joints, 3])
                #     box_2d = self.get_box_from_kpts(keys).transpose()[0]
                #     bboxes_2d.append(box_2d)
                #     bboxes_2d.append(frame_box_annotations[p['track_id']])
                #     frame_box_annotations.pop(p['track_id'])



                if "scores" not in p:
                    p_score = [-9999 for _ in range(self.n_raw_joints)]
                    scores.append(p_score)
                else:
                    scores.append(p['scores'])

                if is_gt:
                    if "bbox_head" in p:
                        head_bb = p['bbox_head']
                    else:
                        head_bb = [0, 0, 1, 1]
                    x1, y1, w, h = head_bb
                    x2, y2 = x1 + w, y1 + h
                    head_size = self._get_head_size(x1, y1, x2, y2)
                    head_sizes.append(head_size)

            track_ids = np.array(track_ids)
            bboxes_2d = np.array(bboxes_2d)
            keypoints = np.array(keypoints).reshape([-1, self.n_raw_joints, 3])
            scores = np.array(scores).reshape([-1, self.n_raw_joints])

            if not is_gt:
                keypoints[:, :, 2] = scores

                raw_data['keypoint_detected'][t] = (keypoints[:, :, 0] > 0) & \
                                                   (keypoints[:, :, 1] > 0)

            if len(frame_annotations) > 0:
                raw_data['dets'][t] = keypoints
                raw_data['ids'][t] = track_ids
                raw_data['boxes_2d'][t] = bboxes_2d
                raw_data['classes'][t] = np.ones_like(raw_data['ids'][t])  # we only have one class
                raw_data['box_wo_keypoints'][t] = frame_box_annotations
                if not is_gt:
                    raw_data['tracker_confidences'][t] = scores
                    # raw_data['body_sizes'][t] = bboxes_2d[:, 2] * bboxes_2d[:, 3]
                elif len(keypoints) > 0:
                    raw_data['is_labeled'][t] = True
                    raw_data['head_sizes'][t] = head_sizes
                    raw_data['body_sizes'][t] = bboxes_2d[:, 2] * bboxes_2d[:, 3]
                    # handle cases where H or W equals 0
                    for idx, s in enumerate(raw_data['body_sizes'][t]):
                        if not s > 0:
                            raw_data['body_sizes'][t][idx] = 1
                else:
                    raw_data['is_labeled'][t] = False

                # else:
                # temp
                # raw_data['gt_extras'][t] = {}  # ToDO
                # raw_data['head_sizes'][t] = head_sizes
            else:
                raw_data['dets'][t] = np.empty((0, self.n_joints, 3))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                raw_data['boxes_2d'][t] = np.empty(0).astype(int)
                raw_data['box_wo_keypoints'][t] = np.empty(0).astype(int)
                # raw_data['body_sizes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
        if is_gt:
            key_map = {
                'ids': 'gt_ids',
                'classes': 'gt_classes',
                'dets': 'gt_dets',
                'boxes_2d': 'gt_boxes_2d',
                'box_wo_keypoints': 'gt_box_wo_keypoints',
                'head_sizes': 'head_sizes',
                'body_sizes': 'body_sizes',
                # 'ignore_regions': 'ignore_regions',
                'image_ids': 'image_ids',
                'is_labeled': 'is_labeled'
            }
        else:
            key_map = {
                'ids': 'tracker_ids',
                'classes': 'tracker_classes',
                'dets': 'tracker_dets',
                'boxes_2d': 'tracker_boxes_2d',
                'box_wo_keypoints': 'tracker_box_wo_keypoints',
                'image_ids': 'image_ids',
                'keypoint_detected': 'keypoint_detected'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    def get_output_fol(self, tracker):
        output_dir = os.path.join(self.output_fol, self.tracker_to_disp[tracker], self.output_sub_fol)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _get_head_size(self, x1, y1, x2, y2):
        head_size = 0.6 * np.linalg.norm(np.subtract([x2, y2], [x1, y1]))
        return head_size

    def get_raw_seq_data(self, tracker, seq):
        raw_gt_data = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker_data = self._load_raw_file(tracker, seq, is_gt=False)

        if len(raw_gt_data['image_ids']) != len(raw_tracker_data['image_ids']):
            raise TrackEvalException("The number of frames does not match ground truth")

        # 1) Remove frames that do not contain gt annotations
        raw_gt_data_new = self.remove_empty_frames(raw_gt_data, raw_gt_data['is_labeled'])
        raw_tracker_data_new = self.remove_empty_frames(raw_tracker_data, raw_gt_data['is_labeled'])

        # 1.2) update timesteps
        raw_gt_data_new['num_timesteps'] = len(raw_gt_data_new['image_ids'])
        raw_tracker_data_new['num_timesteps'] = len(raw_tracker_data_new['image_ids'])

        # remove skeletons which match to unlabeled GT skeletons
        for i, (gt_boxes_w_kpts, gt_boxes_wo_kpts, tracker_boxes) in enumerate(zip(raw_gt_data_new['gt_boxes_2d'],
                                                                                   raw_gt_data_new['gt_box_wo_keypoints'],
                                                                    raw_tracker_data_new['tracker_boxes_2d'])):
            unlabeled_gt_boxes_k = gt_boxes_wo_kpts.keys()

            # gt_boxes_w_kpts = np.vstack(
            #     [gt_boxes_w_kpts, [[0, 50, 74, 100], [0, 0, 100, 120], [0, 0, 75, 100], [0, 0, 50, 30]]])
            # gt_boxes_wo_kpts[100] = [0, 50, 74, 100]
            # gt_boxes_wo_kpts[101] = [0, 0, 100, 120]
            # gt_boxes_wo_kpts[102] = [0, 0, 75, 75]
            # gt_boxes_wo_kpts[103] = [0, 0, 50, 100]

            if len(gt_boxes_wo_kpts) > 0:
                gt_boxes_wo_kpts = xywh2xyxy(np.stack(gt_boxes_wo_kpts.values())).transpose()
            else:
                gt_boxes_wo_kpts = []

            gt_boxes_w_kpts = xywh2xyxy(np.array(gt_boxes_w_kpts)).transpose()


            if len(gt_boxes_wo_kpts) != 0:
                gt_wo_kpts_matched = [-1]
                while len(gt_wo_kpts_matched) != 0:
                    ious = matrix_iou(gt_boxes_w_kpts, gt_boxes_wo_kpts) > UNLABELED_VS_LABELED_IOU_THRESHOLD
                    gt_wo_kpts_idx, gt_w_kpts_idx = linear_sum_assignment(ious, maximize=True)
                    gt_wo_kpts_matched = gt_wo_kpts_idx[ious[gt_wo_kpts_idx, gt_w_kpts_idx] == True]
                    if len(gt_wo_kpts_matched) > 0:
                        gt_boxes_wo_kpts = np.delete(gt_boxes_wo_kpts.transpose(), gt_wo_kpts_matched, axis=0).transpose()

            if len(tracker_boxes) > 0:
                tracker_boxes = xywh2xyxy(np.array(tracker_boxes)).transpose()
            else:
                tracker_boxes = []

            # tracker_boxes = xywh2xyxy(np.array(tracker_boxes)).transpose()
            if len(tracker_boxes) != 0 and len(gt_boxes_wo_kpts)!=0:
                ious = matrix_iou(gt_boxes_wo_kpts, tracker_boxes) > UNLABELED_VS_PREDS_IOU_THRESHOLD
                pr_box_idx, gt_box_idx = linear_sum_assignment(ious, maximize=True)
                pr_box_matched = pr_box_idx[ious[pr_box_idx, gt_box_idx] == True]
            else:
                pr_box_matched = []
            # if len(pr_box_matched)>0:
            #     img_1 = np.zeros([HEIGHT,WIDTH,  3], dtype=np.uint8)
            #     for b in gt_boxes_wo_kpts.transpose():
            #         img_1 = draw_box_2d(img_1,b,[255,0,0])
            #     for b in tracker_boxes.transpose():
            #         img_1 = draw_box_2d(img_1,b,[0,0,255])
            #     cv2.imshow('image',img_1)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            for k in raw_tracker_data_new.keys():
                if k not in ['num_timesteps', 'seq', 'image_ids', 'tracker_box_wo_keypoints']:
                    raw_tracker_data_new[k][i] = np.delete(raw_tracker_data_new[k][i], pr_box_matched, axis=0)

        raw_data = {**raw_tracker_data_new, **raw_gt_data_new}  # Merges dictionaries
        similarity_scores = []
        keypoint_visibilities = []
        keypoint_matches = []
        oks_kpts_sims = []

        for t, (gt_dets_t, tracker_dets_t, head_sizes_t, body_sizes_t) in enumerate(
                zip(raw_data['gt_dets'], raw_data['tracker_dets'], raw_data['head_sizes'], raw_data['body_sizes'])):
            assert len(gt_dets_t) == len(head_sizes_t)
            pckhs, kpts_visibility, matches,oks_kpts_sim = self._calculate_p_similarities(gt_dets_t, tracker_dets_t, head_sizes_t,
                                                                       body_sizes_t)
            similarity_scores.append(pckhs)
            keypoint_visibilities.append(kpts_visibility)
            keypoint_matches.append(matches)
            oks_kpts_sims.append(oks_kpts_sim)
        raw_data['similarity_scores'] = similarity_scores
        raw_data['keypoint_visibilities'] = keypoint_visibilities
        raw_data['keypoint_matches'] = keypoint_matches
        raw_data['oks_kpts_sims'] = oks_kpts_sims
        return raw_data

    def remove_empty_frames(self, raw_data, is_labeled):
        new_data = {}
        for k, v in raw_data.items():

            if isinstance(v, list):
                new_data[k] = []
                assert len(is_labeled) == len(v)

                for t in range(len(v)):
                    if is_labeled[t]:
                        new_data[k].append(v[t])
            else:
                new_data[k] = v
        return new_data

    def _calculate_pckh(self, gt_dets_t, tracker_dets_t, head_sizes_t, body_sizes_t, dist_thres=0.2):
        assert len(gt_dets_t) == len(head_sizes_t)

        dist = np.full((len(gt_dets_t), len(tracker_dets_t), self.n_joints), np.inf)
        oks_kpts_sim = np.full((len(gt_dets_t), len(tracker_dets_t), self.n_joints), np.inf)
        oks_similarities = np.full((len(gt_dets_t), len(tracker_dets_t)), np.inf)
        kpts_v = np.zeros([len(gt_dets_t), self.n_joints])

        # joint_has_gt = (gt_dets_t[:, :, 0] > 0) & (gt_dets_t[:, :, 1] > 0)
        # joint_has_pr = (tracker_dets_t[:, :, 0] > 0) & (tracker_dets_t[:, :, 1] > 0)

        # JRDB assumption: all joints are valid
        # joint_has_gt = np.tile(True, [gt_dets_t.shape[0],gt_dets_t.shape[1]])
        # joint_has_pr = np.tile(True, [tracker_dets_t.shape[0],tracker_dets_t.shape[1]])

        for gt_i in range(len(gt_dets_t)):
            # head_size_i = head_sizes_t[gt_i]
            body_sizes_i = body_sizes_t[gt_i]

            for det_i in range(len(tracker_dets_t)):
                # for j in range(self.n_joints):
                #     if joint_has_gt[gt_i, j] and joint_has_pr[det_i, j]:
                #         gt_point = gt_dets_t[gt_i, j, :2]
                #         det_point = tracker_dets_t[det_i, j, :2]
                #
                #         dist[gt_i, det_i, j] = np.linalg.norm(np.subtract(gt_point, det_point)) / head_size_i

                # For OKS distance
                x_gt, y_gt = gt_dets_t[gt_i, :, 0], gt_dets_t[gt_i, :, 1]
                kpts_v[gt_i] = gt_dets_t[gt_i,:,2] # visibility
                x_det, y_det = tracker_dets_t[det_i, :, 0], tracker_dets_t[det_i, :, 1]
                v = gt_dets_t[gt_i, :, 2] + 1
                vars = (SIGMAS * 2) ** 2
                dx = np.subtract(x_gt, x_det)
                dy = np.subtract(y_gt, y_det)
                e = (dx ** 2 + dy ** 2) / (vars * body_sizes_i * 2)
                oks_kpts_sim[gt_i, det_i, :] = np.exp(-e)
                oks = np.sum(np.exp(-e)) / e.shape[0] if len(e) != 0 else 0
                oks_similarities[gt_i, det_i] = oks

        # number of annotated joints
        # nGTp = np.sum(joint_has_gt, axis=1)
        match = oks_kpts_sim <= dist_thres
        # pck = 1.0 * np.sum(match, axis=2)
        # for i in range(joint_has_gt.shape[0]):
        #     for j in range(joint_has_pr.shape[0]):
        #         if nGTp[i] > 0:
        #             pck[i, j] = pck[i, j] / nGTp[i]

        return oks_similarities, kpts_v, match, oks_kpts_sim
        # return oks_similarities, dist, match, oks_kpts_sim


    def _calculate_p_similarities(self, gt_dets_t, tracker_dets_t, head_sizes_t, body_sizes_t):
        similarity_scores, kpts_v, matches,oks_kpts_sim = self._calculate_pckh(gt_dets_t, tracker_dets_t, head_sizes_t,
                                                                     body_sizes_t)
        return similarity_scores, kpts_v, matches,oks_kpts_sim

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        raise NotImplementedError("Not implemented")



def draw_box_2d(image, box, color=(0, 0, 255), thickness=2, text=None, occluded=0
                ):
    style = 'dashed' if occluded == 1 else None
    x0, y0, x1, y1 = list(map(int, box))
    cv2.line(image, (x0, y0), (x0, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x1, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x0, y0), (x1, y0), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x0, y1), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
    return image