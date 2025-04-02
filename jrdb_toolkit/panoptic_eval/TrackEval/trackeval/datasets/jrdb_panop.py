import json
import os
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from ._base_dataset import _BaseDataset
from .. import _timing
from .. import utils
from ..utils import TrackEvalException

TRAIN = [
    'bytes-cafe-2019-02-07_0',
    'clark-center-2019-02-28_0',
    'clark-center-intersection-2019-02-28_0',
    'cubberly-auditorium-2019-04-22_0',
    'forbes-cafe-2019-01-22_0',
    'gates-159-group-meeting-2019-04-03_0',
    'gates-basement-elevators-2019-01-17_1',
    'gates-to-clark-2019-02-28_1',
    'hewlett-packard-intersection-2019-01-24_0',
    'huang-basement-2019-01-25_0',
    'huang-lane-2019-02-12_0',
    'jordan-hall-2019-04-22_0',
    'memorial-court-2019-03-16_0',
    'packard-poster-session-2019-03-20_0',
    'packard-poster-session-2019-03-20_1',
    'packard-poster-session-2019-03-20_2',
    'stlc-111-2019-04-19_0',
    'svl-meeting-gates-2-2019-04-08_0',
    'svl-meeting-gates-2-2019-04-08_1',
    'tressider-2019-03-16_0',
]
VAL = [
    'clark-center-2019-02-28_1',
    'gates-ai-lab-2019-02-08_0',
    'huang-2-2019-01-25_0',
    'meyer-green-2019-03-16_0',
    'nvidia-aud-2019-04-18_0',
    'tressider-2019-03-16_1',
    'tressider-2019-04-26_2',
]
TEST = [
    'cubberly-auditorium-2019-04-22_1',
    'discovery-walk-2019-02-28_0',
    'discovery-walk-2019-02-28_1',
    'food-trucks-2019-02-12_0',
    'gates-ai-lab-2019-04-17_0',
    'gates-basement-elevators-2019-01-17_0',
    'gates-foyer-2019-01-17_0',
    'gates-to-clark-2019-02-28_0',
    'hewlett-class-2019-01-23_0',
    'hewlett-class-2019-01-23_1',
    'huang-2-2019-01-25_1',
    'huang-intersection-2019-01-22_0',
    'indoor-coupa-cafe-2019-02-06_0',
    'lomita-serra-intersection-2019-01-30_0',
    'meyer-green-2019-03-16_1',
    'nvidia-aud-2019-01-25_0',
    'nvidia-aud-2019-04-18_1',
    'nvidia-aud-2019-04-18_2',
    'outdoor-coupa-cafe-2019-02-06_0',
    'quarry-road-2019-02-28_0',
    'serra-street-2019-01-30_0',
    'stlc-111-2019-04-19_1',
    'stlc-111-2019-04-19_2',
    'tressider-2019-03-16_2',
    'tressider-2019-04-26_0',
    'tressider-2019-04-26_1',
    'tressider-2019-04-26_3'
]

OW_KNOWN = [
    'road', 'terrain', 'sky', 'vegetation', 'wall', 'column', 'building', 'stair', 'ceiling',
    'barrier/fence', 'glass', 'big_socket/button', 'controller', 'monitor', 'cabinet', 'jacket', 'box',
    'television', 'light_pole', 'board', 'manhole', 'phone', 'waterbottle', 'picture_frames', 'standee',
    'tableware', 'decoration', 'hanging_light', 'window', 'floor/walking_path', 'door', 'machines',
    'trash_bin', 'shelf', 'car', 'bicycle/scooter', 'bag', 'chair/sofa', 'pedestrian', 'sign',
    'pole/trafficcone', 'bicyclist/rider', 'table'
]
OW_UNKNOWN = ['curtain', 'trolley', 'helmet', 'wall_panel', 'fire_extinguisher', 'lift', 'animal',
              'accessory', 'other', 'umbrella', 'fountain', 'door_handle', 'carpet', 'vent',
              'clock', 'crutch', 'peripheral', 'skateboard/segway/hoverboard', 'poster', 'document',
              'store_sign', 'ladder', 'statues', 'cargo', 'child', 'emergency_pole', 'golfcart',
              'big_vehicle', 'airpot']

CLASSES_OW = [
    'road', 'terrain', 'sky', 'vegetation', 'wall',
    'column', 'building', 'stair', 'ceiling', 'barrier/fence',
    'crutch', 'fire_extinguisher', 'big_socket/button', 'airpot',
    'controller', 'umbrella', 'animal',
    'monitor', 'carpet', 'cabinet', 'jacket', 'emergency_pole',
    'helmet', 'peripheral', 'curtain', 'box', 'document',
    'television', 'light_pole', 'board', 'manhole', 'phone', 'ladder',
    'waterbottle', 'picture_frames', 'standee', 'store_sign', 'statues',
    'cargo', 'tableware', 'accessory', 'other', 'clock', 'golfcart', 'wall_panel',
    'decoration', 'hanging_light', 'window', 'poster', 'lift', 'vent',
    'fountain', 'door_handle', 'floor/walking_path', 'door', 'machines',
    'child', 'trash_bin', 'shelf', 'car', 'bicycle/scooter',
    'bag', 'big_vehicle', 'chair/sofa', 'pedestrian', 'sign',
    'trolley', 'pole/trafficcone',
    'bicyclist/rider', 'table', 'skateboard/segway/hoverboard', 'glass']

UNKNOWN_THING = ['crutch', 'fire_extinguisher', 'airpot', 'umbrella', 'animal', 'carpet', 'emergency_pole', 'helmet',
                 'peripheral', 'curtain', 'document', 'ladder', 'store_sign', 'statues', 'cargo', 'accessory', 'other',
                 'golfcart', 'wall_panel', 'poster', 'lift', 'vent', 'fountain', 'door_handle', 'child', 'big_vehicle',
                 'trolley', 'skateboard/segway/hoverboard']
KNOWN_THING = ['big_socket/button', 'controller', 'monitor', 'cabinet', 'jacket', 'box', 'television', 'light_pole',
               'board', 'manhole', 'phone', 'waterbottle', 'picture_frames', 'standee', 'tableware', 'decoration',
               'hanging_light', 'window', 'floor/walking_path', 'door', 'machines', 'trash_bin', 'shelf', 'car',
               'bicycle/scooter', 'bag', 'chair/sofa', 'pedestrian', 'sign', 'pole/trafficcone', 'bicyclist/rider',
               'table']
CLASSES_STUFF = ['road', 'terrain', 'sky', 'vegetation', 'wall', 'column', 'building', 'stair',
                 'ceiling', 'barrier/fence', 'glass']

OW_THING = [
    'crutch', 'fire_extinguisher', 'big_socket/button', 'airpot', 'controller', 'umbrella', 'animal',
    'monitor', 'carpet', 'cabinet', 'jacket', 'emergency_pole', 'helmet', 'peripheral', 'curtain',
    'box', 'document', 'television', 'light_pole', 'board', 'manhole', 'phone', 'ladder',
    'waterbottle', 'picture_frames', 'standee', 'store_sign', 'statues', 'cargo', 'tableware',
    'accessory', 'other', 'golfcart', 'wall_panel', 'decoration', 'hanging_light', 'window', 'poster',
    'lift', 'vent', 'fountain', 'door_handle', 'floor/walking_path', 'door', 'machines', 'child',
    'trash_bin', 'shelf', 'car', 'bicycle/scooter', 'bag', 'big_vehicle', 'chair/sofa', 'pedestrian',
    'sign', 'trolley', 'pole/trafficcone', 'bicyclist/rider', 'table', 'skateboard/segway/hoverboard']

CW_THING = ['big_socket/button', 'controller', 'monitor', 'cabinet', 'jacket', 'box',
                    'television', 'light_pole', 'board', 'manhole', 'phone', 'waterbottle',
                    'picture_frames', 'standee', 'tableware', 'decoration', 'hanging_light', 'window',
                    'floor/walking_path', 'door', 'machines', 'trash_bin', 'shelf', 'car', 'bicycle/scooter', 'bag',
                    'chair/sofa', 'pedestrian', 'sign', 'pole/trafficcone', 'bicyclist/rider', 'table']
std_id2cls = {0: 'road', 1: 'terrain', 2: 'sky', 3: 'vegetation', 4: 'wall', 5: 'column', 6: 'building', 7: 'stair',
              8: 'ceiling', 9: 'barrier/fence', 10: 'crutch', 11: 'fire_extinguisher', 12: 'big_socket/button',
              13: 'airpot', 14: 'controller', 15: 'umbrella', 16: 'animal', 17: 'monitor', 18: 'carpet', 19: 'cabinet',
              20: 'jacket', 21: 'emergency_pole', 22: 'helmet', 23: 'peripheral', 24: 'curtain', 25: 'box',
              26: 'document', 27: 'television', 28: 'light_pole', 29: 'board', 30: 'manhole', 31: 'phone', 32: 'ladder',
              33: 'waterbottle', 34: 'picture_frames', 35: 'standee', 36: 'store_sign', 37: 'statues', 38: 'cargo',
              39: 'tableware', 40: 'accessory', 41: 'other', 42: 'clock', 43: 'golfcart', 44: 'wall_panel',
              45: 'decoration', 46: 'hanging_light', 47: 'window', 48: 'poster', 49: 'lift', 50: 'vent', 51: 'fountain',
              52: 'door_handle', 53: 'floor/walking_path', 54: 'machines', 55: 'child', 56: 'trash_bin', 57: 'shelf',
              58: 'car', 59: 'bicycle/scooter', 60: 'bag', 61: 'big_vehicle', 62: 'chair/sofa', 63: 'pedestrian',
              64: 'sign', 65: 'trolley', 66: 'pole/trafficcone', 67: 'bicyclist/rider', 68: 'table',
              69: 'skateboard/segway/hoverboard', 70: 'glass', 71: 'door'}
std_cls2id = {v: k for k, v in std_id2cls.items()}


class JRDB_Panoptic(_BaseDataset):
    def __init__(self, config=None):
        super().__init__()
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.gt_fol = config['GT_FOLDER']
        self.tracker_fol = config['TRACKERS_FOLDER']
        self.output_sub_fol = ''
        self.split = self.config['SPLIT_TO_EVAL']
        self.eval_stitched = self.config['EVAL_STITCHED']
        self.eval_OW = self.config['EVAL_OW']
        self.should_classes_combine = False

        self.output_fol = None
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        curr_path = Path(self.tracker_fol)
        parent_path = curr_path.parent.absolute()
        self.tracker_fol = parent_path
        self.tracker_list = [os.path.basename(curr_path)]
        self.tracker_to_disp = {folder: '' for folder in self.tracker_list}

        if not self.eval_OW:
            self.class2eval = {'THING': CW_THING, 'STUFF': CLASSES_STUFF}
        else:
            # self.class2eval = {'THING': OW_THING, 'STUFF':   CLASSES_STUFF}
            # self.class2eval = {'KNOWN': OW_KNOWN, 'UNKNOWN': OW_UNKNOWN}
            self.class2eval = {'UNKNOWN_THING': UNKNOWN_THING, 'KNOWN_THING': KNOWN_THING, 'KNOWN_STUFF': CLASSES_STUFF}
        self.class_list = []
        for splits in self.class2eval.values():
            self.class_list.extend(splits)

        self.id2class = std_id2cls
        self.class2id = std_cls2id
        self.seq_list, self.seq_lengths = self._get_seq_info()

        for tracker in self.tracker_list:
            for seq in self.seq_list:
                det_file = os.path.join(self.tracker_fol, tracker, seq)
                if not os.path.isfile(det_file):
                    print(f"DET file {det_file} not found for tracker {tracker}")
                    raise TrackEvalException(f"DET file {det_file} not found for tracker {tracker}")

    @staticmethod
    def get_default_dataset_config():
        code_path = utils.get_code_path()
        default_config = {
            # 'GT_FOLDER': os.path.join(code_path, 'data/gt/tao/tao_training'),  # Location of GT data
            # 'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/tao/tao_training'),  # Trackers location
            # 'GT_FOLDER': "/mnt/SSD3TB/dataset/Panotrack-0411/04-11/stitched/test_dataset_without_labels/labels_2d_panoptic_stitched_OW",
            'GT_FOLDER': "/path/to/labels_2d_panoptic_stitched_CW",
            # Location of GT data
            # 'TRACKERS_FOLDER': "/home/tho/Downloads/OW_tracking/open_world/bytetrack/trained_fc_clip",
            # 'TRACKERS_FOLDER': "/mnt/SSD3TB/dataset/Panotrack-0411/04-11/stitched/test_dataset_without_labels/labels_2d_panoptic_stitched_CW",
            'TRACKERS_FOLDER': "/path/to/json_predictions",
            # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': None,  # Classes to eval (if None, all classes)
            'SPLIT_TO_EVAL': 'val',  # Valid: 'train', 'val', 'test' (only for benchmark server)
            'EVAL_OW': False,  # Whether to evaluate open-world (if False, closed-world)
            'EVAL_STITCHED': False,  # Whether to evaluate on stitched or individual images
            'PRINT_CONFIG': True,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'MAX_DETECTIONS': 300,  # Number of maximal allowed detections per image (0 for unlimited)
            'SUBSET': 'all'
        }
        return default_config

    def _get_seq_info(self):
        if self.split == 'train':
            locations = TRAIN
        elif self.split == 'val':
            locations = VAL
        elif self.split == 'test':
            locations = TEST
        elif self.split == 'debug':
            locations = ['gates-to-clark-2019-02-28_1']
        else:
            raise TrackEvalException('Unknown split {}'.format(self.split))
        sequence_files = []
        if self.eval_stitched:
            sequence_files.extend([loc + '.json' for loc in locations])
        else:
            for loc in locations:
                sequence_files.extend([f"{loc}_camera_{i}.json" for i in [0, 2, 4, 6, 8]])
        seq_lengths = dict()

        # reading sequence lengths
        for seq in sequence_files:
            seq_path = os.path.join(self.gt_fol, seq)

            with open(seq_path, 'r') as f:
                seq_data = json.load(f)

            annotated_images = [img for img in seq_data['images']]
            # # TODO: actual remove the last 5.2 seconds
            seq_lengths[seq] = len(annotated_images)

        return sequence_files, seq_lengths

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the JRDB panoptic tracking format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        if is_gt:
            file_path = os.path.join(self.gt_fol, seq)
        else:
            file_path = os.path.join(self.tracker_fol, tracker, seq)

        with open(file_path, 'r') as f:
            read_data = json.load(f)

        if 'images' in read_data.keys():
            # TODO actual remove 5.2 seconds
            image_data = {img['id']: {**img, 'annotations': []} for img in read_data['images'] if
                          img['id'] in range(1, self.seq_lengths[seq] + 1)}
        else:
            image_data = {i: {'annotations': []} for i in range(1, self.seq_lengths[seq] + 1)}

        file_id2catname = {cat['id']: cat['name'] for cat in read_data['categories']}
        self.id_generator = UniqueIDGenerator()

        for ann in read_data['annotations']:
            if file_id2catname[ann['category_id']] not in self.class_list:
                continue
            if ann['image_id'] not in image_data.keys():
                continue
            image_data[ann['image_id']]['annotations'].append(ann)
        image_ids = sorted(list(image_data.keys()))

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets', 'image_ids']
        if not is_gt:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        for t, (img_id, frame_data) in enumerate(image_data.items()):
            unique_ids = set()
            unique_mask = []
            raw_data['image_ids'][img_id - 1] = img_id
            frame_annos = frame_data['annotations']
            tracking_ids = []
            for ann in frame_annos:
                standard_catid = self.class2id[file_id2catname[ann['category_id']]]
                if 'tracking_id' in ann.keys():
                    track_id = self.id_generator.generate_unique_id(standard_catid, ann['tracking_id'])
                elif 'floor_id' in ann['attributes'].keys():
                    # assert int(ann['attributes']['floor_id']) >= 0, "Floor_id must be non-negative."
                    # TODO: process floor_id as tracking_id and must be > 0
                    track_id = self.id_generator.generate_unique_id(standard_catid,
                                                                    int(ann['attributes']['floor_id']) + 1)
                elif 'tracking_id' in ann['attributes'].keys():
                    track_id = self.id_generator.generate_unique_id(standard_catid, ann['attributes']['tracking_id'])
                else:
                    assert self.id2class[standard_catid] in CLASSES_STUFF, \
                        (f"Unknown tracking id for class {self.id2class[standard_catid]},"
                         f"Note: Each stuff class only has ONLY ONE predicted instance per frame.")
                    track_id = self.id_generator.generate_unique_id(standard_catid, 1)  # auto assign with 1
                if track_id not in unique_ids:
                    unique_mask.append(True)
                    unique_ids.add(track_id)
                else:
                    unique_mask.append(False)
                tracking_ids.append(track_id)

                # if tracking_ids[-1] in unique_ids:
                #     print(f"Duplicate tracking id {tracking_ids[-1]} found in sequence {seq} at image_id {t + 1}.")
                # unique_ids.add(tracking_ids[-1])
            raw_data['ids'][t] = np.array(tracking_ids)
            raw_data['classes'][t] = np.array(
                [self.class2id[file_id2catname[ann['category_id']]] for ann in frame_annos])
            raw_data['dets'][t] = np.array([ann['segmentation'] for ann in frame_annos])
            if not is_gt:
                raw_data['tracker_confidences'][t] = np.array(
                    [ann['score'] if 'score' in ann else -9999 for ann in frame_annos])
                raw_data['tracker_confidences'][t] = raw_data['tracker_confidences'][t][unique_mask]

            # filter out duplicate detections
            raw_data['ids'][t] = np.array(raw_data['ids'][t][unique_mask])
            raw_data['classes'][t] = np.array(raw_data['classes'][t][unique_mask])
            raw_data['dets'][t] = np.array(raw_data['dets'][t][unique_mask])

        # TODO: remove this
        for t, d in enumerate(raw_data['dets']):
            if len(d) > 0:
                prev_det = raw_data['dets'][t]
                prev_id = raw_data['ids'][t]
                prev_cls = raw_data['classes'][t]
                break

        for t, d in enumerate(raw_data['dets']):
            # TODO: remove this
            # ensure no empty detections by copying from previous timestep
            if len(d) != 0:
                prev_det = d
                prev_id = raw_data['ids'][t]
                prev_cls = raw_data['classes'][t]
            if len(d) == 0:
                raw_data['dets'][t] = prev_det
                raw_data['ids'][t] = prev_id
                raw_data['classes'][t] = prev_cls
            # assert d is not None and len(d) > 0, (f"No detections found for "
            #                                       f"tracker {tracker} on sequence {seq} at "
            #                                       f"image_id {t + 1}.")

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data["num_timesteps"] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    def _get_mask_size_classes(self, masks):
        # classify masks into sizes (0: small: area < 32^2, 1: medium: 32^2 <= area < 96^2, 2: large: area >= 96^2)
        small_mask = np.array([np.sum(mask_utils.decode(mask)) < 32 ** 2 for mask in masks])
        medium_mask = np.array([32 ** 2 <= np.sum(mask_utils.decode(mask)) < 96 ** 2 for mask in masks])
        large_mask = np.array([np.sum(mask_utils.decode(mask)) >= 96 ** 2 for mask in masks])
        # classify masks into 0,1,2
        mask_sizes = np.zeros(len(masks))
        mask_sizes[small_mask] = 0
        mask_sizes[medium_mask] = 1
        mask_sizes[large_mask] = 2
        return mask_sizes

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t, gt_classes_t, tracker_classes_t):
        """ Calculates similarities between all gt and tracker detections at a single timestep."""
        mask_similarity_scores = self._calculate_mask_ious(gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False)
        label_similarity_scores = self._calculate_label_similarities(gt_classes_t, tracker_classes_t)
        similarity_scores = mask_similarity_scores * label_similarity_scores
        gt_mask_sizes = self._get_mask_size_classes(gt_dets_t)
        tracker_mask_sizes = self._get_mask_size_classes(tracker_dets_t)
        return similarity_scores, gt_mask_sizes, tracker_mask_sizes, mask_similarity_scores, label_similarity_scores

    def get_preprocessed_seq_data(self, raw_data, cls):
        self._check_unique_ids(raw_data)
        cls_id = self.class2id[cls]
        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores',
                     # 'gt_mask_sizes','tracker_mask_sizes' ,'mask_sims', 'cls_sims','ious_per_class'
                     'original_gt_ids', 'original_tracker_ids', ]
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []

        per_class_unique_gt_ids = []
        per_class_unique_tracker_ids = []

        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):
            for k in data_keys:
                if k in raw_data.keys():
                    data[k][t] = raw_data[k][t]

            gt_class_mask = np.atleast_1d(raw_data['gt_classes'][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets
            data['tracker_ids'][t] = tracker_ids
            data['tracker_dets'][t] = tracker_dets
            data['similarity_scores'][t] = similarity_scores
            if len(tracker_ids) > 0:
                if np.max(np.unique(tracker_ids, return_counts=True)[1]) > 1:
                    print("Tracker id not unique")

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['original_tracker_ids'][t] = data['tracker_ids'][t].copy()
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int64)
                    # tracker_dets = data['tracker_dets'][t]
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['original_gt_ids'][t] = data['gt_ids'][t].copy()
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.uint8)
                    # gt_dets = data['gt_dets'][t]

        # Record overview statistics.
        data['cls'] = cls
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']
        data['id2class'] = self.id2class

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    @_timing.time
    def get_raw_seq_data(self, tracker, seq):
        """ Loads raw data (tracker and ground-truth) for a single tracker on a single sequence.
        Raw data includes all of the information needed for both preprocessing and evaluation, for all classes.
        A later function (get_processed_seq_data) will perform such preprocessing and extract relevant information for
        the evaluation of each class.

        This returns a dict which contains the fields:
        [num_timesteps]: integer
        [gt_ids, tracker_ids, gt_classes, tracker_classes, tracker_confidences]:
                                                                list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, tracker_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [similarity_scores]: list (for each timestep) of 2D NDArrays.
        [gt_extras]: dict (for each extra) of lists (for each timestep) of 1D NDArrays (for each det).

        gt_extras contains dataset specific information used for preprocessing such as occlusion and truncation levels.

        Note that similarities are extracted as part of the dataset and not the metric, because almost all metrics are
        independent of the exact method of calculating the similarity. However datasets are not (e.g. segmentation
        masks vs 2D boxes vs 3D boxes).
        We calculate the similarity before preprocessing because often both preprocessing and evaluation require it and
        we don't wish to calculate this twice.
        We calculate similarity between all gt and tracker classes (not just each class individually) to allow for
        calculation of metrics such as class confusion matrices. Typically the impact of this on performance is low.
        """
        # Load raw data.
        raw_gt_data = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker_data = self._load_raw_file(tracker, seq, is_gt=False)
        raw_data = {**raw_tracker_data, **raw_gt_data}  # Merges dictionaries

        # Calculate similarities for each timestep.
        similarity_scores = []
        gt_mask_sizes = []
        tracker_mask_sizes = []
        mask_sims = []
        cls_sims = []
        ious_per_class = []

        for t, (gt_dets_t, tracker_dets_t, gt_classes_t, tracker_classes_t) in enumerate(
                zip(raw_data['gt_dets'], raw_data['tracker_dets'],
                    raw_data['gt_classes'], raw_data['tracker_classes'])):
            # gt_dets_t, tracker_dets_t, = np.array(gt_dets_t), np.array(tracker_dets_t)
            # iou_per_class = {}
            # for c in np.unique(gt_classes_t):
            #     gt_dets_tc = gt_dets_t[gt_classes_t == c]
            #     gt_classes_tc = gt_classes_t[gt_classes_t == c]
            #     tracker_dets_tc = tracker_dets_t[tracker_classes_t == c]
            #     tracker_classes_tc = tracker_classes_t[tracker_classes_t == c]
            #     if len(tracker_dets_tc) == 0:
            #         iou_per_class[c] = 0
            #     else:
            #         iou_per_class[c] = self._calculate_mask_ious(list(gt_dets_tc), list(tracker_dets_tc), is_encoded=True, do_ioa=False)
            #
            # ious_per_class.append(iou_per_class)
            # ious, gt_mask_size, tracker_mask_size, mask_sim, cls_sim = self._calculate_similarities(list(gt_dets_t), list(tracker_dets_t), gt_classes_t, tracker_classes_t)
            if tracker_dets_t is None:
                print('cc')
            if len(gt_dets_t) == 0 or len(tracker_dets_t) == 0:
                ious = np.zeros((len(gt_dets_t), len(tracker_dets_t)))
            else:
                ious = self._calculate_mask_ious(list(gt_dets_t), list(tracker_dets_t), is_encoded=True, do_ioa=False)
            similarity_scores.append(ious)
            # gt_mask_sizes.append(gt_mask_size)
            # tracker_mask_sizes.append(tracker_mask_size)
            # mask_sims.append(mask_sim)
            # cls_sims.append(cls_sim)

        raw_data['similarity_scores'] = similarity_scores
        # raw_data['gt_mask_sizes'] = gt_mask_sizes
        # raw_data['tracker_mask_sizes'] = tracker_mask_sizes
        # raw_data['mask_sims'] = mask_sims
        # raw_data['cls_sims'] = cls_sims
        # raw_data['ious_per_class'] = ious_per_class
        return raw_data

    def _calculate_label_similarities(self, gt_classes_t, tracker_classes_t):
        gt_classes_t = np.array(gt_classes_t)[:, np.newaxis]
        tracker_classes_t = np.array(tracker_classes_t)[np.newaxis, :]
        label_similarity_scores = (gt_classes_t == tracker_classes_t).astype(np.uint8)
        return label_similarity_scores

    # def generate_unique_id(self, class_id, tracking_id):
    #     return (class_id << 11) + int(tracking_id)
    #
    # def retrieve_ids(self, unique_id):
    #     class_id = unique_id >> 11
    #     tracking_id = unique_id & 0x7FF
    #     return class_id, tracking_id


class UniqueIDGenerator:

    def __init__(self):
        # Define the number of bits required for class_id and tracking_id
        self.CLASS_BITS = 7  # 2^7 = 128, which can represent up to 128 classes.
        self.TRACKING_BITS = 18  # 2^18 = 262144, which can represent up to 262144 instances per class.

        # Define masks for bitwise operations
        self.CLASS_MASK = (1 << self.CLASS_BITS) - 1
        self.TRACKING_MASK = (1 << self.TRACKING_BITS) - 1
        # self.min_track_id = min_track_id

    def generate_unique_id(self, class_id, tracking_id):
        """Generate a unique ID using class_id and tracking_id."""
        class_id, tracking_id = int(class_id), int(tracking_id)
        if class_id > self.CLASS_MASK:
            raise ValueError(f"class_id should be less than {self.CLASS_MASK + 1}")
        if tracking_id > self.TRACKING_MASK:
            raise ValueError(f"tracking_id should be less than {self.TRACKING_MASK + 1}")

        # Use bitwise shift and OR operation to combine class_id and tracking_id
        return (class_id << self.TRACKING_BITS) | tracking_id

    def retrieve_ids(self, unique_id):
        """Retrieve class_id and tracking_id from a unique ID."""
        class_id = (unique_id >> self.TRACKING_BITS) & self.CLASS_MASK
        tracking_id = unique_id & self.TRACKING_MASK
        return class_id, tracking_id
