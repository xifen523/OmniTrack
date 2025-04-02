import argparse
import csv
import json
import os
# multiprocessing
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

from TrackEval.trackeval.datasets.jrdb_panop import CLASSES_STUFF, CW_THING, UNKNOWN_THING, KNOWN_THING

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

CLASSES = [
    'road', 'terrain', 'sky', 'vegetation', 'wall', 'column', 'building', 'stair', 'ceiling',
    'barrier/fence', 'glass', 'big_socket/button', 'controller', 'monitor', 'cabinet', 'jacket', 'box',
    'television', 'light_pole', 'board', 'manhole', 'phone', 'waterbottle', 'picture_frames', 'standee',
    'tableware', 'decoration', 'hanging_light', 't_window', 't_floor/walking_path', 't_door', 't_machines',
    'trash_bin', 'shelf', 'car', 'bicycle/scooter', 'bag', 'chair/sofa', 'pedestrian', 'sign',
    'pole/trafficcone', 'bicyclist/rider', 'table'
]

CLASSES_OW = ['trolley', 'cargo', 'helmet', 'accessory', 'clock', 'skateboard/segway/hoverboard',
              'golfcart', 'umbrella', 'peripheral', 'airpot', 'door_handle', 'child', 'ladder',
              'emergency_pole', 'carpet', 'lift', 'document', 'poster', 'vent', 'statues', 'crutch',
              'fountain', 'store_sign', 'fire_extinguisher', 'animal', 'curtain', 'big_vehicle', 'other',
              'wall_panel']
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


def safe_mean(data_list):
    clean_list = [x for x in data_list if not np.isnan(x)]
    if not clean_list:  # if clean_list is empty
        return np.nan
    return np.mean(clean_list)


def safe_sum(data_list):
    clean_list = [x for x in data_list if not np.isnan(x)]
    if not clean_list:  # if clean_list is empty
        return np.nan
    return np.sum(clean_list)


class JRDB_PanoSeg():
    def __init__(self, pred_path, gt_path, split='val', metric='OSPA', eval_OW=False, eval_stitched=False,
                 run_parallel=False):
        self.pred_path = pred_path
        self.gt_path = gt_path
        self.split = split
        self.metric = metric
        self.eval_OW = eval_OW
        self.eval_stitched = eval_stitched
        self.metrics = ['OSPA', 'OSPA_CARD', 'OSPA_LOC']
        self.run_parallel = run_parallel
        if not self.eval_OW:
            self.class2eval = {'THING': CW_THING, 'STUFF': CLASSES_STUFF}
        else:
            # self.class2eval = {'THING': OW_THING, 'STUFF': OW_STUFF}
            # self.class2eval = {'KNOWN': OW_KNOWN, 'UNKNOWN': OW_UNKNOWN}
            self.class2eval = {'UNKNOWN_THING': UNKNOWN_THING, 'KNOWN_THING': KNOWN_THING, 'KNOWN_STUFF': CLASSES_STUFF}
        all_classes = []
        for splits in self.class2eval.values():
            all_classes.extend(splits)
        self.std_cls2id = std_cls2id
        self.std_id2cls = std_id2cls

        self.seq_list, self.seg_len = self._get_seq_info()

    def _get_seq_info(self):
        if self.split == 'train':
            locations = TRAIN
        elif self.split == 'val':
            locations = VAL
        elif self.split == 'test':
            locations = TEST
        else:
            raise ValueError('Unknown split {}'.format(self.split))
        sequence_files = []
        if self.eval_stitched:
            sequence_files.extend([loc + '.json' for loc in locations])
        else:
            for loc in locations:
                sequence_files.extend([f"{loc}_camera_{i}.json" for i in [0, 2, 4, 6, 8]])

        gt_files = [seq for seq in os.listdir(self.gt_path) if 'json' in seq]
        pred_files = [seq for seq in os.listdir(self.pred_path) if 'json' in seq]
        seg_lengths = {}

        for seq in sequence_files:
            assert seq in pred_files, 'Sequence {} not found in prediction'.format(seq)
            assert seq in gt_files, 'Sequence {} not found in ground truth'.format(seq)
            seq_path = os.path.join(self.gt_path, seq)
            with open(seq_path, 'r') as f:
                seq_data = json.load(f)
            annotated_images = [img for img in seq_data['images']]
            # TODO process the annotation before release, remove 5.2 seconds
            seg_lengths[seq] = len(annotated_images) - 5 if self.split == 'test' else len(annotated_images)

        return sequence_files, seg_lengths

    def worker_function(self, seq):
        return seq, self.eval_sequence(seq)

    def eval(self):
        res_summary = {}
        res_detail = {}
        if self.run_parallel:
            with Pool() as p:
                results = p.map(self.worker_function, self.seq_list)
            for seq, (seq_res_sumary, seq_res_detail) in results:
                res_summary[seq] = seq_res_sumary
                res_detail[seq] = seq_res_detail
        else:
            for seq in self.seq_list:
                print('Evaluating sequence {}'.format(seq))
                seq_res_sumary, seq_res_detail = self.eval_sequence(seq)
                res_summary[seq] = seq_res_sumary
                res_detail[seq] = seq_res_detail

        # average over sequences
        sums = {category: {'OSPA': [], 'OSPA_CARD': [], 'OSPA_LOC': []} for category in
                next(iter(res_summary.values())).keys()}
        num_locations = {category: 0 for category in next(iter(res_summary.values())).keys()}

        # Summing up the values and calculating averages
        averages = {}

        for category in sums:
            ospas = []

            for location in res_summary:
                sums[category]['OSPA'].append(res_summary[location][category]['OSPA'])
                sums[category]['OSPA_CARD'].append(res_summary[location][category]['OSPA_CARD'])
                sums[category]['OSPA_LOC'].append(res_summary[location][category]['OSPA_LOC'])

            averages[category] = {metric: safe_mean(sums[category][metric]) for metric in sums[category]}

        flattened_data = []
        for category, metrics in averages.items():
            flattened_data.append([category, metrics['OSPA'], metrics['OSPA_CARD'], metrics['OSPA_LOC']])

        # CSV file path
        os.makedirs(os.path.join(self.pred_path, 'ospa'), exist_ok=True)
        file_path = os.path.join(self.pred_path, 'ospa', 'average_metrics.csv')

        # Writing to CSV
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Writing the header
            writer.writerow(['Category', 'OSPA', 'OSPA_CARD', 'OSPA_LOC'])
            # Writing the data
            writer.writerows(flattened_data)

        return averages

    def eval_sequence(self, seq):
        results_detail = {}
        results_summary = {}
        gt_path = os.path.join(self.gt_path, seq)
        pred_path = os.path.join(self.pred_path, seq)
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)

        class2eval = []
        for splits in self.class2eval.values():
            class2eval.extend(splits)

        # process gt
        gt_annos = []
        gt_id2cls = {}
        for cat in gt_data['categories']:
            gt_id2cls[cat['id']] = cat['name']
        for anno in gt_data['annotations']:
            if gt_id2cls[anno['category_id']] not in class2eval:
                continue
            anno['category_id'] = self.std_cls2id[gt_id2cls[anno['category_id']]]
            gt_annos.append(anno)
        gt_data['annotations'] = gt_annos

        # get OSPA per class
        per_class_res = {}
        image_ids = [img['id'] for img in gt_data['images']]
        for c in class2eval:
            gt_c = [anno for anno in gt_data['annotations'] if
                    anno['category_id'] == self.std_cls2id[c]]
            pred_c = [anno for anno in pred_data['annotations'] if
                      anno['category_id'] == self.std_cls2id[c]]
            res_c = {}
            if len(gt_c) == 0:
                continue
            for t in image_ids:
                gt_ct = [anno['segmentation'] for anno in gt_c if anno['image_id'] == t]
                pred_ct = [anno['segmentation'] for anno in pred_c if anno['image_id'] == t]
                if len(gt_ct) == 0:
                    continue
                res_c[t] = self.OSPA(gt_ct, pred_ct)
            # average over time
            res_c = {field: safe_mean([res_c[t][field] for t in res_c.keys()]) for field in
                     res_c[list(res_c.keys())[0]].keys()}
            per_class_res[c] = res_c
            results_detail[c] = res_c

        for split_name, classes in self.class2eval.items():
            res = {}
            for field in per_class_res[list(per_class_res.keys())[0]].keys():
                res[field] = safe_mean([per_class_res[c][field] for c in classes if c in per_class_res.keys()])
            results_summary[split_name] = res
        res = {}
        for field in per_class_res[list(per_class_res.keys())[0]].keys():
            res[field] = safe_mean([per_class_res[c][field] for c in per_class_res.keys()])
        results_summary['COMBINED_CLASS_ALL'] = res

        # get OSPA per mask size
        per_size_res = {}
        for size in ['SMALL', 'MEDIUM', 'LARGE']:
            thresh_min = 0 if size == 'SMALL' else 32 ** 2 if size == 'MEDIUM' else 96 ** 2
            thresh_max = 32 ** 2 if size == 'SMALL' else 96 ** 2 if size == 'MEDIUM' else np.inf
            gt_s = [anno for anno in gt_data['annotations'] if thresh_min < anno['area'] < thresh_max]
            pred_s = [anno for anno in pred_data['annotations'] if
                      thresh_min < anno['area'] < thresh_max]
            res_s = {}
            if len(gt_s) == 0:
                continue
            for t in image_ids:
                gt_st = [anno['segmentation'] for anno in gt_s if anno['image_id'] == t]
                pred_st = [anno['segmentation'] for anno in pred_s if anno['image_id'] == t]
                if len(gt_st) == 0:
                    continue
                res_s[t] = self.OSPA(gt_st, pred_st)
            # average over time
            res_s = {field: np.nanmean([res_s[t][field] for t in res_s.keys()]) for field in
                     res_s[list(res_s.keys())[0]].keys()}

            per_size_res[size] = res_s
            results_summary[size] = res_s
        # average over sizes
        res = {}
        for field in per_size_res[list(per_size_res.keys())[0]].keys():
            res[field] = safe_mean([per_size_res[c][field] for c in per_size_res.keys()])
        results_summary['COMBINED_SIZE_ALL'] = res
        return results_summary, results_detail

    @staticmethod
    def _calculate_mask_ious(masks1, masks2, is_encoded=False, do_ioa=False):
        """ Calculates the IOU (intersection over union) between two arrays of segmentation masks.
        If is_encoded a run length encoding with pycocotools is assumed as input format, otherwise an input of numpy
        arrays of the shape (num_masks, height, width) is assumed and the encoding is performed.
        If do_ioa (intersection over area) , then calculates the intersection over the area of masks1 - this is commonly
        used to determine if detections are within crowd ignore region.
        :param masks1:  first set of masks (numpy array of shape (num_masks, height, width) if not encoded,
                        else pycocotools rle encoded format)
        :param masks2:  second set of masks (numpy array of shape (num_masks, height, width) if not encoded,
                        else pycocotools rle encoded format)
        :param is_encoded: whether the input is in pycocotools rle encoded format
        :param do_ioa: whether to perform IoA computation
        :return: the IoU/IoA scores
        """

        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils

        # use pycocotools for run length encoding of masks
        if not is_encoded:
            masks1 = mask_utils.encode(np.array(np.transpose(masks1, (1, 2, 0)), order='F'))
            masks2 = mask_utils.encode(np.array(np.transpose(masks2, (1, 2, 0)), order='F'))

        # use pycocotools for iou computation of rle encoded masks
        ious = mask_utils.iou(masks1, masks2, [do_ioa] * len(masks2))
        if len(masks1) == 0 or len(masks2) == 0:
            ious = np.asarray(ious).reshape(len(masks1), len(masks2))
        assert (ious >= 0 - np.finfo('float').eps).all()
        assert (ious <= 1 + np.finfo('float').eps).all()

        return ious

    def OSPA(self, gt_annos, pred_annos):
        res = {}
        if len(gt_annos) == 0 and len(pred_annos) == 0:
            return {'OSPA': 0, 'OSPA_CARD': 0, 'OSPA_LOC': 0}
        if len(gt_annos) == 0 or len(pred_annos) == 0:
            return {'OSPA': 1, 'OSPA_CARD': 1, 'OSPA_LOC': 0}
        m = len(gt_annos)
        n = len(pred_annos)
        similarity_scores = self._calculate_mask_ious(gt_annos, pred_annos, is_encoded=True, do_ioa=False)
        dist = 1 - similarity_scores
        match_gt, match_trk = linear_sum_assignment(dist)
        cost = np.sum(dist[match_gt, match_trk])  # match_rows, match_cols
        ospa2 = np.power(((1 * np.absolute(m - n) + cost) / max(m, n)), 1)
        term1 = np.absolute(m - n) / max(m, n)
        term2 = cost / max(m, n)
        # round to 3 decimal places
        res['OSPA'] = round(ospa2, 3)
        res['OSPA_CARD'] = round(term1, 3)
        res['OSPA_LOC'] = round(term2, 3)
        return res


if __name__ == "__main__":
    ### Example usage
    # python panoptic_eval/run_jrdb_panoseg.py
    # --TRACKERS_FOLDER /path/to/predictions_2d_panoptic_CW
    # --GT_FOLDER /path/to/labels_2d_panoptic_CW
    # --split val
    # --eval_OW False
    # --eval_stitched False
    # --run_parallel False
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRACKERS_FOLDER', type=str, help="The directory containing pose predictions for each scene")
    parser.add_argument('--GT_FOLDER', type=str, help="the pose directory for JRDB (e.g. ../labels_2d_pose_stitched)")
    parser.add_argument('--metric', choices=['OSPA'], default='OSPA')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    # eval_OW choose True or False
    parser.add_argument('--eval_OW', type=bool, default=False)
    parser.add_argument('--eval_stitched', type=bool, default=False)
    parser.add_argument('--run_parallel', type=bool, default=False)
    args = parser.parse_args()
    evaluator = JRDB_PanoSeg(pred_path=args.TRACKERS_FOLDER,
                             gt_path=args.GT_FOLDER,
                             split=args.split,
                             eval_OW=args.eval_OW,
                             eval_stitched=args.eval_stitched,
                             run_parallel=args.run_parallel
                             )
    res = evaluator.eval()
    print(res)


