#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json
import time
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback, rgb2id,id2rgb
from tabulate import tabulate
import copy

OFFSET = 256 * 256 * 256
VOID = 0

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

CLASSES_OW = {'trolley', 'cargo', 'helmet', 'accessory', 'clock', 'skateboard/segway/hoverboard',
              'golfcart', 'umbrella', 'peripheral', 'airpot', 'door_handle', 'child', 'ladder',
              'emergency_pole', 'carpet', 'lift', 'document', 'poster', 'vent', 'statues', 'crutch',
              'fountain', 'store_sign', 'fire_extinguisher', 'animal', 'curtain', 'big_vehicle', 'other',
              'wall_panel'}

def print_Known_and_Unknown(original_pq_res):
    def compute_known(original_resuls):
        #print("known_Unknown_list",known_Unknown_list)
        original_per_classes_resuls=original_resuls['per_class']
        common_results={'pq':0.0,'rq':0.0,'sq':0.0,'n': 0}
        common_counts=0
        hidden_results={'pq':0.0,'rq':0.0,'sq':0.0,'n': 0}
        hidden_counts=0
        for class_id,class_res in original_per_classes_resuls.items():
            if std_id2cls[class_id] not in CLASSES_OW:
                #print("class_res['pq']",class_res['pq'])
                common_results['pq']+=class_res['pq']
                common_results['sq']+=class_res['sq']
                common_results['rq']+=class_res['rq']
                common_counts+=1
            else:
                
                hidden_results['pq']+=class_res['pq']
                hidden_results['sq']+=class_res['sq']
                hidden_results['rq']+=class_res['rq']
                
                hidden_counts+=1
        
        common_results['pq']=common_results['pq']/common_counts
        common_results['sq']=common_results['sq']/common_counts
        common_results['rq']=common_results['rq']/common_counts
        common_results['n']=common_counts
        if hidden_counts==0:
            return {'All':original_resuls['All'], "Known":common_results,"Unknown":hidden_results}
        hidden_results['pq']=hidden_results['pq']/hidden_counts
        hidden_results['sq']=hidden_results['sq']/hidden_counts
        hidden_results['rq']=hidden_results['rq']/hidden_counts
        hidden_results['n']=hidden_counts
    #'Things': {'pq': 0.07104665015314164, 'sq': 0.48318250048498496, 'rq': 0.08818044985076043, 'n': 60}, 'Stuff': {'pq': 0.26361055864981325, 'sq': 0.6408838213811121, 'rq': 0.3305255934838552, 'n': 11}}
        return {'All':original_resuls['All'], "Known":common_results,"Unknown":hidden_results}
        
    
    Table=compute_known(original_pq_res)
    #headers = ["", "PQ", "SQ", "RQ", "#categories"]
    #data = []
    #for name in ["All", "Known", "Unknown"]:
    #    row = [name] + [Table[name][k] * 100 for k in ["pq", "sq", "rq"]] + [Table[name]["n"]]
    #    data.append(row)
    #    #print("name is",name)
    #table = tabulate(
    #    data, headers=headers, tablefmt="pipe", floatfmt=".5f", stralign="center", numalign="center"
    #)
    return Table
    #print("table",table)
    # Save the data to a JSON file
    


class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            #print("non-zero")
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class
            #print("pq_class",pq_class)
            #print("sq_class*rq_class",sq_class*rq_class)
        #print("n is",n)
        if n==0:
            return {'pq':pq, 'sq': sq, 'rq': rq, 'n':n}, per_class_results
        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    pq_stat = PQStat()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1
        #print(cv2.imread(os.path.join(gt_folder, gt_ann['file_name'])).shape)
        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)

        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        #print("labels,labels_cnt",labels,labels_cnt)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                continue
                #raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            
            if gt_label not in gt_segms:
               # print("gt_label",gt_label)
                
                continue
            if pred_label not in pred_segms:
                #print("pred_label",pred_label)
                #return
                continue

            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false positives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            #if gt_info['iscrowd'] == 1:
            #    crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = 1 #multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    return pq_stat


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None,openworld=False):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}
    #print("categories",len(categories))
    filted_categories = {}
    for anns in gt_json['annotations']:
        for seg in anns["segments_info"]:
            if seg["category_id"] not in filted_categories:
                filted_categories[seg["category_id"]]=0
    categories = { k:v for k, v in categories.items() if k in filted_categories}
    #categories = filted_categories
   # print("filted_categories",len(filted_categories))
    #categories = dict(sorted(categories.items()))
    print("categories",len(categories))
    #return
    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']

        # TODO:
        if image_id not in pred_annotations:
            continue
            #raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
            if openworld:
                temp_result = print_Known_and_Unknown(results)
                results["Known"],results["Unknown"] = temp_result["Known"],temp_result["Unknown"]
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    saving_results = copy.deepcopy(results)
    #rint("saving_results",saving_results.keys)
    saving_results.pop("per_class")
    for k,values in saving_results.items():
        #print("k",k)
        #print("values",values)
        for metr in ['pq','sq','rq','n']:
            saving_results[k][metr] = 100 * saving_results[k][metr]

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )

    t_delta = time.time() - start_time
    #print("Time elapsed: {:0.2f} seconds".format(t_delta))
    
    return saving_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pq_saving_path', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--pred_folder', type=str, default=None,
                       help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--OW', action='store_true',
                        help="Open World Metric")
    args = parser.parse_args()
    results = pq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder,args.OW)
    #print("results",results)
    results_record_name = os.path.join(args.pq_saving_path,(args.pred_json_file).split("/")[-1].replace(".json","_pq.json"))
    print("results_record_name",results_record_name)
    with open(results_record_name, 'w') as json_file:
         json.dump(results, json_file, indent=4)
    
    