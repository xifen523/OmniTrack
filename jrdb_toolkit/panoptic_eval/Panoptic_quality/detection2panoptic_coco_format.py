import os
import argparse
import numpy as np
import json
import time
import multiprocessing
from PIL import Image
from panopticapi.utils import IdGenerator, save_json

try:
    from pycocotools import mask as COCOmask
    from pycocotools.coco import COCO as COCO
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")
from panopticapi.utils import get_traceback, rgb2id,id2rgb

#input_json_filename='/mnt/SSD3/chenhui/google_doc_results/remaped_test/labels/jrdb_categories_cocoformat.json'
#with open(input_json_filename, 'r') as input_file:
#    json_data = json.load(input_file)

#id_to_priority = {item["id"]: item["priority"] for item in json_data}
#print("id_to_priority",id_to_priority)

def convert_int64_to_int(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_int64_to_int(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = convert_int64_to_int(item)
    elif isinstance(data, np.int64):
        data = data.item()
    return data

def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno == os.errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise

def convert_detection_to_panoptic_coco_format_single_core_deal_with_overlap(
    proc_id, coco_detection, img_ids, categories, segmentations_folder,id_to_priority
):
    id_generator = IdGenerator(categories)
    annotations_panoptic = []

    for working_idx, img_id in enumerate(img_ids):
        print("img_id", img_id)
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, working_idx, len(img_ids)))

        img = coco_detection.loadImgs(int(img_id))[0]
        pan_format = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
        overlaps_map = np.zeros((img['height'], img['width']), dtype=np.uint32)
        id_map = -1 * np.ones((img['height'], img['width']), dtype=np.uint32)

        anns_ids = coco_detection.getAnnIds(img_id)
        anns = coco_detection.loadAnns(anns_ids)
        panoptic_record = {}
        panoptic_record['image_id'] = img_id
        file_name = '{}.png'.format(img['file_name'].rsplit('/')[-1].split('.')[0])
        panoptic_record['file_name'] = file_name
        segments_info = []

        for ann in anns:
            if ann['category_id'] not in categories:
                continue

            segment_id, color = id_generator.get_id_and_color(ann['category_id'])
            mask = coco_detection.annToMask(ann)

            overlaps_map += mask

            if np.sum(pan_format) != 0:
                for row in range(id_map.shape[0]):
                    for col in range(id_map.shape[1]):
                        previous_classes = id_map[row, col]
                        if mask[row, col] == 1:
                            if previous_classes < 0:
                                pan_format[row, col] = color
                                id_map[row, col] = ann['category_id']
                            elif id_to_priority[previous_classes] < id_to_priority[ann['category_id']]:
                                pan_format[row, col] = color
                                id_map[row, col] = ann['category_id']
            else:
                id_map[mask == 1] = ann['category_id']
                pan_format[mask == 1] = color

            ann.pop('segmentation')
            ann.pop('image_id')
            ann['id'] = segment_id



            #gt_labels,gt_labels_cnt = np.unique(pan_format, return_counts=True)
            #print("gt_labels",gt_labels)
            #for label, label_cnt in zip(gt_labels, gt_labels_cnt):
            #    if label==segment_id:
            #        ann['area'] = label_cnt
            segments_info.append(ann)
        segments_info = calculated_area(segments_info,pan_format)
        panoptic_record['segments_info'] = segments_info
        annotations_panoptic.append(panoptic_record)

        Image.fromarray(pan_format).save(os.path.join(segmentations_folder, file_name))

    print('Core: {}, all {} images processed'.format(proc_id, len(img_ids)))
    return annotations_panoptic

def convert_detection_to_panoptic_coco_format_single_core(
    proc_id, coco_detection, img_ids, categories, segmentations_folder
):
    id_generator = IdGenerator(categories)

    annotations_panoptic = []
    for working_idx, img_id in enumerate(img_ids):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(img_ids)))
        img = coco_detection.loadImgs(int(img_id))[0]
        pan_format = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
        overlaps_map = np.zeros((img['height'], img['width']), dtype=np.uint32)

        anns_ids = coco_detection.getAnnIds(img_id)
        anns = coco_detection.loadAnns(anns_ids)

        panoptic_record = {}
        panoptic_record['image_id'] = img_id
        #file_name = '{}.png'.format(img['file_name'].rsplit('.')[0])
        file_name = '{}.png'.format(img['file_name'].rsplit('/')[-1].split('.')[0])
        panoptic_record['file_name'] = file_name
        segments_info = []
        for ann in anns:
            if ann['category_id'] not in categories:
                continue
            segment_id, color = id_generator.get_id_and_color(ann['category_id'])
            mask = coco_detection.annToMask(ann)
            overlaps_map += mask
            pan_format[mask == 1] = color
            ann.pop('segmentation')
            ann.pop('image_id')
            ann['id'] = segment_id
            segments_info.append(ann)

        if np.sum(overlaps_map > 1) != 0:
            raise Exception("Segments for image {} overlap each other.".format(img_id))
        panoptic_record['segments_info'] = segments_info
        annotations_panoptic.append(panoptic_record)

        Image.fromarray(pan_format).save(os.path.join(segmentations_folder, file_name))

    print('Core: {}, all {} images processed'.format(proc_id, len(img_ids)))
    return annotations_panoptic


def calculated_area(segments_info,pan_gt):
    gt_segms = {el['id']: el for el in segments_info}
    #print("gt_segms.keys()",gt_segms.keys())
    #annotations_coco_panoptic=[]
    pan_gt=rgb2id(pan_gt)
    gt_labels,gt_labels_cnt = np.unique(pan_gt, return_counts=True)
    #print("gt_labels",gt_labels)
    for gt_label, gt_label_cnt in zip(gt_labels, gt_labels_cnt):
            if gt_label != 0:
                gt_segms[gt_label]['area'] = gt_label_cnt  
                
    annotations_coco_panoptic = [v for k,v in gt_segms.items()]

    
    return annotations_coco_panoptic

def convert_detection_to_panoptic_coco_format(input_json_file, segmentations_folder, output_json_file,categories_json_file, non_overlap):
    start_time = time.time()

    if segmentations_folder is None:
        segmentations_folder = output_json_file.rsplit('.', 1)[0]

    if not os.path.isdir(segmentations_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
        os.makedirs(segmentations_folder)

    print("CONVERTING...")
    print("COCO detection format:")
    print("\tJSON file: {}".format(input_json_file))
    print("TO")
    print("COCO panoptic format")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(output_json_file))
    print('\n')
    with open(input_json_file, 'r') as f:
        input_file = json.load(f)
    coco_detection = COCO(input_json_file)
    img_ids = coco_detection.getImgIds()#[0:2]

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    
    #with open(input_json_file, 'r') as f:
    #    categories_list = json.load(f)["categories"]
    categories = {category['id']: category for category in categories_list}
    id_to_priority = {item["id"]: item["priority"] for item in categories_list}
    cpu_num = multiprocessing.cpu_count()
    img_ids_split = np.array_split(img_ids, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(img_ids_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []

    for proc_id, img_ids in enumerate(img_ids_split):
        #p = workers.apply_async(convert_detection_to_panoptic_coco_format_single_core,
        #                        (proc_id, coco_detection, img_ids, categories, segmentations_folder,id_to_priority))
        if non_overlap:
            p = workers.apply_async(convert_detection_to_panoptic_coco_format_single_core_deal_with_overlap,
                                (proc_id, coco_detection, img_ids, categories, segmentations_folder,id_to_priority))
        else:
            p = workers.apply_async(convert_detection_to_panoptic_coco_format_single_core,
                                (proc_id, coco_detection, img_ids, categories, segmentations_folder))
        processes.append(p)

    annotations_coco_panoptic = []
    for p in processes:
        annotations_coco_panoptic.extend(p.get())

    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)



    d_coco['annotations'] = annotations_coco_panoptic
    d_coco['categories'] = categories_list

    for key, value in d_coco.items():
        if isinstance(value, np.int64):
            d_coco[key] = value.item()

    d_coco = convert_int64_to_int(d_coco)
    save_json(d_coco, output_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts detection COCO format to panoptic COCO format."
    )
    parser.add_argument('--input_json_file', type=str,
                        help="JSON file with detection COCO format")
    parser.add_argument('--output_json_file', type=str,
                        help="JSON file with panoptic COCO format")
    parser.add_argument(
        '--segmentations_folder', type=str, default=None,
        help="Folder with panoptic COCO format segmentations. Default: X if output_json_file is X.json"
    )
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')

    parser.add_argument('--non_overlap', action='store_true',
                    help="Enable or disable non-overlapping feature in processing")

    args = parser.parse_args()
    convert_detection_to_panoptic_coco_format(args.input_json_file,
                                              args.segmentations_folder,
                                              args.output_json_file,
                                              args.categories_json_file,
                                              args.non_overlap)
