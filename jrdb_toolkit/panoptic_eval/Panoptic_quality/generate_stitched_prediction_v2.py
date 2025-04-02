import numpy as np
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO as COCO
import os 
import cv2
import json
# Load calibration data
import glob
import re
import PIL.Image as Image
import argparse
import copy
from panopticapi.utils import get_traceback, rgb2id,id2rgb, IdGenerator, save_json
from multiprocessing import Pool, cpu_count
from scipy.ndimage import binary_dilation


def mask2rle(mask: np.ndarray):
    from pycocotools.mask import encode
    rle_encoded = encode(mask)
    return rle_encoded['counts'].decode()

def rle2mask(rle_encoded):
    from pycocotools.mask import decode
    binary_mask = decode(rle_encoded)
    return binary_mask

########### copied from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
def annToRLE(ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        #t = imgs[ann['original_image_id']]
        #h, w = t['height'], t['width']
        segm = ann['segmentation']
        h, w = segm['size']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle



def find_mapped_annotations(input_directory, scene_name,camera_mapping,output_folder,cats_info):
        categories,categories_id2_thing,categories_id2_name = cats_info
        merged_annotations={}
        def find_matched_files(directory, pattern):
            """
            Searches for files in the specified directory that match a given pattern.

            :param directory: The directory to search for files.
            :param pattern: The pattern to match file names.
            :return: A list of matched file paths.
            """
            # Create a pattern that matches the desired files
            search_pattern = os.path.join(directory, f"*{pattern}*.json")
            # Use glob to find all files matching the pattern
            matched_files = glob.glob(search_pattern)
            return matched_files

        annotation_file_list=find_matched_files(input_directory, scene_name)
        
        
        annotation_dict={}
        for files in annotation_file_list:
            camera_id=re.compile(r"_camera_(\d+)\.json").search(files).group(1)
            with open(files, 'r') as f:
                annotation_dict[int(camera_id)] = json.load(f)
        
        #print("annotation_file_list is",annotation_file_list)
        #return
        base_name = scene_name

        merged_annotations['images']=[]#annotation_dict["0"]['images']
        merged_annotations["annotations"]=[]
        img_ids=[]

        id_generator = IdGenerator(categories)
        file_names = []

        global_id = 0 
        for img in annotation_dict[0]['images']: # Simply using camera id = 0 
            img_info=img
            img_info["id"]=img["original_id"]
            img_info["file_name"]=img["original_file_name"].split("/")[1]+"/"+img["original_file_name"].split("/")[-1]
            img_ids.append(img["original_id"])
            img_info["height"]=img["height"]
            img_info["width"]=img_info["width"]*5
            img_info.pop("camera_id")
            img_info.pop("original_id")
            file_names.append(img_info["original_file_name"])
            #img_info["file_name"] = '{}.png'.format(img_info["original_file_name"].rsplit('/')[-1].split('.')[0])
            merged_annotations['images'].append(img_info)
        #print("merged_annotations",merged_annotations)
        stitched_image_annotations_dict = {}
        single_image_annotations_dict = {}
        single_ann = {}
        for img_id,image_name in zip(img_ids,file_names):

            for cam_id in annotation_dict.keys():
                single_image_annotations_dict[cam_id] = [ann for ann in annotation_dict[cam_id]["annotations"]  if ann["original_image_id"]==img_id]
            pan_format, segments_info ,BaseColor, global_id= stitching_annotations(single_image_annotations_dict,camera_mapping,id_generator,cats_info,global_id)

            if BaseColor is not None:
                image = Image.fromarray(BaseColor)
                out_file_name = '{}.png'.format(image_name.rsplit('/')[-1].split('.')[0])
                #print("out_file_name is ",out_file_name)
                #image.save(out_file_name))
            
            

            merged_annotations["annotations"].extend(segments_info)

            
        merged_annotations["licenses"] = annotation_dict[0]['licenses']
        merged_annotations["info"] = annotation_dict[0]['info']
        merged_annotations["categories"] = annotation_dict[0]['categories']
        return merged_annotations    




def calculate_intersections(A, B,pred_class,seg_id2_category_id,seg_id2_camera_id,cur_camera_id):
    if A.shape != B.shape:
        raise ValueError("Both maps A and B must have the same shape.")
    
    # Identify unique segment IDs in A
    unique_segments = np.unique(A)
    mask_B = B != 0
    #if structure is None:
    structure = np.ones((5, 5))
    # dilation mask to merge masks
    dilated_maskB = binary_dilation(mask_B, structure=structure)
    dilated_maskB
    # Dictionary to store intersection counts
    intersections = {}
    
    # Calculate intersection for each unique segment in A
    top_1_overlaped={}
    top1_area = 0
    for segment_id in unique_segments:
        if segment_id == 0:
            continue
        
        mask_A = A == segment_id
        intersection = np.logical_and(mask_A, dilated_maskB)
        intersection_area = np.sum(intersection)
        
        if intersection_area > 0 and (pred_class == seg_id2_category_id[segment_id]) and (cur_camera_id not in seg_id2_camera_id[segment_id]): # not from same camera
            intersections[segment_id] = intersection_area
            if intersection_area> top1_area:
                top1_area=intersection_area
                top_1_overlaped={}
                top_1_overlaped[segment_id] = top1_area
    #return intersections
    return top_1_overlaped

def stitching_annotations(single_image_annotations_dict,camera_mapping,id_generator,cats_info,global_id):

    
    categories,categories_id2_thing,categories_id2_name = cats_info
    BaseMask = np.zeros((480, 3760), dtype=np.uint32)
    ### BaseMask stores the segment id of stitched image
    BaseColor = np.zeros((480, 3760,3), dtype=np.uint8)
    stitched_segments_info = []
    seg_id = 0

    seg_id2_category_id = {}
    seg_id2_camera_id = {}

    stuff_memory_list = {}
    for camera_id in [6, 8, 0, 2, 4]:
        # Decode Y and X coordinates
        Ys = camera_mapping[camera_id][:, :, ::2].astype(np.int32)
        Xs = camera_mapping[camera_id][:, :, 1::2].astype(np.int32)

        valid_mask = (Ys != -1) & (Xs != -1) & (Ys < BaseMask.shape[0]) & (Xs < BaseMask.shape[1])

        #print("Ys.shape",Ys.shape) #(480,752,5)
        #print("Xs.shape",Xs.shape) #(480,752,5)

        flat_Ys = Ys[valid_mask]
        flat_Xs = Xs[valid_mask]

        #single_panoptic_mask = np.zeros((480, 752), dtype=np.uint32)
        for ann in single_image_annotations_dict[camera_id]:
            rle = annToRLE(ann)
            binary_mask = rle2mask(rle) > 0
            pred_class = ann['category_id']
            isthing =  categories_id2_thing[pred_class]

            ## For Stuff classes, different segments merges into same segments.
            if not isthing:
                if int(pred_class) in stuff_memory_list.keys():
                    temp_seg_id = stuff_memory_list[int(pred_class)]
                    single_panoptic_mask = np.zeros((480, 752), dtype=np.uint32)
                    single_panoptic_mask[binary_mask] = temp_seg_id #stuff_memory_list[temp_seg_id]
                    mask_flat = single_panoptic_mask[:, :, np.newaxis].repeat(5, axis=2)
                    mask_flat = mask_flat[valid_mask]
                    temp_mask = np.zeros((480, 3760), dtype=np.uint32)
                    temp_mask[flat_Ys, flat_Xs] = mask_flat
                    BaseMask[temp_mask==temp_seg_id] = temp_seg_id
                    continue
                else:
                    stuff_memory_list[int(pred_class)] = seg_id + 1

            temp_seg_id = seg_id + 1
            single_panoptic_mask = np.zeros((480, 752), dtype=np.uint32)
            single_panoptic_mask[binary_mask] = temp_seg_id

            
            mask_flat = single_panoptic_mask[:, :, np.newaxis].repeat(5, axis=2)
            mask_flat = mask_flat[valid_mask]
            temp_mask = np.zeros((480, 3760), dtype=np.uint32)
            temp_mask[flat_Ys, flat_Xs] = mask_flat

            if not isthing:
                intersections={}
            else:
                intersections=calculate_intersections(BaseMask,temp_mask,pred_class,seg_id2_category_id,seg_id2_camera_id,camera_id)
            #intersections=calculate_intersections(BaseMask,temp_mask,pred_class,seg_id2_category_id)
            if intersections:
                ### if has intersection, resetting previous masks
                for inter_seg_id in intersections:
                    BaseMask[temp_mask==temp_seg_id] = inter_seg_id #setting current one to previous one
                #seg_id2_category_id[temp_seg_id] = pred_class
                seg_id2_camera_id[inter_seg_id].append(camera_id) #inter_seg_id must already exist

            else:
                BaseMask[temp_mask==temp_seg_id] = temp_seg_id
                seg_id2_category_id[temp_seg_id] = pred_class
                seg_id2_camera_id[temp_seg_id] = [camera_id]
                seg_id = temp_seg_id

    labels, labels_cnt = np.unique(BaseMask, return_counts=True)
    tmp_global_id = global_id
    for lable,labels_cnt in zip(labels,labels_cnt):
        #segment,counts_number = lable
        if lable==0:
            continue
        #global_id = global_id+lable
        binary_map=np.zeros([480, 3760], dtype=np.int32)
        binary_map[BaseMask==lable]=1
        binary_mask_contiguous = np.asfortranarray(binary_map)
        segmentation_rle = mask2rle(binary_mask_contiguous.astype(np.uint8))
        if tmp_global_id+lable>global_id:
            global_id = int(tmp_global_id+lable)
        stitched_segments_info.append({
                            "id": global_id,
                            "id_on_image": int(lable),
                            "image_id":ann['original_image_id'],
                            "segmentation": {"counts": segmentation_rle, "size": [480, 3760]},
                            "isthing": bool(categories_id2_thing[seg_id2_category_id[lable]]),
                            #"category_id": pred_class,
                            "category_id": int(seg_id2_category_id[lable]),
                            "area": int(labels_cnt),
                            "bbox": [],
                            "attributes":{'is_crowded': False,'occluded': False}
                        }
                    )
    ####### Now decode the stitched id Map
    
    
    ### output color
    labels, labels_cnt = np.unique(BaseMask, return_counts=True)
    #print("labels",labels)
    for label in labels:
        if label==0:
            continue
        segment_id, color = id_generator.get_id_and_color(seg_id2_category_id[label])
        BaseColor[BaseMask==label] = color
    return BaseMask, stitched_segments_info,None,global_id
    #return BaseMask, stitched_segments_info,BaseColor,global_id



def find_unique_json_names(directory):
    pattern = re.compile(r"(.+)_camera_\d+\.json")
    unique_names = set()
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            unique_names.add(match.group(1))
    return list(unique_names)

def process_file(input_dir, output_dir, file_patterns, camera_mapping,cats_info):
    #print("file_patterns is",file_patterns)
    #return
    for file_pattern in file_patterns:
        #print("file_pattern is",file_pattern)
        #return
        segmentations_folder = os.path.join(output_dir, file_pattern)
        if not os.path.isdir(segmentations_folder):
            os.makedirs(segmentations_folder, exist_ok=True)

        output_json_file = os.path.join(output_dir, file_pattern + ".json")
        output_folder = os.path.join(output_dir, file_pattern)
        single_view_json = find_mapped_annotations(input_dir, file_pattern, camera_mapping,output_folder,cats_info)

        with open(output_json_file, 'w') as json_file:
            json.dump(single_view_json, json_file, indent=4)



#def process_chunk(directory_path, output_dir, file_patterns_chunk, camera_mapping, cats_info):
#    for file_pattern in file_patterns_chunk:
#        # Your file processing logic here
#        print(f"Processing {file_pattern} on pid {os.getpid()}...")  # Example action

def chunkify(lst, n):
    """Return a list of successive n-sized chunks from lst."""
    #for i in range(0, len(lst), n):
    #    yield lst[i:i + n]
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def main(args):
    with open(args.categories_json_file, 'r') as f:
        categories_list = json.load(f)

    categories = {category['id']: category for category in categories_list}
    categories_id2_thing = {category['id']: category["isthing"] for category in categories_list}
    categories_id2_name = {category['id']: category["name"] for category in categories_list}
    cats_info = [categories,categories_id2_thing,categories_id2_name]
    camera_mapping = {}
    for i, cam_id in enumerate([6, 8, 0, 2, 4]):
        camera_mapping[cam_id] = np.load(os.path.join(args.calibration_dir, "indi2stitch_mappings", f"indi2stitch_mapping_camera_{cam_id}.npy"))

    file_patterns = find_unique_json_names(args.directory_path)
    num_cpus = cpu_count()

    # Split file patterns into chunks, one per CPU
    chunks = chunkify(file_patterns, len(file_patterns) // num_cpus + 1)

    # Create a pool of workers
    with Pool(processes=num_cpus) as pool:
        # Each process gets a chunk of file patterns to process
        pool.starmap(process_file, [(args.directory_path, args.output_dir, chunk, camera_mapping, cats_info) for chunk in chunks])

    #process_file(args.directory_path, args.output_dir, file_patterns, camera_mapping,cats_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files for panoptic segmentation.")
    parser.add_argument("--directory_path", help="Input directory where single view JSON files are regenerated.")
    parser.add_argument("--output_dir", help="Output directory for stitched inference JSONs.")
    parser.add_argument("--calibration_dir", help="Directory where calibration files are stored.")
    parser.add_argument("--categories_json_file", help="Path to the categories JSON file in COCO format.")
    parser.add_argument("--vis", action="store_true", help="Output visualization or not") ### future support

    args = parser.parse_args()
    main(args)