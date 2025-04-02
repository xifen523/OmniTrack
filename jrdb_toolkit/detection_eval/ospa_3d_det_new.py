import glob
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import argparse


import torch
import iou3d_nms_cuda

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D object detection results using OSPA metric')
    parser.add_argument('--gt_dir', type=str, \
                        default='/home/lk/workspase/python/dection/Sparse4D/infer/det/gt_2', \
                        help='Directory containing ground truth files')
    parser.add_argument('--det_dir', type=str, \
                        default='/home/lk/workspase/python/dection/Sparse4D/jrdb_toolkit/detection_eval/KITTI2', \
                        help='Directory containing detection files')
    parser.add_argument('--score_x', type=float, default=1.0, \
                        help='Score for detections in x-direction')
    
    parser.add_argument('--save_dir', type=str, default='./', \
                        help='Directory to save evaluation results')
    
    return parser.parse_args()
    

def load_file(file_path):
    folders = glob.glob(file_path+"/*/")
    folders = sorted(folders)
    return_list = []
    file_dict=dict()
    for folder in folders:
        file_list = glob.glob(folder + "/*.txt")
        file_list = sorted(file_list)
        return_list.extend(file_list)
        if folder not in file_dict:
            file_dict[folder.split('/')[-2]]=file_list
        else:
            file_dict[folder.split('/')[-2]].append(file_list)

    return file_dict, return_list
def main(args):
    gt_file_dict, gt_file_list = load_file(args.gt_dir)
    infer_file_dict, infer_file_list = load_file(args.det_dir)
    assert len(gt_file_list) == len(infer_file_list), 'the number of test files is not equal to the number of gt files'
    
    ospa_dict = dict()
    for (gt_key, gt_path), (infer_key, infer_path)in zip(gt_file_dict.items(),infer_file_dict.items()):
        print("handle seq:",gt_key)
        assert gt_key == infer_key, 'the sequence name is not equal'
        for gt_file, infer_file in zip(gt_path, infer_path):
            gt_df = pd.read_csv(gt_file, header = None, sep = ' ').values
            infer_df = pd.read_csv(infer_file, header = None, sep = ' ').values

            # fiter the gt_df and infer_df But I can't understand the meaning of the code below
            # distance_test=(infer_df[:,8]**2+infer_df[:,10]**2)**0.5   # KITTI format 
            distance_infer=(infer_df[:,9]**2+infer_df[:,11]**2)**0.5
            infer_df=infer_df[distance_infer<5]
            distance_gt=(gt_df[:,9]**2+gt_df[:,11]**2)**0.5
            gt_df=gt_df[distance_gt<5]

            # get bbox
            # infer_bbox = infer_df[:,9:16]
            infer_bbox = infer_df[:,9:16]
            gt_bbox = gt_df[:,9:16]
            
            # swap the order of the bbox  from 
            # KITTI:x_size, y_size, z_size, x, y, z, theta to 
            # ours: x, y, z, x_size, y_size, z_size, theta
            infer_bbox = infer_bbox[:,[3,4,5,0,1,2,6]]
            gt_bbox = gt_bbox[:,[3,4,5,0,1,2,6]]

            # get score
            infer_score = infer_df[:,16]

            # get category 
            
            # compute ospa
            ospa = calculate_ospa_single_frame(infer_bbox, gt_bbox, infer_score, c=1)
            if gt_key not in ospa_dict:
                ospa_dict[gt_key]= [ospa]
            else:
                ospa_dict[gt_key].append(ospa)
            
    # ospa_dict={k:np.array(v).mean(axis=0) for k,v in ospa_dict.items()}
    # _sum=np.zeros(3)
    # for k,v in ospa_dict.items():
    #     _sum=_sum+v
    # ospa_dict['overall']=_sum/len(ospa_dict.keys())
    # with open("ospa1_1.txt", 'w') as f: 
    #     for key, value in ospa_dict.items(): 
    #         value=list(value)
    #         value=[str(v) for v in value]
    #         f.write('%s,%s\n' % (key, ','.join(list(value))))
    # print('the result is saved in ospa1_1.txt')

    ospa_dict = {k: np.mean(v, axis=0) for k, v in ospa_dict.items()}

    # 计算总和
    _sum = np.sum(list(ospa_dict.values()), axis=0)

    # 计算 overall 的均值
    ospa_dict['overall'] = _sum / len(ospa_dict)

    # 将结果写入文件
    with open("ospa1_1.txt", 'w') as f:
        for key, value in ospa_dict.items():
            value_str = ','.join(map(str, value))
            f.write(f'{key},{value_str}\n')

    return ospa_dict

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
    overlaps_3d = overlaps_bev * overlaps_h * score.view(1, -1)

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    # convert to numpy
    iou3d = iou3d.cpu().numpy()
    return iou3d


# usage: python --det_dir xxx --gt_dir xxx
if __name__ == "__main__":
    args = parse_args()
    main(args)