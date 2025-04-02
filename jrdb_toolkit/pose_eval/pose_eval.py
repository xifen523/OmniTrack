import argparse
import numpy as np
import glob
import os
import sys
import json
from scipy.optimize import linear_sum_assignment
import time
import cv2


IOU_THRESHOLD = 0.5

TRAIN_LOCATIONS = [
    'bytes-cafe-2019-02-07_0',
    'huang-lane-2019-02-12_0',
    'clark-center-2019-02-28_0',
    'clark-center-2019-02-28_1',
    'clark-center-intersection-2019-02-28_0',
    'cubberly-auditorium-2019-04-22_0',
    'forbes-cafe-2019-01-22_0',
    'gates-159-group-meeting-2019-04-03_0',
    'gates-ai-lab-2019-02-08_0',
    'gates-basement-elevators-2019-01-17_1',
    'gates-to-clark-2019-02-28_1',
    'hewlett-packard-intersection-2019-01-24_0',
    'huang-2-2019-01-25_0',
    'huang-basement-2019-01-25_0',
    'jordan-hall-2019-04-22_0',
    'memorial-court-2019-03-16_0',
    'meyer-green-2019-03-16_0',
    'nvidia-aud-2019-04-18_0',
    'packard-poster-session-2019-03-20_0',
    'packard-poster-session-2019-03-20_1',
    'packard-poster-session-2019-03-20_2',
    'stlc-111-2019-04-19_0',
    'svl-meeting-gates-2-2019-04-08_0',
    'svl-meeting-gates-2-2019-04-08_1',
    'tressider-2019-03-16_0',
    'tressider-2019-03-16_1',
    'tressider-2019-04-26_2'
]
TEST_LOCATIONS = [
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
TEST_INDI_LOCATIONS = ['stlc-111-2019-04-19_2_image6', 'tressider-2019-04-26_3_image2', 'outdoor-coupa-cafe-2019-02-06_0_image8',
        'gates-ai-lab-2019-04-17_0_image6', 'gates-ai-lab-2019-04-17_0_image0',
        'cubberly-auditorium-2019-04-22_1_image0', 'stlc-111-2019-04-19_1_image4', 'nvidia-aud-2019-04-18_2_image2',
        'nvidia-aud-2019-04-18_1_image2', 'food-trucks-2019-02-12_0_image6', 'nvidia-aud-2019-01-25_0_image2',
        'indoor-coupa-cafe-2019-02-06_0_image4', 'cubberly-auditorium-2019-04-22_1_image2',
        'quarry-road-2019-02-28_0_image2', 'hewlett-class-2019-01-23_1_image2', 'gates-foyer-2019-01-17_0_image8',
        'food-trucks-2019-02-12_0_image0', 'meyer-green-2019-03-16_1_image2', 'indoor-coupa-cafe-2019-02-06_0_image8',
        'discovery-walk-2019-02-28_0_image2', 'discovery-walk-2019-02-28_1_image4', 'quarry-road-2019-02-28_0_image0',
        'gates-ai-lab-2019-04-17_0_image8', 'gates-foyer-2019-01-17_0_image0', 'tressider-2019-04-26_3_image8',
        'hewlett-class-2019-01-23_0_image2', 'outdoor-coupa-cafe-2019-02-06_0_image0',
        'hewlett-class-2019-01-23_0_image0', 'quarry-road-2019-02-28_0_image4',
        'gates-basement-elevators-2019-01-17_0_image0', 'tressider-2019-04-26_0_image8',
        'nvidia-aud-2019-04-18_2_image0', 'lomita-serra-intersection-2019-01-30_0_image2',
        'huang-2-2019-01-25_1_image8', 'nvidia-aud-2019-01-25_0_image0', 'quarry-road-2019-02-28_0_image6',
        'tressider-2019-04-26_0_image0', 'huang-2-2019-01-25_1_image4', 'huang-intersection-2019-01-22_0_image0',
        'nvidia-aud-2019-04-18_1_image8', 'huang-2-2019-01-25_1_image2', 'meyer-green-2019-03-16_1_image0',
        'outdoor-coupa-cafe-2019-02-06_0_image4', 'hewlett-class-2019-01-23_1_image8',
        'cubberly-auditorium-2019-04-22_1_image4', 'hewlett-class-2019-01-23_1_image4', 'stlc-111-2019-04-19_2_image4',
        'stlc-111-2019-04-19_1_image2', 'nvidia-aud-2019-04-18_1_image6',
        'gates-basement-elevators-2019-01-17_0_image2', 'indoor-coupa-cafe-2019-02-06_0_image2',
        'nvidia-aud-2019-01-25_0_image6', 'food-trucks-2019-02-12_0_image8', 'serra-street-2019-01-30_0_image8',
        'hewlett-class-2019-01-23_0_image6', 'tressider-2019-04-26_3_image0', 'tressider-2019-04-26_1_image4',
        'serra-street-2019-01-30_0_image6', 'tressider-2019-04-26_0_image2', 'nvidia-aud-2019-01-25_0_image4',
        'discovery-walk-2019-02-28_0_image0', 'huang-2-2019-01-25_1_image6', 'tressider-2019-04-26_0_image6',
        'gates-ai-lab-2019-04-17_0_image4', 'cubberly-auditorium-2019-04-22_1_image6',
        'hewlett-class-2019-01-23_1_image6', 'discovery-walk-2019-02-28_1_image0',
        'lomita-serra-intersection-2019-01-30_0_image4', 'serra-street-2019-01-30_0_image4',
        'huang-intersection-2019-01-22_0_image8', 'gates-foyer-2019-01-17_0_image2',
        'gates-to-clark-2019-02-28_0_image8', 'outdoor-coupa-cafe-2019-02-06_0_image2',
        'nvidia-aud-2019-04-18_1_image0', 'nvidia-aud-2019-04-18_2_image6', 'stlc-111-2019-04-19_2_image0',
        'tressider-2019-04-26_3_image4', 'huang-2-2019-01-25_1_image0', 'gates-to-clark-2019-02-28_0_image2',
        'lomita-serra-intersection-2019-01-30_0_image0', 'nvidia-aud-2019-04-18_2_image8',
        'quarry-road-2019-02-28_0_image8', 'nvidia-aud-2019-01-25_0_image8', 'meyer-green-2019-03-16_1_image6',
        'stlc-111-2019-04-19_1_image0', 'hewlett-class-2019-01-23_1_image0',
        'gates-basement-elevators-2019-01-17_0_image4', 'huang-intersection-2019-01-22_0_image6',
        'tressider-2019-03-16_2_image4', 'lomita-serra-intersection-2019-01-30_0_image8',
        'tressider-2019-04-26_1_image6', 'meyer-green-2019-03-16_1_image8', 'stlc-111-2019-04-19_2_image8',
        'gates-ai-lab-2019-04-17_0_image2', 'outdoor-coupa-cafe-2019-02-06_0_image6', 'tressider-2019-04-26_0_image4',
        'indoor-coupa-cafe-2019-02-06_0_image6', 'hewlett-class-2019-01-23_0_image4',
        'gates-basement-elevators-2019-01-17_0_image6', 'discovery-walk-2019-02-28_0_image6',
        'tressider-2019-04-26_1_image2', 'gates-foyer-2019-01-17_0_image6', 'tressider-2019-04-26_1_image8',
        'gates-to-clark-2019-02-28_0_image0', 'gates-to-clark-2019-02-28_0_image6', 'hewlett-class-2019-01-23_0_image8',
        'food-trucks-2019-02-12_0_image2', 'cubberly-auditorium-2019-04-22_1_image8', 'tressider-2019-03-16_2_image0',
        'stlc-111-2019-04-19_2_image2', 'discovery-walk-2019-02-28_0_image8', 'tressider-2019-03-16_2_image6',
        'serra-street-2019-01-30_0_image0', 'tressider-2019-03-16_2_image8', 'stlc-111-2019-04-19_1_image8',
        'lomita-serra-intersection-2019-01-30_0_image6', 'tressider-2019-04-26_3_image6',
        'food-trucks-2019-02-12_0_image4', 'stlc-111-2019-04-19_1_image6', 'huang-intersection-2019-01-22_0_image4',
        'discovery-walk-2019-02-28_0_image4', 'huang-intersection-2019-01-22_0_image2',
        'nvidia-aud-2019-04-18_1_image4', 'tressider-2019-03-16_2_image2', 'serra-street-2019-01-30_0_image2',
        'gates-basement-elevators-2019-01-17_0_image8', 'gates-to-clark-2019-02-28_0_image4',
        'discovery-walk-2019-02-28_1_image6', 'discovery-walk-2019-02-28_1_image8',
        'discovery-walk-2019-02-28_1_image2', 'tressider-2019-04-26_1_image0', 'nvidia-aud-2019-04-18_2_image4',
        'gates-foyer-2019-01-17_0_image4', 'indoor-coupa-cafe-2019-02-06_0_image0', 'meyer-green-2019-03-16_1_image4']

def get_per_kp_oks_matrix(gt_annots, pr_annots, sigmas=[
        0.079, 0.025, 0.025, 0.079, 0.026, 0.079, 0.072, 0.072, 0.107, 
        0.062, 0.107, 0.107, 0.062, 0.087, 0.087, 0.089, 0.089
    ]):
    """
    gt_annots: ground-truth list of coco keypoint annotations
    pr_annots: predicted list of coco keypoint annotations
    
    returns:
        dist_matrix: For a set of G ground-truth poses and P predicted poses,
            this is a GxP matrix of metric between each pair of GT and PR pose. 
    """
    sigmas = np.asarray(sigmas)
    var = (sigmas * 2)**2

    results = np.zeros((len(gt_annots), len(pr_annots), 17))
    k= len(sigmas)

    # Based on https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    for j, gt in enumerate(gt_annots):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; 
        
        vg = np.ones_like(g[2::3]) ## MODIFIED
        
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(pr_annots):
            d = np.array(dt['keypoints'])
            xd = d[0::3]; yd = d[1::3]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / var / (gt['area']+np.spacing(1)) / 2
            if k1 > 0:
                e=e[vg > 0]
            results[j, i] = np.exp(-e)
            
    return results

def get_oks_matrix(gt_annots, pr_annots, sigmas=[
        0.079, 0.025, 0.025, 0.079, 0.026, 0.079, 0.072, 0.072, 0.107, 
        0.062, 0.107, 0.107, 0.062, 0.087, 0.087, 0.089, 0.089
    ]):
    """
    gt_annots: ground-truth list of coco keypoint annotations
    pr_annots: predicted list of coco keypoint annotations
    
    returns:
        dist_matrix: For a set of G ground-truth poses and P predicted poses,
            this is a GxP matrix of metric between each pair of GT and PR pose. 
    """
    sigmas = np.asarray(sigmas)
    var = (sigmas * 2)**2

    results = np.zeros((len(gt_annots), len(pr_annots)))

    # Based on https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    for j, gt in enumerate(gt_annots):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(pr_annots):
            d = np.array(dt['keypoints'])
            xd = d[0::3]; yd = d[1::3]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((len(g)))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / var / (gt['area']+np.spacing(1)) / 2
            if k1 > 0:
                e=e[vg > 0]
            results[j, i] = np.sum(np.exp(-e)) / e.shape[0]
            
    return results

def get_ospa_old(gt_annots_iid, pr_annots_iid):
    # If both sets emtpy, return 0
    if len(gt_annots_iid) == 0 and len(pr_annots_iid) == 0:
        return 0
        
    # If exactly one set empty, return 1
    if len(gt_annots_iid) == 0 and len(pr_annots_iid) != 0:
        return 1
    if len(gt_annots_iid) != 1 and len(pr_annots_iid) == 0:
        return 1
    
    # Cost matrix between each GT and PR pose
    cost_matrix = 1 - get_oks_matrix(gt_annots_iid, pr_annots_iid)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matching_cost = cost_matrix[row_ind, col_ind].sum() 
    cardinality_cost = abs(len(gt_annots_iid) - len(pr_annots_iid))
    best_metric = (matching_cost + cardinality_cost) / max(len(gt_annots_iid), len(pr_annots_iid))
    return best_metric

def box_xywh_to_xyxy(box):
    x, y, w, h = box
    b = [x, y, x + w, y + h]
    return b

def boxes_from_annos(annos):
    box_list = [box_xywh_to_xyxy(anno['bbox']) for anno in annos]
    return np.stack(box_list).T


def matrix_iou(X, Y):
    """Calculates IoU between each box in X and each box in Y.
    """
    if not np.any(np.any(X[2:4,:] > X[0:2,:])):
        print("The coordinates of the second corner of a rectangle should be bigger than the coordinates of the first corner")
        return np.zeros((Y.shape[1], X.shape[1]))

    if not np.any(np.any(Y[2:4,:] > Y[0:2,:])):
        print("The coordinates of the second corner of a rectangle should be bigger than the coordinates of the first corner")
        return np.zeros((Y.shape[1], X.shape[1]))

    # Calculate sizes of the input point patterns
    n = X.shape[1]
    m = Y.shape[1]
    
    score_X = np.ones((len(X[0]),))
    
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
    return V_IoU.reshape((-1, X.shape[1]))

def get_unseen_boxes(boxes, annos):
    pose_track_ids = set([anno['track_id'] for anno in annos])

    unseen_box_list = []
    for box in boxes:
        tid = int(box['label_id'].split(':')[-1])
        if tid not in pose_track_ids:
            unseen_box_list.append(box_xywh_to_xyxy(box['box']))
    
    if len(unseen_box_list) == 0:
        return []
    return np.stack(unseen_box_list).T

def get_ospa(gt_annots_iid, pr_annots_iid, gt_boxes_unlabeled):

    def calculate_forgiveness(gt_idx):
        if len(gt_boxes_unlabeled) == 0 or True:
            return 0
        # Find optimal matching from predicted boxes to all unlabeled boxes
        pr_boxes = boxes_from_annos(pr_annots_iid)
        ious = matrix_iou(gt_boxes_unlabeled, pr_boxes) > IOU_THRESHOLD
        gt_box_idx, pr_box_idx = linear_sum_assignment(ious, maximize=True)

        # Forgiveness for poses whose boxes match above IoU threshold to unlabeled box
        gt_box_idx_matched = gt_box_idx[ious[gt_box_idx, pr_box_idx] == True]
        forgive = len(set(gt_box_idx_matched) - set(gt_idx))
        return forgive
    
    # If both sets emtpy, return 0
    if len(gt_annots_iid) == 0 and len(pr_annots_iid) == 0:
        return 0
        
    # If exactly one set empty, return 1
    if len(gt_annots_iid) == 0 and len(pr_annots_iid) != 0:
        return 1
    if len(gt_annots_iid) != 1 and len(pr_annots_iid) == 0:
        return 1

    cost_matrix = 1 - get_oks_matrix(gt_annots_iid, pr_annots_iid)
    gt_idx, pr_idx = linear_sum_assignment(cost_matrix)
    forgive = calculate_forgiveness(gt_idx)

    num_gt, num_pr = len(gt_annots_iid), len(pr_annots_iid) - forgive
    matching_cost = cost_matrix[gt_idx, pr_idx].sum() 
    cardinality_cost = abs(num_gt - num_pr)
    best_metric = (matching_cost + cardinality_cost) / max(num_gt, num_pr)
    return best_metric


def ospa_for_loc(gt_dir, pr_dir, box_dir, location):
    """Gets metric for all frames of a location.
    
    gt_dir: path to ground-truth annotation file
    pr_dir: path to predictions file
    box_dir: path to 2d box file
    location: scene name, such as 'cubberly-auditorium-2019-04-22_1'
    
    Returns a list of floats containing the OSPA score for each frame.
    """
    with open(os.path.join(gt_dir, location+".json"), 'r') as data_file:
        data_gt = json.load(data_file)

    with open(os.path.join(pr_dir, location+".json"), 'r') as data_file:
        data_pr = json.load(data_file)
        
    with open(os.path.join(box_dir, location+".json"), 'r') as data_file:
        boxes = json.load(data_file)

    all_iids = [im['id'] for im in data_gt['images']]
    iid_to_filename = lambda iid : '{:06d}.jpg'.format(iid-1)

    gt_annots_by_iid = {iid: [] for iid in all_iids}
    for ann in data_gt['annotations']:
        gt_annots_by_iid[ann['image_id']].append(ann)

    pr_annots_by_iid = {iid: [] for iid in all_iids}
    for ann in data_pr['annotations']:
        if ann['image_id'] in pr_annots_by_iid:
            pr_annots_by_iid[ann['image_id']].append(ann)

    ospa_list = []

    for iid in all_iids:
        gt_annots_iid = gt_annots_by_iid[iid] if iid in gt_annots_by_iid else []       
        pr_annots_iid = pr_annots_by_iid[iid] if iid in pr_annots_by_iid else []

#         print(iid, iid_to_filename(iid))
        if iid_to_filename(iid) in boxes['labels']:
            boxes_iid = boxes['labels'][iid_to_filename(iid)] 
        else:
            boxes_iid = []
            
        gt_boxes_unlabeled = get_unseen_boxes(boxes_iid, gt_annots_iid)
        
        score = get_ospa(gt_annots_iid, pr_annots_iid, gt_boxes_unlabeled)
        ospa_list.append(score)

    return ospa_list

# compute recall/precision curve (RPC) values
def computeRPC(scores,labels,totalPos):

    precision = np.zeros(len(scores))
    recall    = np.zeros(len(scores))
    npos = 0;

    idxsSort = np.array(scores).argsort()[::-1]
    labelsSort = labels[idxsSort];

    for sidx in range(len(idxsSort)):
        if (labelsSort[sidx] == 1):
            npos += 1
        # recall: how many true positives were found out of the total number of positives?
        recall[sidx]    = 1.0*npos / totalPos
        # precision: how many true positives were found out of the total number of samples?
        precision[sidx] = 1.0*npos / (sidx + 1)

    return precision, recall, idxsSort


# compute Average Precision using recall/precision values
def VOCap(rec,prec):

    mpre = np.zeros([1,2+len(prec)])
    mpre[0,1:len(prec)+1] = prec
    mrec = np.zeros([1,2+len(rec)])
    mrec[0,1:len(rec)+1] = rec
    mrec[0,len(rec)+1] = 1.0

    for i in range(mpre.size-2,-1,-1):
        mpre[0,i] = max(mpre[0,i],mpre[0,i+1])

    i = np.argwhere( ~np.equal( mrec[0,1:], mrec[0,:mrec.shape[1]-1]) )+1
    i = i.flatten()

    # compute area under the curve
    ap = np.sum( np.multiply( np.subtract( mrec[0,i], mrec[0,i-1]), mpre[0,i] ) )

    return ap


def computeMetrics(scoresAll, labelsAll, nGTall):
    apAll = np.zeros((nGTall.shape[0] + 1, 1))
    recAll = np.zeros((nGTall.shape[0] + 1, 1))
    preAll = np.zeros((nGTall.shape[0] + 1, 1))
    # iterate over joints
    for j in range(nGTall.shape[0]):
        scores = np.zeros([0, 0], dtype=np.float32)
        labels = np.zeros([0, 0], dtype=np.int8)
        # iterate over images
        for imgidx in range(nGTall.shape[1]):
            scores = np.append(scores, scoresAll[j][imgidx])
            labels = np.append(labels, labelsAll[j][imgidx])
        # compute recall/precision values
        nGT = sum(nGTall[j, :])
        precision, recall, scoresSortedIdxs = computeRPC(scores, labels, nGT)
        if (len(precision) > 0):
            apAll[j] = VOCap(recall, precision) * 100
            preAll[j] = precision[len(precision) - 1] * 100
            recAll[j] = recall[len(recall) - 1] * 100
    idxs = np.argwhere(~np.isnan(apAll[:nGTall.shape[0],0]))
    apAll[nGTall.shape[0]] = apAll[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(recAll[:nGTall.shape[0],0]))
    recAll[nGTall.shape[0]] = recAll[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(preAll[:nGTall.shape[0],0]))
    preAll[nGTall.shape[0]] = preAll[idxs, 0].mean()

    return apAll, preAll, recAll

def average_precision_for_loc(gt_dir, pr_dir, box_dir, location, oks_threshold=0.5, nJoints=17):
    """Gets metric for all frames of a location.
    
    gt_dir: path to ground-truth annotation file
    pr_dir: path to predictions file
    box_dir: path to 2d box file
    location: scene name, such as 'cubberly-auditorium-2019-04-22_1'
    
    Returns a list of floats containing the AP score for each joint.
    """
    with open(os.path.join(gt_dir, location+".json"), 'r') as data_file:
        data_gt = json.load(data_file)

    with open(os.path.join(pr_dir, location+".json"), 'r') as data_file:
        data_pr = json.load(data_file)
        
    with open(os.path.join(box_dir, location+".json"), 'r') as data_file:
        boxes = json.load(data_file)

    all_iids = [im['id'] for im in data_gt['images']]
    iid_to_filename = lambda iid : '{:06d}.jpg'.format(iid-1)

    gt_annots_by_iid = {iid: [] for iid in all_iids}
    for ann in data_gt['annotations']:
        gt_annots_by_iid[ann['image_id']].append(ann)

    pr_annots_by_iid = {iid: [] for iid in all_iids}
    for ann in data_pr['annotations']:
        pr_annots_by_iid[ann['image_id']].append(ann)
        
    # part detection scores
    scoresAll = {}
    # positive / negative labels
    labelsAll = {}
    # number of annotated GT joints per image
    nGTall = np.zeros([nJoints, len(gt_annots_by_iid)])
    for pidx in range(nJoints):
        scoresAll[pidx] = {}
        labelsAll[pidx] = {}
        for imgidx in range(len(all_iids)):
            scoresAll[pidx][imgidx] = np.zeros([0, 0], dtype=np.float32)
            labelsAll[pidx][imgidx] = np.zeros([0, 0], dtype=np.int8)

    for iid in range(len(all_iids)):
        gtFrames = gt_annots_by_iid[all_iids[iid]]
        prFrames = pr_annots_by_iid[all_iids[iid]]

        # img = cv2.imread(f"/home/tho/datasets/JRDB2022/test_dataset_without_labels/images/image_stitched/{location}/{gtFrames[0]['image_id']-1:06d}.jpg")
        # for p in prFrames:
        #     kpts = np.array(p['keypoints']).reshape([-1,3])
        #     kpts = kpts[:,:2]
        #     for point in kpts:
        #         cv2.circle(img, (int(point[0]),int(point[1])),
        #                    radius=2, color=[255,255,255],thickness=3)
        #         cv2.imshow('debug', img)
        #         cv2.waitKey(0)


        if iid_to_filename(all_iids[iid]) in boxes['labels']:
            boxes_iid = boxes['labels'][iid_to_filename(all_iids[iid])] 
        else:
            boxes_iid = []

        gt_boxes_unlabeled = get_unseen_boxes(boxes_iid, gtFrames)

        dist = get_per_kp_oks_matrix(gtFrames, prFrames)
        # score of the predicted joint
        score = np.ones((len(prFrames), nJoints), dtype=bool)
        # body joint prediction exist
        hasPr = np.ones((len(prFrames), nJoints), dtype=bool)
        # body joint is annotated
        hasGT = np.ones((len(gtFrames), nJoints), dtype=bool)

        if len(gtFrames) and len(prFrames):
            match = dist > oks_threshold
            pck = match.sum(-1)

            # preserve best GT match only
            idx = np.argmax(pck, axis=0)
            val = np.max(pck, axis=0)
            for ridxPr in range(pck.shape[1]):
                for ridxGT in range(pck.shape[0]):
                    if (ridxGT != idx[ridxPr]):
                        pck[ridxGT, ridxPr] = 0
            prToGT = np.argmax(pck, axis=0)
            val = np.max(pck, axis=0)
            prToGT[val == 0] = -1

            # match greedily
            prToGT = np.asarray([-1]*len(prFrames))
            left = list(range(len(prFrames)))
            for gtIdx in range(len(gtFrames)):
                matchIdxTmp = pck[gtIdx,left].argmax()
                matchIdx = left[matchIdxTmp]
                del left[matchIdxTmp]
    #             prToGT[gtIdx] = matchIdx
                prToGT[matchIdx] = gtIdx
        
                if len(left) == 0:
                    break

            # assign predicted poses to GT poses
            for ridxPr in range(hasPr.shape[0]):
                if (ridxPr in prToGT):  # pose matches to GT
                    # GT pose that matches the predicted pose
                    ridxGT = np.argwhere(prToGT == ridxPr)
                    assert(ridxGT.size == 1)
                    ridxGT = ridxGT[0,0]
                    s = score[ridxPr, :]
                    m = np.squeeze(match[ridxPr, ridxGT, :])
                    hp = hasPr[ridxPr, :]
                    for i in range(len(hp)):
                        if (hp[i]):
                            scoresAll[i][iid] = np.append(scoresAll[i][iid], s[i])
                            labelsAll[i][iid] = np.append(labelsAll[i][iid], m[i])

                else:  # no matching to GT
                    pr_boxes = boxes_from_annos([prFrames[ridxPr]])
                    got_match = False
                    if len(gt_boxes_unlabeled) > 0:
                        got_match = (matrix_iou(gt_boxes_unlabeled, pr_boxes) > IOU_THRESHOLD).max()
                    # ignore PRs that match to bounding boxes
                    if not got_match:
                        s = score[ridxPr, :]
                        m = np.zeros([match.shape[2], 1], dtype=bool)
                        hp = hasPr[ridxPr, :]
                        for i in range(len(hp)):
                            if (hp[i]):
                                scoresAll[i][iid] = np.append(scoresAll[i][iid], s[i])
                                labelsAll[i][iid] = np.append(labelsAll[i][iid], m[i])
        elif not len(gtFrames):
            # No GT available. All predictions are false positives
            for ridxPr in range(hasPr.shape[0]):
                pr_boxes = boxes_from_annos([prFrames[ridxPr]])
                got_match = False
                if len(gt_boxes_unlabeled) > 0:
                    got_match = (matrix_iou(gt_boxes_unlabeled, pr_boxes) > IOU_THRESHOLD).max()
                # ignore PRs that match to bounding boxes
                if not got_match:
                    s = score[ridxPr, :]
                    m = np.zeros([nJoints, 1], dtype=bool)
                    hp = hasPr[ridxPr, :]
                    for i in range(len(hp)):
                        if hp[i]:
                            scoresAll[i][iid] = np.append(scoresAll[i][iid], s[i])
                            labelsAll[i][iid] = np.append(labelsAll[i][iid], m[i])


        for ridxGT in range(hasGT.shape[0]):
            hg = hasGT[ridxGT, :]
            for i in range(len(hg)):
                nGTall[i, iid] += hg[i]
                
    scores, labels, nGT = scoresAll, labelsAll, nGTall
    ap, pre, rec = computeMetrics(scores, labels, nGT)
    
    return ap.flatten().tolist(), rec.flatten().tolist()

def compute_ospa_pose(pred_path, 
                      box_path,
                      gt_path,
                      out_dir=None,
                      save_results=False):
    if out_dir is None:
        out_dir = pred_path

    gt_files = glob.glob(os.path.join(gt_path, "*.json"))
    box_files = glob.glob(os.path.join(box_path, "*.json"))
    assert len(gt_files) == len(box_files)


    real_pred_folder = pred_path
    file_list = glob.glob(real_pred_folder + "/*.json")
    if len(file_list) != len(gt_files):
        pred_folders = glob.glob(pred_path+"/*/")
        for folder in pred_folders:
            file_list = glob.glob(folder + "/*.json")
            if len(file_list) == len(gt_files):
                real_pred_folder = folder
                break

    ospa_dict = {}
    all_ospas = []

    # for location in TEST_INDI_LOCATIONS:
    for location in LOCATIONS:
        ospa_list = ospa_for_loc(gt_path, real_pred_folder, box_path, location)
        all_ospas.append(ospa_list)
        ospa_dict[location] = np.mean(ospa_list).item()
        print('ospa: {:.4f} \t {:s}'.format(np.mean(ospa_list), location))

    ospa_dict['overall'] = np.concatenate(all_ospas).mean().item()
    print('Overall:', ospa_dict['overall'])

    if save_results:
        with open(os.path.join(out_dir,"ospa.txt"), 'w') as f: 
            for key, value in ospa_dict.items(): 
                f.write('%s,%s\n' % (key, str(value)))

def compute_ap_pose(pred_path, 
                    box_path, 
                    gt_path, 
                    out_dir=None,
                    save_results=False):
    if out_dir is None:
        out_dir = pred_path

    gt_files = glob.glob(os.path.join(gt_path, "*.json"))
    box_files = glob.glob(os.path.join(box_path, "*.json"))
    assert len(gt_files) == len(box_files)

    real_pred_folder = pred_path
    file_list = glob.glob(real_pred_folder + "/*.json")
    if len(file_list) != len(gt_files):
        pred_folders = glob.glob(pred_path+"/*/")
        for folder in pred_folders:
            file_list = glob.glob(folder + "/*.json")
            if len(file_list) == len(gt_files):
                real_pred_folder = folder
                break

    ap_dict = {}
    all_aps = []

    # for location in TEST_INDI_LOCATIONS:
    for location in LOCATIONS:
        ap_list, _ = average_precision_for_loc(gt_path, real_pred_folder, box_path, location)
        all_aps.append(ap_list)
        ap_dict[location] = np.mean(ap_list).item()
        print('ap: {:.4f} \t {:s}'.format(np.mean(ap_list), location))

    ap_dict['overall'] = np.concatenate(all_aps).mean().item()
    print('Overall:', ap_dict['overall'])

    if save_results:
        with open(os.path.join(out_dir,"ap.txt"), 'w') as f: 
            for key, value in ap_dict.items(): 
                f.write('%s,%s\n' % (key, str(value)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help="The directory containing pose predictions for each scene")
    parser.add_argument('--pose_path', type=str, help="the pose directory for JRDB (e.g. ../labels_2d_pose_stitched)")
    parser.add_argument('--box_path', type=str, help="the box directory for JRDB (e.g. ../labels_2d_stitched)")
    parser.add_argument('--metric', choices=['OSPA', 'AP'])
    args = parser.parse_args()
    global LOCATIONS
    LOCATIONS = TRAIN_LOCATIONS
    if args.metric == 'OSPA':
        compute_ospa_pose(pred_path=args.input_path,
                          box_path=args.box_path,
                          gt_path=args.pose_path)
    else:
        compute_ap_pose(pred_path=args.input_path,
                          box_path=args.box_path,
                          gt_path=args.pose_path)

