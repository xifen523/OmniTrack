import glob
import os
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import statistics
import sys

def iou_matrix_3d(objs, hyps, score_x, max_iou=1.):
    """Computes 'intersection over union (IoU)' distance matrix between object and hypothesis rectangles.
    The IoU is computed as
        IoU(a,b) = 1. - isect(a, b) / union(a, b)
    where isect(a,b) is the area of intersection of two rectangles and union(a, b) the area of union. The
    IoU is bounded between zero and one. 0 when the rectangles overlap perfectly and 1 when the overlap is
    zero.
    Params
    ------
    objs : Nx4 array
        Object rectangles (x,y,w,h) in rows
    hyps : Kx4 array
        Hypothesis rectangles (x,y,w,h) in rows
    Kwargs
    ------
    max_iou : float
        Maximum tolerable overlap distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to 0.5
    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """
#     print("we are running this")
    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0,0))
    #print(objs.shape,hyps.shape)
    assert objs.shape[1] == 7
    assert hyps.shape[1] == 7

    C = np.empty((objs.shape[0], hyps.shape[0]))
    for o in range(objs.shape[0]):
        for h in range(hyps.shape[0]):
            base_area = find_area(clip_polygon(objs[o], hyps[h]))
            height = max(objs[o][1], hyps[h][1]) - min(objs[o][1] - objs[o][4], hyps[h][1]-hyps[h][4])
            intersect = base_area*height * score_x[h]
            union = objs[o][3]*objs[o][4]*objs[o][5] + hyps[h][3]*hyps[h][4]*hyps[h][5] - intersect
            if union != 0:
                if (1. - intersect / union) < 0:
                    C[o, h] = max_iou
                else:
                    C[o, h] = 1. - intersect / union
            else:
                C[o, h] = max_iou
    C[C > max_iou] = max_iou
    return C

def find_area(vertices):
    area = 0
    for i in range(len(vertices)):
        area += vertices[i][0]*(vertices[(i+1)%len(vertices)][1] - vertices[i-1][1])
    return 0.5*abs(area)

def compute_intersection_point(pt1, pt2, line1):
    if pt1[0] == pt2[0]:
        slope = np.inf
    else:
        slope = (pt1[1]-pt2[1])/(pt1[0] - pt2[0])
    if np.isinf(slope):
        line2 = (1, 0, pt1[0])
    else:
        line2 = (slope, -1, pt1[0]*slope-pt1[1])
    # print("Line1:", line1)
    # print("Line2:", line2)
    if line1[1] == 0:
        x = line1[2]/line1[0]
        y = (line2[2] - line2[0]*x)/line2[1]
    elif line1[0] == 0:
        y = line1[2]/line1[1]
        x = (line2[2] - line2[1]*y)/line2[0]
    elif line2[1] == 0:
        x = pt1[0]
        y = (line1[2]-x*line1[0])/line1[1]
    else:
        tmp_line = (line2 - line1*(line2[1]/line1[1]))
        x = tmp_line[2]/tmp_line[0]
        y = (line2[2] - line2[0]*x)/line2[1]
    return (x,y)

def point_inside_edge(pt, edge):
    lhs = pt[0]*edge[0] + pt[1]*edge[1]
    if lhs < edge[2] - 1e-6:
        return True
    else:
        return False

def get_angle(p):
    x, y = p
    angle = np.arctan2(y,x)
    if angle < 0:
        angle += np.pi*2
    return angle

def sort_points(pts, center):
    x, z = center
    sorted_pts = sorted([(i, (v[0]-x, v[1]-z)) for i,v in enumerate(pts)], key = lambda p: get_angle((p[1][0],p[1][1])))
    idx, _ = zip(*sorted_pts)
    return [pts[i] for i in idx]

def clip_polygon(box1, box2):
    #clips box 1 by the edges in box2
    x,y,z,l,h,w,theta = box2
    theta = -theta

    box2_edges = np.asarray([(-np.cos(theta), -np.sin(theta), l/2-x*np.cos(theta)-z*np.sin(theta)),
                    (-np.sin(theta), np.cos(theta), w/2-x*np.sin(theta)+z*np.cos(theta)),
                    (np.cos(theta), np.sin(theta), l/2+x*np.cos(theta)+z*np.sin(theta)),
                    (np.sin(theta), -np.cos(theta), w/2+x*np.sin(theta)-z*np.cos(theta))])
    x,y,z,l,h,w,theta = box1
    theta = -theta

    box1_vertices = [(x+l/2*np.cos(theta)-w/2*np.sin(theta), z+l/2*np.sin(theta)+w/2*np.cos(theta)),
                        (x+l/2*np.cos(theta)+w/2*np.sin(theta), z+l/2*np.sin(theta)-w/2*np.cos(theta)),
                        (x-l/2*np.cos(theta)-w/2*np.sin(theta), z-l/2*np.sin(theta)+w/2*np.cos(theta)),
                        (x-l/2*np.cos(theta)+w/2*np.sin(theta), z-l/2*np.sin(theta)-w/2*np.cos(theta))]
    out_vertices = sort_points(box1_vertices, (x, z))
    for edge in box2_edges:
        vertex_list = out_vertices.copy()
        out_vertices = []
        for idx, current_vertex in enumerate(vertex_list):
            previous_vertex = vertex_list[idx-1]
            if point_inside_edge(current_vertex, edge):
                if not point_inside_edge(previous_vertex, edge):
                    out_vertices.append(compute_intersection_point(previous_vertex, current_vertex, edge))
                out_vertices.append(current_vertex)
            elif point_inside_edge(previous_vertex, edge):
                out_vertices.append(compute_intersection_point(previous_vertex, current_vertex, edge))
    to_remove = []
    for i in range(len(out_vertices)):
        if i in to_remove:
            continue
        for j in range(i+1, len(out_vertices)):
            if abs(out_vertices[i][0] - out_vertices[j][0]) < 1e-6 and abs(out_vertices[i][1] - out_vertices[j][1]) < 1e-6:
                to_remove.append(j)
    out_vertices = sorted([(v[0]-x, v[1]-z) for i,v in enumerate(out_vertices) if i not in to_remove], key = lambda p: get_angle((p[0],p[1])))
    return out_vertices



def ospa_iou_with_cat_3d(X, Y, cat_X, cat_Y, score_X, allCat, c, p, verbose=False):
    #print(X,Y,X.shape,Y.shape)
    '''
    Inputs:
    X (prediction), Y (truth) - matrices of column vector representing the
    rectangles with two corners' representation, e.g. X=(x1,y1,x2,y2), x2>x1,
    y2>y1
    cat_X - categories of X
    cat_Y - categories of Y
    score_X - categories of X (note scores of Y are implicitly ones)
    c - cut-off cost (for OSPA)
    p - norm order (for OSPA, Wasserstein)

    Outputs:
    dist_opa: OSPA distance
    '''

#     if (0 in X.shape) and (0 in Y.shape):
#         return 0
#     elif (not (0 in X.shape)) and (0 in Y.shape):
#         return c
#     elif (0 in X.shape) and (not (0 in Y.shape)):
#         return c
#     else:
#         pass
    large_number = 1
#     if not np.any(np.any(X[2:4,:] > X[0:2,:])):
#         print("The coordinates of the second corner of a rectangle should be bigger than the coordinates of the first corner")
#         return c

#     if not np.any(np.any(Y[2:4,:] > Y[0:2,:])):
#         print("The coordinates of the second corner of a rectangle should be bigger than the coordinates of the first corner")
#         return c

#     print("We hit here")  

    #print(X.shape,Y.shape)
    D = iou_matrix_3d(Y, X, score_X).T
    D = np.minimum(large_number, D)
#     print(D.shape)
    dist_ospa = 0
    countCat = 0

    for ctidx in range(len(allCat)):
        xindc = (cat_X == allCat[ctidx])
        sub_n = np.sum(xindc.astype(int))
#         currX = X[:, xindc]
        yindc = (cat_Y == allCat[ctidx])
        sub_m = np.sum(yindc.astype(int))
#         currY = Y[:, yindc]
        if len(xindc) == 0 and len(yindc) == 0:
            pass
        elif len(xindc) == 0 and len(yindc) != 0:
            dist_ospa += c
            countCat += 1
            term1 = 0
            term2 = 0
        elif len(xindc) != 0 and len(yindc) == 0:
            dist_ospa += c
            countCat += 1
            term1 = 0
            term2 = 0
        else:
            # print(D.shape,xindc.shape,yindc.shape)
            D_type_1 = D[xindc,:][:,yindc]
            D_type_1 = np.clip(D_type_1,None,c)
            test = linear_sum_assignment(D_type_1)
            cost = np.sum(D_type_1[test[0],test[1]])
            if cost<0:
                print(D)
            if verbose:
                print(cost)
#             dist_ospa += (1 / min(sub_m, sub_n) * (cost))
            term1 = 1 / max(sub_m, sub_n) * (c * abs(sub_m - sub_n))
            term2 = 1 / max(sub_m, sub_n) * cost
            dist_ospa += (1 / max(sub_m, sub_n) * (c * abs(sub_m - sub_n) + cost))
            countCat += 1
#     print(dist_ospa)
    if dist_ospa == 0:
        return 1
    return dist_ospa,term1,term2

OUTDOOR_DIRS = ('discovery-walk-2019-02-28_0', 'meyer-green-2019-03-16_1',
                'discovery-walk-2019-02-28_1', 'food-trucks-2019-02-12_0',
                'quarry-road-2019-02-28_0', 'outdoor-coupa-cafe-2019-02-06_0',
                'serra-street-2019-01-30_0', 'gates-to-clark-2019-02-28_0',
                'tressider-2019-03-16_2',
                'lomita-serra-intersection-2019-01-30_0',
                'huang-intersection-2019-01-22_0')

INDOOR_DIRS = ('cubberly-auditorium-2019-04-22_1', 'gates-ai-lab-2019-04-17_0',
               'gates-basement-elevators-2019-01-17_0',
               'nvidia-aud-2019-01-25_0', 'nvidia-aud-2019-04-18_1',
               'nvidia-aud-2019-04-18_2', 'gates-foyer-2019-01-17_0',
               'stlc-111-2019-04-19_1', 'stlc-111-2019-04-19_2',
               'hewlett-class-2019-01-23_0', 'hewlett-class-2019-01-23_1',
               'huang-2-2019-01-25_1', 'indoor-coupa-cafe-2019-02-06_0',
               'tressider-2019-04-26_0', 'tressider-2019-04-26_1',
               'tressider-2019-04-26_3')

def label_distance(x,z):
    return (float(x)**2 + float(z)**2)**0.5

def compute_3d_det_ospa(test_path, gt_path):
    gt_folders = glob.glob(gt_path+"/*/")
    gt_file_list = []
    gt_file_dict=dict()
    for folder in gt_folders:
        file_list = glob.glob(folder + "/*.txt")
        gt_file_list.extend(file_list)
        if folder not in gt_file_dict:
            gt_file_dict[folder.split('/')[-2]]=file_list
        else:
            gt_file_dict[folder.split('/')[-2]].append(file_list)

    test_folders = glob.glob(test_path+"/*/")
    test_file_list = []
    test_file_dict=dict()
    for folder in test_folders:
        file_list = glob.glob(folder + "*.txt")
        test_file_list.extend(file_list)
        if folder not in test_file_dict:
            test_file_dict[folder.split('/')[-2]]=file_list
        else:
            test_file_dict[folder.split('/')[-2]].append(file_list)
    gt_file_list = sorted(gt_file_list)
    test_file_list = sorted(test_file_list)
    good_indices = [0, 2]
    print(len(gt_file_list),len(test_file_list))
    assert len(gt_file_list) == len(test_file_list)
    count = 0
    ospa_list = []
    ospa_dict = dict()
    for k,v in test_file_dict.items():
        print(k)
        for i in range(len(v)):
            if i%100==0:
                print(i)
            gt_file_path=gt_file_dict[k][i]
            test_file_path=v[i]
            #print(gt_file_path,test_file_path)
            #assert gt_file_path.split('/')[-2:] == [test_file_path.split('/')[-3:][index] for index in good_indices]
            # gt_file_df = pd.read_csv(gt_file_path, header = None, sep = ' ')
            # test_df = pd.read_csv(test_file_path, header = None, sep = ' ')
            gt_file_df = pd.read_csv(gt_file_path, header = None, sep = ' ').values
            test_df = pd.read_csv(test_file_path, header = None, sep = ' ').values
            distance_test=(test_df[:,8]**2+test_df[:,10]**2)**0.5
            test_df=test_df[distance_test<5]
            distance_gt=(gt_file_df[:,9]**2+gt_file_df[:,11]**2)**0.5
            gt_file_df=gt_file_df[distance_gt<5]

            # gt_file_df['distance'] = gt_file_df.apply(lambda row: label_distance(row[9],row[11]), axis=1)
            # test_df['distance'] = test_df.apply(lambda row: label_distance(row[8],row[10]), axis=1)
            # gt_file_df = gt_file_df[gt_file_df['distance'] < 5]
            # test_df = test_df[test_df['distance'] < 5]
            X = test_df[:,8:15]
            Y = gt_file_df[:,9:16]
            #print(X.shape,Y.shape)
            cat_X = np.zeros(len(test_df))
            cat_Y = np.zeros(len(gt_file_df))
            #score_X = test_df[15].to_numpy()
            score_X = test_df[:,15]
            if k not in ospa_dict:
                ospa_dict[k]=[np.array(ospa_iou_with_cat_3d(X,Y,cat_X,cat_Y,score_X,[0],1,1))]
            else:
                ospa_dict[k].append(np.array(ospa_iou_with_cat_3d(X,Y,cat_X,cat_Y,score_X,[0],1,1)))
    ospa_dict={k:np.array(v).mean(axis=0) for k,v in ospa_dict.items()}
    _sum=np.zeros(3)
    for k,v in ospa_dict.items():
        _sum=_sum+v
    ospa_dict['overall']=_sum/len(ospa_dict.keys())
    with open("ospa.txt", 'w') as f: 
        for key, value in ospa_dict.items(): 
            value=list(value)
            value=[str(v) for v in value]
            f.write('%s,%s\n' % (key, ','.join(list(value))))
            
# usage: python ospa_3d_det.py path/to/pred path/to/gt
if __name__ == "__main__":
    compute_3d_det_ospa(test_path=sys.argv[1],gt_path=sys.argv[2])
