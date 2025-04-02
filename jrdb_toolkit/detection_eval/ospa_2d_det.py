import glob
import os
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import sys
def ospa_iou_with_cat(X, Y, cat_X, cat_Y, score_X, allCat, c, p, occlusion_mask = None):
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
    if (0 in X.shape) and (0 in Y.shape):
        print('asd')
        return [0,0,0]
    elif (not (0 in X.shape)) and (0 in Y.shape):
        print('asd1')
        return [c,c,c]
    elif (0 in X.shape) and (not (0 in Y.shape)):
        print('asd2')
        return [c,c,c]
    else:
        pass
    large_number = 1
    if not np.any(np.any(X[2:4,:] > X[0:2,:])):
        print("The coordinates of the second corner of a rectangle should be bigger than the coordinates of the first corner")
        return c

    if not np.any(np.any(Y[2:4,:] > Y[0:2,:])):
        print("The coordinates of the second corner of a rectangle should be bigger than the coordinates of the first corner")
        return c

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

    D = np.reshape(1-V_IoU, [n, m], order="F")
    D = np.minimum(large_number, D)
    term1 = 0
    term2 = 0
    dist_ospa = 0
    countCat = 0

    for ctidx in range(len(allCat)):
        xindc = (cat_X == allCat[ctidx])
        sub_n = np.sum(xindc.astype(int))
        currX = X[:, xindc]
        yindc = (cat_Y == allCat[ctidx])
        sub_m = np.sum(yindc.astype(int))
        currY = Y[:, yindc]
        if (0 in currX.shape) and (0 in currY.shape):
            pass
        elif (0 in currX.shape) and (not (0 in currY.shape)):
            dist_ospa += c
            countCat += 1
        elif (not (0 in currX.shape)) and (0 in currY.shape):
            dist_ospa += c
            countCat += 1
        else:
            D_type_1 = D[xindc,:][:,yindc]
            D_type_1 = np.clip(D_type_1,None,c)
            test = linear_sum_assignment(D_type_1)
            cost = np.sum(D_type_1[test[0],test[1]])
            term1 += (1 / max(sub_m, sub_n)) * c * abs(sub_m - sub_n)
            term2 += (1 / max(sub_m, sub_n)) * cost
            dist_ospa += (1 / max(sub_m, sub_n) * (c * abs(sub_m - sub_n) + cost))
            countCat += 1
    dist_ospa = dist_ospa/countCat
    term1 = term1/countCat
    term2 = term2/countCat
    return [dist_ospa, term1, term2]

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

def compute_2d_det_ospa(test_path, gt_path):
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
        file_list = glob.glob(folder + "image_stitched/*.txt")
        test_file_list.extend(file_list)
        if folder not in test_file_dict:
            test_file_dict[folder.split('/')[-2]]=file_list
        else:
            test_file_dict[folder.split('/')[-2]].append(file_list)
    gt_file_list = sorted(gt_file_list)
    test_file_list = sorted(test_file_list)
    good_indices = [0, 2]
    assert len(gt_file_list) == len(test_file_list)
    count = 0
    ospa_list = []
    ospa_dict = dict()
    for k,v in test_file_dict.items():
        print(k)
        for i in range(len(v)):
            #print(i)
            gt_file_path=gt_file_dict[k][i]
            test_file_path=v[i]
            assert gt_file_path.split('/')[-2:] == [test_file_path.split('/')[-3:][index] for index in good_indices]
            gt_file_df = pd.read_csv(gt_file_path, header = None, sep = ' ')
            test_df = pd.read_csv(test_file_path, header = None, sep = ' ')
            offset=0
            # print(len(test_df.columns))
            if len(test_df.columns)==17:
                offset=1
                #print(test_df.columns)
            test_df = test_df[test_df[15+offset] > 0.5]
            # print(test_df)
            # gt_file_df = gt_file_df[gt_file_df[2].isin([0,1,2,3])]
            #print(asd)
            X = test_df[test_df.columns[4+offset:8+offset]].to_numpy().T
            Y = gt_file_df[gt_file_df.columns[5:9]].to_numpy().T
            cat_X = np.zeros(len(test_df))
            cat_Y = np.zeros(len(gt_file_df))
            score_X = test_df[15+offset].to_numpy()
            ospa=ospa_iou_with_cat(X,Y,cat_X,cat_Y,score_X,[0],1,1)
            if k not in ospa_dict:
                ospa_dict[k]=[ospa]
            else:
                ospa_dict[k].append(ospa)
        #print(ospa_dict[k])
    _sum=np.zeros(3)
    _length=0
    for k,v in ospa_dict.items():
        _sum=_sum+np.array(v).sum(0)
        _length=_length+len(v)
    ospa_dict={k:np.array(v).mean(0) for k,v in ospa_dict.items()}
    ospa_dict['overall']=_sum/_length
    #print(ospa_dict)
    #print(os.path.join(out_dir,"ospa.txt"))
    with open("ospa.txt", 'w') as f: 
        for key, value in ospa_dict.items(): 
            value=list(value)
            value=[str(v) for v in value]
            f.write('%s,%s\n' % (key, ','.join(list(value))))
# usage: python ospa_2d_det.py path/to/pred path/to/gt
if __name__ == "__main__":
    compute_2d_det_ospa(test_path=sys.argv[1],gt_path=sys.argv[2])
