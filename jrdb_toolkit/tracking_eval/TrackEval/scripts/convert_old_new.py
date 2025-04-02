from pandas.core import frame
import argparse
import collections
import glob
import json
import numpy as np
import os
train = [
    'bytes-cafe-2019-02-07_0',
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
    'huang-lane-2019-02-12_0',
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
test = [
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
BASE_PATH = "/pvol2/jrdb_dev/jrdb_website_dev/media/submissions/u6361796@anu.edu.au/3dt/2021-10-16 08:05:29+00:00_2020-10-04 12_50_37_jrdb-jrmot2/"
INPUT_2D_LABELS_PATH = BASE_PATH
TEST_2D_PATH='/pvol2/jrdb_dev/jrdb_website_dev/backend/groundtruths/MOT/test_sequences/'
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output_kitti_dir",
        default="KITTI_track",
        help="location of the output KITTI-like labels",
    )
    ap.add_argument(
        "-i",
        "--input_jrdb_dir",
        default="test_dataset/labels",
        help="location of the input jrdb labels",
    )
    return ap.parse_args()
def get_labels_submission(index,path):
    file=open(path,'r')
    seq_name=path.split('/')[-1].split('.text')[0]
    lines=file.readlines()
    new_lines=[]
    output_dir=args.output_kitti_dir
    for l in lines:
        l=l.split(',')
        frame=l[0]
        id=l[1]
        truncated=0
        occlusion=0
        conf=l[-1]
        alpha=-1
        if '/2dt/' in path:
            x1_2d,y1_2d,x2_2d,y2_2d=l[2:6]
        else:
            x1_2d,y1_2d,x2_2d,y2_2d=-1,-1,-1,-1
        if '/3dt/' in path:
            height_3d,width_3d,length_3d,centerx_3d,centery_3d,centerz_3d=l[6:12]
            rotation_y=l[12]
        else:
            height_3d,width_3d,length_3d,centerx_3d,centery_3d,centerz_3d=-1,-1,-1,-1,-1,-1
            rotation_y=-1
        line = (
                    "%s %s Pedestrian %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s"
                    % (
                        frame,
                        id,
                        truncated,
                        occlusion,
                        alpha,
                        x1_2d,
                        y1_2d,
                        x2_2d,
                        y2_2d,
                        height_3d,
                        width_3d,
                        length_3d,
                        centerx_3d,
                        centery_3d,
                        centerz_3d,
                        rotation_y,
                        conf,
                    )
        )
        new_lines.append(line)
    # seq_dir = '/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/TrackEval/data/trackers/jrdb/jrdb_2d_box_train/CIWT/data/'
    seq_dir = '/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/TrackEval/data/trackers/jrdb/jrdb_3d_box_train/CIWT/data/'
    # os.makedirs(seq_dir, exist_ok=True)
    # with open(os.path.join(seq_dir, str(index).rjust(4, '0')+ ".txt"), "w") as f:
    #         f.writelines(new_lines)
def get_labels_test(index,path):
    file=open(path,'r')
    file_3d=open(path.replace('/gt/gt.txt','/gt/3d_gt.txt'),'r')
    seq_name=path.split('/')[-1].split('.text')[0]
    lines=file.readlines()
    lines_3d=file_3d.readlines()
    #print(len(lines),len(lines_3d))
    new_lines=[]
    output_dir=args.output_kitti_dir
    for l in lines:
        l=l.split(',')
        frame=l[0]
        id=l[1]
        truncated=0
        occlusion=0
        conf=l[-1]
        alpha=-1
        if '/gt/gt.txt' in path:
            x1_2d,y1_2d,x2_2d,y2_2d=l[2:6]
        else:
            x1_2d,y1_2d,x2_2d,y2_2d=-1,-1,-1,-1
        if '/gt/3d_gt.txt' in path:
            height_3d,width_3d,length_3d,centerx_3d,centery_3d,centerz_3d=l[6:12]
            rotation_y=l[12]
        else:
            height_3d,width_3d,length_3d,centerx_3d,centery_3d,centerz_3d=-1,-1,-1,-1,-1,-1
            rotation_y=-1
        line = (
                    "%s %s Pedestrian %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n"
                    % (
                        frame,
                        id,
                        truncated,
                        occlusion,
                        alpha,
                        x1_2d,
                        y1_2d,
                        x2_2d,
                        y2_2d,
                        height_3d,
                        width_3d,
                        length_3d,
                        centerx_3d,
                        centery_3d,
                        centerz_3d,
                        rotation_y,
                    )
        )
        new_lines.append(line)
    print(str(index).rjust(4, '0'),'empty','000000',str(int(frame)+1).rjust(6, '0'))
    #seq_dir = '/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/TrackEval/data/gt/jrdb/jrdb_2d_box_train/label02/'
    seq_dir = '/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/TrackEval/data/gt/jrdb/jrdb_3d_box_train/label02/'
    # os.makedirs(seq_dir, exist_ok=True)
    # with open(os.path.join(seq_dir, str(index).rjust(4, '0')+ ".txt"), "w") as f:
    #         f.writelines(new_lines)
if __name__ == "__main__":
    args = parse_args()
    # for i,l in enumerate(test):
    #     #labels = get_labels_submission(i,INPUT_2D_LABELS_PATH+l+'_image_stitched.txt')
    #     labels = get_labels_submission(i,INPUT_2D_LABELS_PATH+l+'.txt')
    for i,l in enumerate(test):
        print(i,l)
        #labels = get_labels_test(i,TEST_2D_PATH+l+'/gt/gt.txt')
        labels = get_labels_test(i,TEST_2D_PATH+l+'/gt/3d_gt.txt')