#!/bin/bash

GT_FOLDER="${1:-/path/to/JRDB2022/train_dataset_with_activity/labels/labels_2d_pose_stitched_coco}"
TRACKERS_FOLDER="${2:-/path/to/tracker/jsonfiles}"

python3 scripts/run_posetrack_challenge.py --GT_FOLDER $GT_FOLDER --TRACKERS_FOLDER $TRACKERS_FOLDER \
       --USE_PARALLEL True \
       --NUM_PARALLEL_CORES 8
