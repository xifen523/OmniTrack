# JRDB 2D/3D object detection evaluation script

## Overview

This file describes the JRDB 2D/3D object detection evaluation script.
For overview of the dataset and documentation of the data and label format,
please refer to our website: https://jrdb.stanford.edu

## Label Format

The evaluation script expects a folder in a following structure:

```
cubberly-auditorium-2019-04-22_1/
├── 000000.txt
├── 000001.txt
│      ...
└── 001078.txt
...
tressider-2019-04-26_3/
├── 000000.txt
├── 000001.txt
│      ...
└── 001658.txt
```

Each subfolder represents a sequence and each text file within the subfolder
is a label file of a given frame. The label files contain the following
information. All values (numerical or strings) are separated via spaces, each
row corresponds to one object. The 17 columns represent:

```
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         'Pedestrian' is the only valid type as of now.
   1    truncated    Integer 0 (non-truncated) and 1 (truncated), where
                     truncated refers to the object leaving image boundaries
                     * May be an arbitrary value for evaluation.
   1    occluded     Integer (0, 1, 2, 3) indicating occlusion state:
                     0 = fully visible, 1 = mostly visible
                     2 = severely occluded, 3 = fully occluded
                     * May be an arbitrary value for evaluation.
   1    num_points   Integer, number of points within a 3D bounding box.
                     * May be an arbitrary value for evaluation.
                     * May be a negative value to indicate a 2D bounding box
                       without corresponding 3D bounding box.
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
                     * May be a negative value to indicate a 3D bounding box
                       without corresponding 2D bounding box.
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    conf         Float, indicating confidence in detection, needed for p/r
                     curves, higher is better.
                     * May be an arbitrary value for groundtruth.
```

## 2D Object Detection Benchmark

The goal in the 2D object detection task is to train object detectors for
pedestrian in a 360 panorama image. The object detectors must provide as output
the 2D 0-based bounding box in the image using the format specified above, as
well as a detection score, indicating the confidence in the detection. All
other values must be set to their default values.

In our evaluation, we only evaluate detections on 2D bounding box larger than
500^2 pixel^2 in the image and that are not fully occluded. For evaluation
criterion, inspired by PASCAL, we use 41-point interpolated AP and require the
intersection-over-union of bounding boxes to be larger than 50% for an object
to be detected correctly.

## 3D Object Detection Benchmark

The goal in the 3D object detection task is to train object detectors for
pedestrian in a lidar point clouds. The object detectors must provide the 3D
bounding box (in the format specified above, i.e. 3D dimensions and 3D
locations) and the detection score/confidence. All other values must be set to
their default values.

In our evaluation, we only evaluate detections on 3D bounding box which
encloses more than 10 3D lidar points and lies within 25 meters in bird's eye
view. For evaluation criterion, inspired by PASCAL, we use 41-point
interpolated AP and require the intersection-over-union of bounding boxes to be
larger than 30% for an object to be detected correctly.

## Suggested Validation Split

We provide a suggested validation split to help parameter tune. In the paper,
we show the distribution is very similar to the test dataset. Validation split:

clark-center-2019-02-28_1
gates-ai-lab-2019-02-08_0
huang-2-2019-01-25_0
meyer-green-2019-03-16_0
nvidia-aud-2019-04-18_0
tressider-2019-03-16_1
tressider-2019-04-26_2

## Evaluation Protocol

For transparency, we included the evaluation code. It can be compiled via:

```
g++ -O3 -o evaluate_object evaluate_object.cpp
```

## JRDB -> KITTI data conversion
The script for JRDB format -> KITTI format conversion is also provided, you can run:
```angular2html
python convert_dataset_to_KITTI.py --input_dir=/path/to/JRDB --output_dir=/path/to/converted_JRDB
```
## Acknowledgement

This code is a fork of KITTI 3D Object Detection development kit:
http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
