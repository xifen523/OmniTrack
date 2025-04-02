# JRDB-Act individual action, social grouping and social activity evaluation script.

## Overview

This file describes the evaluation script for three open challenges on JRDB-Act namely individual action, social grouping and social activity detection.
For overview of the dataset and documentation of the data and label format, please refer to our website: https://jrdb.stanford.edu

## Label Format

For each challenge, the evaluation script expects a det.txt and a gt.txt file in the following structure:
All values are separated via spaces and each row corresponds to one idividual action or social group id or social activity label for a box.
For individual action and social activity challenges, each box can have multiple rows in the gt.txt and det.txt files since each box can have multiple action/activity lables.
However, in the social grouping challenge, each box must have exactly one row in the txt files indicating its social group id. 


The 9 columns represent:

```
column    Name      Description
----------------------------------------------------------------------------
   0    sequence_id                 Integer between 0 to 26, indicating the sequence id.
   1    keyframe-id                 Integer, indicating the key-frame id in the specified sequence.
                                    *Evaluation is performed on key-frames which are sampled every one second. [15, 30, 45, ...] 
 [2:6]  bounding-box coordinates    float values of [x1,y1,x2,y2] in the image size.
   6    social group id             Integer, indicating the social group id of the box.
                                    * Must be >0 and boxes within the same social group should have a similar group id.
                                    * An arbitrary value in task_1 and task_4.
   7    individual action id/
        social activity id         Integer, indicating the individual action or social activity id of the box.
                                    * Must be >0.
                                    * An arbitrary value in task_2, task_3.
   8    score(Pred)/Diff(GT)      Float, in gt.txt it indicates the difficulty level of the label which is being evaluated. 
                                  In det.txt, it indicates the confidence score of the predicted label which is being evaluated.
                                  In social grouping challenge, it must be the confidence score of detected bounding boxes.

```
Also, in the annotation files, "cluster_stat" means the difficulty of the grouping, and numbers in "action_label" variable indicate difficulties of the action. 

## Individual action detection

The goal in this challenge is to train a classifier to predict the set of individual action labels
for each detected bounding box in the keyframes of each video sequence. We utilize task-1 to evaluate the performance of the trained model.
The expected text files must be named as det_acion.txt and gt_action.txt.

## Social group detection

The goal in this challenge is to train a model to divide exisitng bounding boxes into different social groups each indicated by a unique id.
We utilize task-2 and task-3 to evaluate the performance of the trained model.
The expected text files must be named as det_group.txt and gt_group.txt.

## Social activity detection

The goal in this challenge is to train a classifier to predict the set of social activity labels for each detected social group in the
keyframes of each video sequence. We utilize task-4 and task-5 to evaluate the performance of the trained model.
The set of social activity labels for each group is the individual actions which are being performed by more than two people in that group.
The expected text files must be named as det_activity.txt and gt_activity.txt.

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

For transparency, we included the evaluation code.

To evaluate individual action detection, run:

```
python JRDB_eval.py -g ./gt_action.txt -d ./det_action.txt -t task_1 --labelmap ./label_map/task_1.pbtxt
```

To evaluate social group detection, run:

```
python JRDB_eval.py -g ./gt_group.txt -d ./det_group.txt -t task_2 --labelmap ./label_map/task_2.pbtxt
```

and 

```
python JRDB_eval.py -g ./gt_group.txt -d ./det_group.txt -t task_3 --labelmap ./label_map/task_3.pbtxt
```

To evaluate social activity detection, run:

```
python JRDB_eval.py -g ./gt_activity.txt -d ./det_activity.txt -t task_4 --labelmap ./label_map/task_4.pbtxt
```

and 

```
python JRDB_eval.py -g ./gt_activity.txt -d ./det_activity.txt -t task_5 --labelmap ./label_map/task_5.pbtxt
```
## Acknowledgement

This code is a fork of AVA action detection evaluation script:
https://github.com/NVlabs/STEP/tree/master/external/ActivityNet/Evaluation

