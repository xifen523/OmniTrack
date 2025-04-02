# JRDB-Pose Evaluation Toolkit

This toolkit evaluation JRDB-Pose with the AP and OSPA metrics.

For example, to evaluate train set predictions with OSPA, run
```
python pose_eval.py \
  --input_path path/to/predictions \
  --pose_path path/to/train_dataset_with_activity/labels/labels_2d_pose_stitched \
  --box_path path/to/train_dataset_with_activity/labels/labels_2d_stitched \
  --metric OSPA
```
