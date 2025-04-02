# This folder provides the necessary scripts for Panoptic Quality result of JRDB PanoTrack
## To get the PQ result.
Users must first obtain the complete test set of panoptic segmentation predictions in JSON format. After setting the paths in calculate_pq_for_all.sh for ground truth, the panoptic segmentation prediction path, and the jrdb_categories_cocoformat, users can obtain the PQ results. Please note that the test set ground truth is not released; this script automatically runs on our server when users submit their prediction files.
```
bash calculated_pq_for_all.sh

```
###### 
In calculated_pq_for_all.sh, users can add the OW option for evaluation.py and generate_all_result.py to obtain Open World PQ results. Remove the OW option to obtain Close World results. For example:
```
python evaluation.py \
    --gt_json_file "$gt_json_file" \
    --pred_json_file "$output_json_file" \
    --pq_saving_path "$SAVE_PREDICTION_DIR" \
    #--OW ###If open world

python generate_all_result.py \
    --pred_dir "$SAVE_PREDICTION_DIR" \
    --CSV \
    #--OW ###If open world
```

## For submitting the results on the JRDB leaderboard, we require stitched segmentation result.


Users can develop their own strategies to stitch individual view outputs into a stitched output if their prediction output is from a single view.

The JRDB team provides a simple stitching method for users to reference.

The core idea is to first map all segments from each individual image to a stitched map and then reindex these segments, since the original index of each image is based on a single view. 

During reindexing, we merge all the stuff segments that belong to the same stuff class. For thing segments, when we map each segment, we first search their nearby segments on the stitched view, if there exists any segments belong to same category label but different cameras, we consider there two segments are belong to same instance. For searching nearby segments, we dilating the segment binary mask and search whether any other segments have intersection area with current segment. (the dilation structure is set to be (5x5) square ones.) User can also modifies the dilation but modifies the function calculate_intersections in generate_stitched_prediction_v2.py

```
bash generating_stitch_pngs.sh
```


## Required Installation
[coco panopticapi](https://github.com/cocodataset/panopticapi)