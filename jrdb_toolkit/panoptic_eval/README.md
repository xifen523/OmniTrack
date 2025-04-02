# Evaluation toolkit for JRDB-PanoTrack

This repo contain code for [JRDB dataset and benchmark](https://jrdb.erc.monash.edu/panotrack)

<details open>
<summary><font size="4">
Predictions format
</font></summary>
Note that the predictions must be in the form of a json file with the following format, file names and structure must be the same as the ground truth labels.:

```json
{
  "categories": [
    {
      "id": 1,
      "name": "road"
    },
    {
      "id": 2,
      "name": "terrain"
    },
    ...
    ],
    
    "images": [
        {
            ...
        }
    ],
  "annotations": [
    {
      "image_id": 0,
      "category_id": 1,
      "segmentation": {
        "counts": ... RLE encoded mask ...,
        "size": [height, width]
      },
      "tracking_id": 0 # for tracking evaluation only
    },
    ...
  ]
}
```
</details>

<details open>
<summary><font size="4">
Get OSPA result for panoptic segmentation and tracking.
</font></summary>

#### For Closed-world panoptic segmentation evaluation, run:

```bash
python panoptic_eval/run_jrdb_panoseg.py
    --TRACKERS_FOLDER /path/to/predictions_2d_panoptic_CW
    --GT_FOLDER /path/to/labels_2d_panoptic_CW
    --split val 
    --eval_OW False # False for closed-world evaluation
    --eval_stitched False # False for individual frames, True for stitched frames
    --run_parallel False # False for single process, True for parallel processing
```

#### For Closed-world panotic tracking evaluation, run:

```bash
python panoptic_eval/TrackEval/scripts/run_jrdb_panotrack.py
    --GT_FOLDER /path/to/labels_2d_panoptic_CW
    --TRACKERS_FOLDER /path/to/predictions_2d_panoptic_CW
    --EVAL_OW False # False for closed-world evaluation
    --EVAL_STITCHED False  # False for individual frames, True for stitched frames
    --SPLIT_TO_EVAL "val" 
    --RUN_PARALLEL False # False for single process, True for parallel processing
```
</details>

<details open>
<summary><font size="4">
Get PQ result for panoptic segmentation.
</font></summary>
Check detail information in ./Panoptic_quality folder.
</details>

#### For Open-world panoptic segmentation and tracking evaluation, please submit your results to the JRDB-PanoTrack challenge website at [JRDB-PanoTrack](https://jrdb.erc.monash.edu/panotrack)

The leaderboard will be available and submission will be allowed on 15th April 2024