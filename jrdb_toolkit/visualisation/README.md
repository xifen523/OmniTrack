# Visualisation toolkit for JRDB dataset

This repo contains the visualisation script for JRDB dataset:

You can change the setting inside the script and run it (please download
the dataset set on https://jrdb.erc.monash.edu/ and place them in the [proper
directory](https://jrdb.erc.monash.edu/static/downloads/JRDB_sample_structure.zip) first)


## Running the script 
In the __main__ function at the end of the script (line ~1570)

- Change the ``root_dir`` to your JRDB dataset 
- Change the location/sequence names (e.g ``bytes-cafe-2019-02-07_0``)
- Select the visualisation options by set them to ``True``, currently the script supports:
  - **Individual images**: 2D box + pose + social grouping/action + individual action
  - **Stitched images**: 2D/3D box + pose + social grouping/action + individual action
  - **3D pointcloud**: 3D box + social grouping/action + individual action
- If visualising 3D detection, change the ``prediction_dir`` to the KITTI format .txt files
- If save the predictions as videos, set ``save_as_vid`` to True and change the save location, you can also filter low confident score predictions by changing ``score_filter`` 
- If project pointcloud to stitched image, set ``show_velo_points`` to ``True`` and ``color_velo`` to ``False``, otherwise the points will have the same colors as the pixels and you won't see any difference 

Finally, run the script: ```python ./visualise.py```

#### Note
Before saving the visualisation as video (set ``save_as_vid`` to ``True``), we suggest to set ``save_as_vid`` to ``False`` and run the script to inspect, only set to ``True`` once you're happy with the visualisation.

Please let me know if you have any problems running the script

#### Sample setting
stitched image + 2D boxes + projected 3D boxes + social grouping + pose:
```h2
viz_2D                  = True   # True is viz 2D, False if viz 3D
show_individual_image   = False  # Show individual cameras
show_2d_labels          = True   # Show 2D bboxes
show_3d_labels          = True   # show 3d bboxes
show_2d_poses           = True   # show 2d poses
show_social_group       = True   # show social grouping
show_social_action      = False  # show social actions
show_individual_action  = False  # show individual actions
show_velo_points        = False  # Project 3D lidar on to STITCHED IMAGE, Used when visualise 2D stitched image
```
Result:
<p align="center"> <img src='sample.png' align="center"> </p>
