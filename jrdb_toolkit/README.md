# JRDB Evaluation Toolkit

Toolkit for JRDB dataset

This repo contain code for [JRDB dataset and benchmark](https://jrdb.erc.monash.edu/). For tracking evaluation, our code
is adapted from [TrackEval](https://github.com/JonathonLuiten/TrackEval). With addition code from 3D tracking and OSPA
metric. For detection evaluation, the code is adapted
from [Kitti Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php). For action evaluation, the code is adapted
from [AVA dataset](https://research.google.com/ava/index.html). Please preceed to sub directory for specific
instructions.

The visualisation is based on [KITTI visualisation from Kuixu](https://github.com/kuixu/kitti_object_vis), we did some
modifications (still developing).

## Camera & Lidar coordinate systems

Please note these coordinate system changes before generating submissions  
Coordinates in __LiDAR__: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
&emsp;&emsp;&emsp;&emsp;&emsp; Coordinates in __camera__:

                up z                                                  z front                          
                   ^   x front                                       /                                  
                   |  /                                             /                          
                   | /                                             0 ------> x right                          
    left y <------ 0                                               |                          
                                                                   |      
                                                                   v      
                                                              down y      

The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0), and the yaw is around the z axis, thus the
rotation axis=2.

Groundtruth box's coordinates:  
Box in __LIDAR__ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; Box in __camera__:

        /-------------/|                                            /-------------/|                                       
       /|            / |                                           /|            / |                                       
      /_____________/  |                                          /_____________/  |                                       
     |  |          |   |                                         |  |          |   |                                       
     |  |    z|    |   |                                         |  |          |   |                                       
     |  |     | /x |   |                                         |  |          |   |                                       
     |  |     |/   |   |                                         |  |          |   |                                       
     |  |y<---+    |   |                                         |  |          |   |                                       
     |  |          |   |                                         |  |      /z  |   |                                       
     | /+----------+--/                                          | /+-----/----+--/                                       
     |/            | /                                           |/      +--->x| /                                         
     |_____________|/                                            |_______|_____|/                                          
                                                                         |y

x, y, z, l (x_size), w (y_size), h (z_size) &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;
x, y, z, l (z_size), w (x_size), h (y_size)
