import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES



# @PIPELINES.register_module()
# class LoadMultiViewImageFromFiles(object):
#     """Load multi channel images from a list of separate channel files.

#     Expects results['img_filename'] to be a list of filenames.

#     Args:
#         to_float32 (bool, optional): Whether to convert the img to float32.
#             Defaults to False.
#         color_type (str, optional): Color type of the file.
#             Defaults to 'unchanged'.
#     """

#     def __init__(self, to_float32=False, color_type="unchanged"):
#         self.to_float32 = to_float32
#         self.color_type = color_type

#     def __call__(self, results):
#         """Call function to load multi-view image from files.

#         Args:
#             results (dict): Result dict containing multi-view image filenames.

#         Returns:
#             dict: The result dict containing the multi-view image data.
#                 Added keys and values are described below.

#                 - filename (str): Multi-view image filenames.
#                 - img (np.ndarray): Multi-view image arrays.
#                 - img_shape (tuple[int]): Shape of multi-view image arrays.
#                 - ori_shape (tuple[int]): Shape of original image arrays.
#                 - pad_shape (tuple[int]): Shape of padded image arrays.
#                 - scale_factor (float): Scale factor.
#                 - img_norm_cfg (dict): Normalization configuration of images.
#         """
#         filename = results["img_filename"]
#         # img is of shape (h, w, c, num_views)
#         img = np.stack(
#             [mmcv.imread(name, self.color_type) for name in filename], axis=-1
#         )
#         if self.to_float32:
#             img = img.astype(np.float32)
#         results["filename"] = filename
#         # unravel to list, see `DefaultFormatBundle` in formatting.py
#         # which will transpose each image separately and then stack into array
#         results["img"] = [img[..., i] for i in range(img.shape[-1])]
#         results["img_shape"] = img.shape
#         results["ori_shape"] = img.shape
#         # Set initial values for default meta_keys
#         results["pad_shape"] = img.shape
#         results["scale_factor"] = 1.0
#         num_channels = 1 if len(img.shape) < 3 else img.shape[2]
#         results["img_norm_cfg"] = dict(
#             mean=np.zeros(num_channels, dtype=np.float32),
#             std=np.ones(num_channels, dtype=np.float32),
#             to_rgb=False,
#         )
#         return results

#     def __repr__(self):
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f"(to_float32={self.to_float32}, "
#         repr_str += f"color_type='{self.color_type}')"
#         return repr_str


@PIPELINES.register_module()
class JRDB_LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        use_dim=[0, 1, 2],

    ):

        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))

        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.use_dim = use_dim


    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # first try to use open3d to read point clouds data
        # try:
        #     import open3d as o3d
        #     pcd = o3d.io.read_point_cloud(pts_filename).points
        #     points = np.asarray(pcd)

        #     return points
        # except ImportError:
        #     raise ImportError("Please install open3d to read point clouds data.")
        
        try:
            from pypcd import pypcd
            pcd = pypcd.PointCloud.from_path(pts_filename)
            point_cloud = np.zeros((pcd.points, len(pcd.fields)), dtype=np.float32)

            for i, field in enumerate(pcd.fields):
                point_cloud[:, i] = np.transpose(pcd.pc_data[field])

            return point_cloud[:,self.use_dim]
        except ImportError:
            raise ImportError("Please install pypcd to read point clouds data. Please refer to https://github.com/dimatura/pypcd/issues/28")


    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results["pts_filename"]
        lower_pcd = self._load_points(pts_filename[0])
        upper_pcd = self._load_points(pts_filename[1])

        # homogenize the points
        lower_pcd = np.hstack((lower_pcd, np.ones((lower_pcd.shape[0], 1))))
        upper_pcd = np.hstack((upper_pcd, np.ones((upper_pcd.shape[0], 1))))

        lower2upper_trans = results['lower2upper']
        lower_pcd = np.dot(lower2upper_trans[0], lower_pcd.T)
        points = np.hstack([upper_pcd.T, lower_pcd]).T

        points = points[:, self.use_dim]

        results["points"] = points
        return results
