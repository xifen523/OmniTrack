import copy
import itertools
import json
import os
import time
from os import listdir
from os.path import isfile, join
# import fire
# import shutil
# import glob
from pprint import PrettyPrinter

import colorcet as cc
import cv2
import numpy as np
import open3d as o3d
import seaborn as sns
import torch
import yaml
from screeninfo import get_monitors
# import matplotlib.pyplot as plt
# from scipy import stats
# from scipy import ndimage
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
from tqdm import tqdm

pp = PrettyPrinter(indent=2, compact=True)

background_color = list(np.array([200, 200, 200]) / 255.0)

social_act_list = {
    "impossible": 0,
    "walking": 1,
    "standing": 2,
    "holding sth": 3,
    "sitting": 4,
    "listening to someone": 5,
    "talking to someone": 6,
    "looking at robot": 7,
    "looking into sth": 8,
    "looking at sth": 9,
    "cycling": 10,
    "going upstairs": 11,
    "bending": 12,
    "typing": 13,
    "interaction with door": 14,
    "eating sth": 15,
    "talking on the phone": 16,
    "pointing at sth": 17,
    "going downstairs": 18,
    "reading": 19,
    "pushing": 20,
    "skating": 21,
    "scootering": 22,
    "greeting gestures": 23,
    "running": 24,
    "opening the door": 14,
    "pushing the door": 14,
    "holding the door": 14,
    "writing": 25,
    "pulling": 26,
    "lying": 27,
}

act_num2text = {
    0: "impossible",
    1: "walking",
    2: "standing",
    3: "holding sth",
    4: "sitting",
    5: "listening to someone",
    6: "talking to someone",
    7: "looking at robot",
    8: "looking into sth",
    9: "looking at sth",
    10: "cycling",
    11: "going upstairs",
    12: "bending",
    13: "typing",
    14: "interaction with door",
    15: "eating sth",
    16: "talking on the phone",
    17: "pointing at sth",
    18: "going downstairs",
    19: "reading",
    20: "pushing",
    21: "skating",
    22: "scootering",
    23: "greeting gestures",
    24: "running",
    25: "writing",
    26: "pulling",
    27: "lying",
}

TRAIN = [
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
TEST = [
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
joint_map = ['head',
             'right eye',
             'left eye',
             'right shoulder',
             'neck',
             'left shoulder',
             'right elbow',
             'left elbow',
             'tailbone',
             'right hand',
             'right hip',
             'left hip',
             'left hand',
             'right knee',
             'left knee',
             'right foot',
             'left foot']
joint_color = {
    ('left eye', 'right eye'): (255, 0, 0),  # 'blue'
    ('head', 'neck'): (0, 255, 0),  # 'lime'
    ('neck', 'left shoulder'): (128, 0, 0),  # 'navy'
    ('neck', 'right shoulder'): (128, 0, 0),  # 'navy'
    ('neck', 'tailbone'): (0, 128, 128),  # 'olive'
    ('left shoulder', 'left elbow'): (128, 0, 128),  # 'purple'
    ('left elbow', 'left hand'): (255, 0, 255),  # 'fuchsia'
    ('right shoulder', 'right elbow'): (255, 255, 0),  # 'cyan'
    ('right elbow', 'right hand'): (128, 128, 0),  # 'teal'
    ('tailbone', 'right hip'): (0, 255, 0),  # 'lime'
    ('tailbone', 'left hip'): (0, 255, 0),  # 'lime'
    ('right hip', 'right knee'): (0, 165, 255),  # 'orange'
    ('right knee', 'right foot'): (0, 0, 255),  # 'red'
    ('left hip', 'left knee'): (0, 255, 255),  # 'yellow'
    ('left knee', 'left foot'): (0, 255, 0),  # 'lime'
}


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_timestamps(timestamps, begin=0, end=None):
    if end is None:
        end = len(timestamps)
    else:
        assert end < len(timestamps)
    timestamps = [x[:-4] for x in timestamps]
    return timestamps[begin:end + 1]


def rotz(t, kitti_format=False):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])
    # if kitti_format:
    #     return np.array([[c, s, 0],
    #                      [-s, c, 0],
    #                      [0, 0, 1]])
    # else:
    #     return np.array([[c, -s, 0],
    #                      [s, c, 0],
    #                      [0, 0, 1]])


empty_label = {
    "box": {
        "cx": None,
        "cy": None,
        "cz": None,
        "h": None,
        "l": None,
        "rot_z": None,
        "w": None
    },
}


def box_camera_to_lidar_jrdb(bboxes):  # kitti to jrdb
    """Convert lidar boxes to camera (ref) boxes

    x = z'
    y = -x'
    # z = -y' + h' / 2
    z = -y'
    l = l'
    w = w'
    h = h'
    theta = -theta'

    Args:
        kitti_boxes (array[B, 7]): x', y', z', h', w', l', theta'
        kitti_boxes (array[B, 7]): x', y', z', l', h', w', theta' KITTI cam

    Returns:
        jrdb_boxes (array[B, 7]): x, y, z, l, w, h, theta decline
        jrdb_boxes (array[B, 7]): x, y, z, w, l, h, theta JRDB/KITTI lidar cordinate
    kitti -> jrdb
    x_cam = -y'_lidar
    y_cam = -z'_lidar
    z_cam = x'_lidar
    l_cam = w'_lidar
    h_cam = h'_lidar
    w_cam = l'_lidar
    theta = -theta'
    """
    x, y, z = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2],
    l, h, w = bboxes[:, 3], bboxes[:, 4], bboxes[:, 5],
    r = bboxes[:, 6]
    jrdb_boxes = np.stack(
        [
            z, -x, -y,  # equal to xyz lidar
            # w, l, h,  # equal to lwh lidar
            l, w, h,  # equal to lwh lidar
            -r
        ], axis=1)
    return jrdb_boxes


def change_box3d_center_(box3d, src, dst):
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)


# def convert_rot_to_2pi(rot):
#     if rot > 0:
#         return rot
#     else:
#         return rot + 2 * np.pi
def convert_txts_to_dict(prediction_path):
    labels = {}
    txt_paths = [f for f in listdir(prediction_path) if isfile(join(prediction_path, f))]

    for txt_path in txt_paths:
        with open(f"{prediction_path}/{txt_path}", 'r') as f:
            lines = f.readlines()
        content = [line.strip().split(' ') for line in lines]
        dims = np.array(
            [[float(info) for info in x[9:12]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]  # hwl -> lhw
        locs = np.array(
            [[float(info) for info in x[12:15]] for x in content]).reshape(-1, 3)
        rots = np.array(
            [float(x[15]) for x in content]).reshape(-1)
        scores = np.array([float(x[16]) for x in content])
        pred_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        pred_boxes = box_camera_to_lidar_jrdb(pred_boxes)
        change_box3d_center_(pred_boxes, [0.5, 0.5, 0],
                             [0.5, 0.5, 0.5])
        labels[f"{txt_path[:-4]}.pcd"] = []
        for box, score in zip(pred_boxes, scores):
            labels[f"{txt_path[:-4]}.pcd"].append({
                "box": {
                    "cx": box[0],
                    "cy": box[1],
                    "cz": box[2],
                    "l": box[3],
                    "w": box[4],
                    "h": box[5],
                    "rot_z": box[6],
                },
                "score": score
            })
    return labels


class Visualizer(object):
    def __init__(self, calib_folder, label_3d_path, label_2d_path,
                 area=None, kitti_submission=False,
                 label_2d_pose_path=None):
        global_config = os.path.join(calib_folder, 'defaults.yaml')
        camera_config = os.path.join(calib_folder, 'cameras.yaml')
        lidar_config = os.path.join(calib_folder, 'lidars.yaml')
        with open(global_config) as f:
            self.global_config_dict = yaml.safe_load(f)

        with open(camera_config) as f:
            self.camera_config_dict = yaml.safe_load(f)

        # Load calibration data
        with open(lidar_config, 'r') as f:
            self.lidar_calib = yaml.safe_load(f)
        self.cam_ids = [0, 2, 4, 6, 8]
        self.show_individual_image = False
        self.area = area
        self.polygon = None
        self.max_heights_by_ID = {}
        self.median_focal_length_y = self.calculate_median_param_value(param='f_y')
        self.median_optical_center_y = self.calculate_median_param_value(param='t_y')
        # image shape is (color channels, height, width)
        self.img_shape = 3, self.global_config_dict['image']['height'], self.global_config_dict['image']['width']
        self.kitti_submission = kitti_submission

        if kitti_submission:
            self.ped_ids, self.ped_colors = None, None
            self.labels_2d = None
            self.labels_3d = convert_txts_to_dict(label_3d_path)
        else:
            with open(label_3d_path, 'r') as f:
                self.labels_3d = json.load(f)
                self.labels_3d = self.labels_3d['labels']
            if not isinstance(label_2d_path, list):
                with open(label_2d_path, 'r') as f:
                    self.labels_2d = json.load(f)
                    self.labels_2d = self.labels_2d['labels']
                self.ped_ids, self.ped_colors = [], {}
                for ts in self.labels_3d.values():
                    for ped in ts:
                        self.ped_ids.append(int(ped['label_id'][ped['label_id'].find(':') + 1:]))
                for ts in self.labels_2d.values():
                    for ped in ts:
                        self.ped_ids.append(int(ped['label_id'][ped['label_id'].find(':') + 1:]))

                self.ped_ids = set(self.ped_ids)

                cluster_ids = [p['social_group']['cluster_ID'] for ts
                               in self.labels_2d.values() for p in ts if 'social_group' in p.keys()]

            else:
                self.labels_2d = {}
                self.ped_ids, self.ped_colors, cluster_ids = [], {}, []

                for ts in self.labels_3d.values():
                    for ped in ts:
                        self.ped_ids.append(int(ped['label_id'][ped['label_id'].find(':') + 1:]))

                for cam_id, path in zip(self.cam_ids, label_2d_path):
                    with open(path, 'r') as f:
                        labels_2d = json.load(f)
                        labels_2d = labels_2d['labels']
                    self.labels_2d[cam_id] = labels_2d
                    for ts in labels_2d.values():
                        for ped in ts:
                            self.ped_ids.append(int(ped['label_id'][ped['label_id'].find(':') + 1:]))
                    cluster_ids += [p['social_group']['cluster_ID'] for ts
                                    in labels_2d.values() for p in ts if 'social_group' in p.keys()]
                self.ped_ids = set(self.ped_ids)

            if len(cluster_ids) > 0:
                self.num_clusters = max(cluster_ids)
                self.cluster_colors = np.asarray(sns.color_palette(cc.glasbey_dark,
                                                                   n_colors=self.num_clusters))  # RGB range (0,1)

            for ped_id, color in zip(self.ped_ids,
                                     np.asarray(sns.color_palette(cc.glasbey_dark, n_colors=len(self.ped_ids)))):
                self.ped_colors[ped_id] = color  # RGB range (0,1)

            if not isinstance(label_2d_pose_path, list):
                if label_2d_pose_path is not None:
                    with open(label_2d_pose_path, 'r') as f:
                        self.labels_2d_pose = json.load(f)
            else:
                self.labels_2d_pose = {}
                for cam_id, path in zip(self.cam_ids, label_2d_pose_path):
                    with open(path, 'r') as f:
                        labels_2d_pose = json.load(f)
                    self.labels_2d_pose[cam_id] = labels_2d_pose

            self.social_act_colors = np.asarray(sns.color_palette(cc.glasbey_light, n_colors=len(social_act_list)))

    def calculate_median_param_value(self, param):
        if param == 'f_y':
            idx = 4
        elif param == 'f_x':
            idx = 0
        elif param == 't_y':
            idx = 5
        elif param == 't_x':
            idx = 2
        elif param == 's':
            idx = 1
        else:
            raise 'Wrong parameter!'

        omni_camera = ['sensor_0', 'sensor_2', 'sensor_4', 'sensor_6', 'sensor_8']
        parameter_list = []
        for sensor, camera_params in self.camera_config_dict['cameras'].items():
            if sensor not in omni_camera:
                continue
            K_matrix = camera_params['K'].split(' ')
            parameter_list.append(float(K_matrix[idx]))
        return np.median(parameter_list)

    def draw_projected_box3d(self, image, qs, color=(255, 255, 255), thickness=2,
                             text=None, z_loc=None, show_action=False, social_action=[]):
        ''' Draw 3d bounding box in image
            qs: (8,3) array of vertices for the 3d box in following order:
                1 -------- 0                 up z
               /|         /|                    ^   x front
              2 -------- 3 .                    |  /
              | |        | |                    | /
              . 5 -------- 4     left y <------ 0
              |/         |/
              6 -------- 7
        '''
        qs = qs.astype(np.int32)
        special = False
        if text is not None:
            ID = text[text.find(':') + 1:] if isinstance(text, str) else text
            cv2.putText(image, f"{ID}",
                        (round((qs[2, 0] + qs[6, 0]) / 2), round((qs[2, 1] + qs[6, 1]) / 2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7, thickness=2, color=(0, 255, 0))  # BGR
        if z_loc is not None and self.show_z:
            cv2.putText(image, f"{round(z_loc, 2)}",
                        qs[4],
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=2, color=(0, 255, 0))  # BGR
        # cv2.line(image, (qs[1, 0], qs[1, 1]), (qs[4, 0], qs[4, 1],), (0, 255, 0), thickness, cv2.LINE_AA)
        # cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[5, 0], qs[5, 1],), (0, 255, 0), thickness, cv2.LINE_AA)
        front_color = (0, 255, 0)
        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            image = self.draw_line(image, qs[i], qs[j], color, thickness)
            # use LINE_AA for opencv3

            i, j = k + 4, (k + 1) % 4 + 4
            image = self.draw_line(image, qs[i], qs[j], color, thickness)

            i, j = k, k + 4
            image = self.draw_line(image, qs[i], qs[j], color, thickness)
            # cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
        for (i, j) in ((0, 5), (1, 4)):
            image = self.draw_line(image, qs[i], qs[j], color, thickness)
        if show_action:
            image = self.draw_social_act(image, qs[2], qs[6], social_action)

        return image

    def draw_social_act(self, image, point1, point2, social_action):
        (x1, y1), (x2, y2) = point1, point2
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, dist / 15):  # max 15 actions
            # print(i)
            r = i / dist
            x = int((x1 * (1 - r) + x2 * r) + .5)
            y = int((y1 * (1 - r) + y2 * r) + .5)
            p = (x, y)
            pts.append(p)
        for idx, act in enumerate(sorted(social_action)):
            cv2.circle(image, pts[idx + 1], radius=max(int(dist / 30), 3), thickness=-1,
                       # cv2.circle(image, pts[idx + 1], radius=3, thickness=-1,
                       color=self.social_act_colors[act][[2, 1, 0]] * 255)
        return image

    def draw_line(self, image, point1, point2, color, thickness, style=None):
        (x1, y1), (x2, y2) = point1, point2
        draw_fn = self.draw_dashed_line if style == "dashed" else cv2.line

        if abs(x2 - x1) > 800:
            (x1, y1, x2, y2) = (x1, y1, x2, y2) if (x1 < x2) else (x2, y2, x1, y1)
            a_shifted = [x1 + image.shape[1], y1]  # shift a to the right
            intersection = list(map(int, line_intersection(((x2, y2), a_shifted), ((image.shape[1], 0),
                                                                                   (image.shape[1], image.shape[0])))))
            draw_fn(image, (0, intersection[1]), (x1, y1),
                    color, thickness, lineType=cv2.LINE_AA)
            draw_fn(image, (x2, y2), (image.shape[1], intersection[1]), color, thickness, cv2.LINE_AA)
        else:
            draw_fn(image, (x1, y1), (x2, y2,), color,
                    thickness, lineType=cv2.LINE_AA)
        return image

    def draw_dashed_line(self, img, point1, point2, color, thickness=1, gap=10, lineType=cv2.LINE_AA):
        (x1, y1), (x2, y2) = point1, point2
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((x1 * (1 - r) + x2 * r) + .5)
            y = int((y1 * (1 - r) + y2 * r) + .5)
            p = (x, y)
            pts.append(p)
        try:
            # if style == 'dashed':
            s = pts[0]
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(img, s, e, color, thickness)
                i += 1
            # else:
            #     for p in pts:
            #         cv2.circle(img, p, thickness, color, -1)
        finally:
            return img

    def project_velo_to_ref(self, pointcloud):
        """
        Adjusts the coordinate system of a given point cloud to match that of the reference camera frame.
        
        This function assumes the input point cloud is in the Velodyne coordinate system, where the X axis points
        forward from the vehicle, the Y axis points to the left, and the Z axis points up. The function converts
        the point cloud to the reference camera coordinate system, where the X axis points to the right, the Y axis
        points down, and the Z axis points forward.
        
        Parameters:
        - pointcloud (numpy.ndarray): The input point cloud data in the Velodyne coordinate system.
        
        Returns:
        - numpy.ndarray: The point cloud adjusted to the reference camera coordinate system.
        """

        pointcloud = pointcloud[:, [1, 2, 0]]
        pointcloud[:, 0] *= -1
        pointcloud[:, 1] *= -1

        return pointcloud

    def move_lidar_to_camera_frame(self, pointcloud, upper=True):
        # assumed only rotation about z axis

        if upper:
            pointcloud[:, :3] = \
                pointcloud[:, :3] - torch.Tensor(self.global_config_dict['calibrated']
                                                 ['lidar_upper_to_rgb']['translation']).type(pointcloud.type())
            theta = self.global_config_dict['calibrated']['lidar_upper_to_rgb']['rotation'][-1]
        else:
            pointcloud[:, :3] = \
                pointcloud[:, :3] - torch.Tensor(self.global_config_dict['calibrated']
                                                 ['lidar_lower_to_rgb']['translation']).type(pointcloud.type())
            theta = self.global_config_dict['calibrated']['lidar_lower_to_rgb']['rotation'][-1]

        rotation_matrix = torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).type(
            pointcloud.type())
        pointcloud[:, :2] = torch.matmul(rotation_matrix, pointcloud[:, :2].unsqueeze(2)).squeeze()
        pointcloud[:, :3] = self.project_velo_to_ref(pointcloud[:, :3])
        return pointcloud

    def project_ref_to_image_torch(self, pointcloud):
        """
        Projects points from the reference camera coordinate system to the image plane using PyTorch.
        
        This function calculates the 2D coordinates of 3D points on the image plane, using the horizontal
        angle (theta) to determine the X coordinate and a combination of vertical position, angle, and
        precomputed camera parameters to determine the Y coordinate. It is specifically designed for
        panoramic image stitching.
        
        Parameters:
        - pointcloud (torch.Tensor): The input point cloud data in the reference camera coordinate system.
        
        Returns:
        - torch.Tensor: The 2D points on the image plane corresponding to the input 3D points.
        """
        theta = (torch.atan2(pointcloud[:, 0], pointcloud[:, 2]) + np.pi) % (2 * np.pi)
        horizontal_fraction = theta / (2 * np.pi)
        x = (horizontal_fraction * self.img_shape[2]) % self.img_shape[2]
        y = -self.median_focal_length_y * (
                pointcloud[:, 1] * torch.cos(theta) / pointcloud[:, 2]) + self.median_optical_center_y
        pts_2d = torch.stack([x, y], dim=1)

        return pts_2d

    def box_to_3d_corners(self, bboxes, ):
        # boxes: (N, 7): including x, y, z, w, l, h, rot
        """ Draw 3d bounding box in image
            qs: (8,3) array of vertices for the 3d box in following order:
                1 -------- 0
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7
        """
        rotated_bboxes = []
        for box in bboxes:
            x, y, z, w, l, h, rot = box
            x_corners = [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2]
            y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
            stacked_corners = np.vstack([x_corners, y_corners, z_corners]).T
            rotated_box = np.matmul(stacked_corners, rotz(rot, kitti_format=self.kitti_submission))
            rotated_box = rotated_box + [x, y, z]
            rotated_bboxes.append(rotated_box)
        return np.asarray(rotated_bboxes)

    def box_to_2d_corners(self, box):
        x, y, w, h = box
        x0, x1 = x, x + w
        y0, y1 = y, y + h
        return [x0, y0, x1, y1]

    def get_points(self,
                   lidar_upper_path=None,
                   lidar_lower_path=None,
                   ):
        all_points = None
        for i, pcd_path in enumerate([lidar_upper_path, lidar_lower_path]):
            if pcd_path == None:
                continue
            pcd = o3d.io.read_point_cloud(pcd_path)
            if i == 0:
                translation = np.asarray(self.global_config_dict['calibrated']['lidar_upper_to_rgb']['translation'])
                theta = self.global_config_dict['calibrated']['lidar_upper_to_rgb']['rotation'][-1]
            else:
                translation = np.asarray(self.global_config_dict['calibrated']['lidar_lower_to_rgb']['translation'])
                theta = self.global_config_dict['calibrated']['lidar_lower_to_rgb']['rotation'][-1]
            # Get the points
            points = torch.tensor(pcd.points).type(torch.float32)

            # Get rotation matrix from lidar to camera
            rotation_matrix = torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).type(
                points.type())
            # Start rotating
            points[:, :2] = torch.matmul(rotation_matrix, points[:, :2].unsqueeze(2)).squeeze()
            # Translate the position of points
            points -= translation

            if all_points is None:
                all_points = np.asarray(points)
            else:
                all_points = np.concatenate((all_points, points), axis=0)

        return all_points

    def filter_points(self, points: np.array, range: list = (-5, 5, -5, 5, -1.5, -0.5)):
        # range (x0,x1,y0,y1,z0,z1)
        return np.array([p for p in points if
                         (range[0] < p[0] < range[1]) & (range[2] < p[1] < range[3]) & (range[4] < p[2] < range[5])])

    def show_stitched_image_and_pcd(self,
                                    image_path,
                                    lidar_upper_path=None,
                                    lidar_lower_path=None,
                                    file_idx="000000",
                                    show_velo_points=False,
                                    show_3d_bboxes=False,
                                    show_2d_bboxes=False,
                                    show_2d_poses=False,
                                    show_social_group=False,
                                    show_social_action=False,
                                    show_individual_action=False,
                                    annos=None,
                                    color_ids=True,
                                    show_id=False):

        # rotation angle in degree
        if not isinstance(image_path, list):
            image = cv2.imread(image_path)
            # image = ndimage.rotate(image, -0.4)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if show_velo_points == True:
                for i, pcd_path in enumerate([lidar_upper_path, lidar_lower_path]):
                    if pcd_path == None:
                        continue
                    pcd = o3d.io.read_point_cloud(pcd_path)
                    points = torch.tensor(pcd.points).type(torch.float32)
                    points = self.move_lidar_to_camera_frame(points, upper=True if i == 0 else False)
                    points = self.project_ref_to_image_torch(points)
                    points = np.asarray(points).T
                    for i in range(points.shape[1]):
                        cv2.circle(hsv_image, (np.int64(points[0][i]), np.int64(points[1][i])), 1, (0, 0, 255), 1)
            hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            # cv2.putText(hsv_image, f"{file_idx}", (10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=1, thickness=3, color=(0, 0, 255))
            if show_3d_bboxes == True:
                bboxes_3d, label_ids, social_groups, actions = [], [], [], []

                for anno in annos.values():
                    if anno["bbox_3d"] is not None:
                        bboxes_3d.append(anno['bbox_3d'])
                        label_ids.append(anno['id'])
                        social_groups.append(anno['social_group'])
                        if show_social_action:
                            actions.append(anno['social_action'])
                        else:
                            actions.append(anno['individual_action'])

                bbox_corners = self.box_to_3d_corners(bboxes_3d)
                reshaped_corners = bbox_corners.reshape((-1, 3))
                velo2ref_corners = self.project_velo_to_ref(reshaped_corners)
                ref2img_corners = self.project_ref_to_image_torch(torch.tensor(velo2ref_corners))
                reshaped_ref2img_corners = ref2img_corners.reshape(-1, 8, 2)

                for box, label_id, social_group, action in zip(reshaped_ref2img_corners, label_ids,
                                                               social_groups, actions):
                    if self.kitti_submission:
                        color = [0, 0, 255]
                    elif show_social_group:
                        if social_group is not None:
                            color = self.cluster_colors[social_group - 1][[2, 1, 0]] * 255
                        else:
                            color = [0, 0, 0]
                    else:
                        color = self.ped_colors[label_id][[2, 1, 0]] * 255

                    # color = self.ped_colors[label_id][[2, 1, 0]] * 255 if not self.kitti_submission else [0, 0, 255]
                    hsv_image = self.draw_projected_box3d(hsv_image, np.asarray(box),
                                                          color=color,
                                                          thickness=2,
                                                          text=label_id if show_id else None,
                                                          show_action=show_social_action or show_individual_action,
                                                          social_action=action
                                                          )  # BGR

            if show_2d_bboxes:
                bboxes_2d, occlusions_2d, label_ids, social_groups, actions = [], [], [], [], []
                for anno in annos.values():
                    if anno["bbox_2d"] is not None:
                        bboxes_2d.append(anno['bbox_2d'])
                        label_ids.append(anno['id'])
                        social_groups.append(anno['social_group'])
                        if show_social_action:
                            actions.append(anno['social_action'])
                        else:
                            actions.append(anno['individual_action'])

                        occlusions_2d.append(anno['occlusion_2d'])

                for box, label_id, occlusion, social_group, action in zip(bboxes_2d, label_ids, occlusions_2d,
                                                                          social_groups, actions):
                    if self.kitti_submission:
                        color = [0, 0, 255]
                    elif show_social_group:
                        if social_group is not None:
                            color = self.cluster_colors[social_group - 1][[2, 1, 0]] * 255
                        else:
                            color = [0, 0, 0]
                    else:
                        color = self.ped_colors[label_id][[2, 1, 0]] * 255

                    box_2d_corners = self.box_to_2d_corners(box)
                    hsv_image = self.draw_box_2d(hsv_image, box_2d_corners,
                                                 color=color,
                                                 thickness=2, text=label_id, occluded=occlusion,
                                                 show_action=show_social_action or show_individual_action,
                                                 social_action=action
                                                 )  # BGR
            if show_2d_poses:
                occlude_color = {
                    0: (0, 0, 0),
                    1: (0, 0, 255),
                    2: (255, 255, 255)
                }
                joints_locs, joint_occlusions = [], []

                for anno in annos.values():
                    if anno["pose"] is not None:
                        joints_locs.append(anno['pose'])
                        joint_occlusions.append(anno['pose_occlusion'])

                # joints_locs = [anno['pose'] for anno in annos]
                # joint_occlusions = [anno['pose_occlusion'] for anno in annos]
                for joint_locs, occlusion in zip(joints_locs, joint_occlusions):
                    if joint_locs == None:
                        continue
                    joints = dict(zip(joint_map, np.round(joint_locs).astype(int)))
                    for (joint, joint_loc), occluded in zip(joints.items(), occlusion):
                        cv2.circle(hsv_image, joint_loc, thickness=2,
                                   color=occlude_color[occluded],
                                   radius=2)
                        cv2.circle(hsv_image, joint_loc, thickness=1,
                                   color=(255, 255, 255),
                                   radius=4)
                    for (joint1, joint2), color in joint_color.items():
                        self.draw_line(hsv_image, joints[joint1], joints[joint2], color,
                                       thickness=1)
            final_image = hsv_image
        else:
            individual_images = {}

            bbox3d_dict = {}
            new_dict = {}
            for index,item in annos.items():
                if isinstance(index,int):
                    bbox3d_dict[index] = item['bbox_3d']
                else:
                    new_dict[index] = item

            annos = new_dict
            for (idx, img_path), (cam_id, cam_annos), in zip(enumerate(image_path), annos.items()):
                hsv_image = cv2.imread(img_path)
                # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    
                if show_2d_bboxes or show_3d_bboxes:
                    bboxes_2d, occlusions_2d, label_ids, social_groups, actions = [], [], [], [], []
                    for anno in cam_annos.values():
                        if anno["bbox_2d"] is not None:
                            bboxes_2d.append(anno['bbox_2d'])
                            label_ids.append(anno['id'])
                            social_groups.append(anno['social_group'])
                            if show_social_action:
                                actions.append(anno['social_action'])
                            else:
                                actions.append(anno['individual_action'])
                            occlusions_2d.append(anno['occlusion_2d'])

                    for box, label_id, occlusion, social_group, action in zip(bboxes_2d, label_ids,
                                                                              occlusions_2d,
                                                                              social_groups, actions):
                        if self.kitti_submission:
                            color = [0, 0, 255]
                        elif show_social_group:
                            if social_group is not None:
                                color = self.cluster_colors[social_group - 1][[2, 1, 0]] * 255
                            else:
                                color = [0, 0, 0]
                        else:
                            color = self.ped_colors[label_id][[2, 1, 0]] * 255

                        if show_2d_bboxes:
                            box_2d_corners = self.box_to_2d_corners(box)
                            hsv_image = self.draw_box_2d(hsv_image, box_2d_corners,
                                                        color=color,
                                                        thickness=2, text=label_id, occluded=occlusion,
                                                        show_action=show_social_action or show_individual_action,
                                                        social_action=action
                                                        )  # BGR
                        if show_3d_bboxes:
                            if label_id in bbox3d_dict:
                                bbox_corners = self.box_to_3d_corners([bbox3d_dict[label_id]])
                                reshaped_corners = bbox_corners.reshape((-1, 3))
                                cam_calib = self.lidar_calib[cam_id]
                                # Reshape and homogenize the intrinsic camera matrix K to prepare for projection calculations.
                                K = np.array(cam_calib['distorted_img_K']).reshape(3, 3)
                                K = np.hstack([K, np.zeros((3, 1))])  # Add a column of zeros to homogenize the K matrix.
                                D = np.array(cam_calib['D'])  # Distortion coefficients.
                                ego2cam = np.linalg.inv(np.array(cam_calib["cam2ego"]))
                                reshaped_corners = np.hstack([np.array(reshaped_corners), np.ones((len(reshaped_corners), 1))])  # Homogenize the 3D points.
                                pts_ref = ego2cam.dot(np.array(reshaped_corners).T).T
                                pts_ref = pts_ref[pts_ref[:, 2] > 0]  # Keep only points in front of the camera.

                                # Normalize the points to the image plane.
                                pts_ref = pts_ref[:, :2] / pts_ref[:, 2].reshape(-1, 1)
                                # Apply radial distortion to the normalized points using the distortion coefficients.
                                x, y = pts_ref[:, 0], pts_ref[:, 1]
                                k1, k2, k3, k4, k5= D
                                k6 = 0  # Assuming no 6th distortion coefficient.
                                r2 = x ** 2 + y ** 2
                                r4 = x ** 4 + y ** 4
                                r6 = x ** 6 + y ** 6
                                fx, _, cx, _ = K[0]
                                _, fy, cy, _ = K[1]
                                rad_dist = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
                                xd = fx * rad_dist * x + cx
                                yd = fy * rad_dist * y + cy
                                pts_ref = np.stack([xd, yd], axis=1)

                                hsv_image = self.draw_projected_box3d(hsv_image, np.asarray(pts_ref),
                                                color=color,
                                                thickness=2,
                                                text=label_id if show_id else None,
                                                show_action=show_social_action or show_individual_action,
                                                social_action=action
                                                )  # BGR
                if show_2d_poses:
                    occlude_color = {
                        0: (0, 0, 0),
                        1: (0, 0, 255),
                        2: (255, 255, 255)
                    }
                    joints_locs, joint_occlusions = [], []

                    for anno in cam_annos.values():
                        if anno["pose"] is not None:
                            joints_locs.append(anno['pose'])
                            joint_occlusions.append(anno['pose_occlusion'])

                    # joints_locs = [anno['pose'] for anno in annos]
                    # joint_occlusions = [anno['pose_occlusion'] for anno in annos]
                    for joint_locs, occlusion in zip(joints_locs, joint_occlusions):
                        if joint_locs == None:
                            continue
                        joints = dict(zip(joint_map, np.round(joint_locs).astype(int)))
                        for (joint, joint_loc), occluded in zip(joints.items(), occlusion):
                            cv2.circle(hsv_image, joint_loc, thickness=2,
                                       color=occlude_color[occluded],
                                       radius=2)
                            cv2.circle(hsv_image, joint_loc, thickness=1,
                                       color=(255, 255, 255),
                                       radius=4)
                        for (joint1, joint2), color in joint_color.items():
                            self.draw_line(hsv_image, joints[joint1], joints[joint2], color,
                                           thickness=1)
                # if idx == 0:
                #     concat_image = hsv_image
                # else:
                #     concat_image = cv2.hconcat([concat_image,hsv_image])
                individual_images[cam_id] = hsv_image

            final_image = cv2.hconcat([individual_images['sensor_6'], individual_images['sensor_8'],
                                       individual_images['sensor_0'], individual_images['sensor_2'],
                                       individual_images['sensor_4']
                                       ])
        if show_social_action or show_individual_action:
            action_list = set(itertools.chain(*actions))
            y0, dy = 15, int(0.06 * window_height)
            for i, act in enumerate(sorted(action_list)):
                y = y0 + i * dy
                cv2.circle(final_image, (20, y), radius=15, thickness=-1,
                           color=self.social_act_colors[act][[2, 1, 0]] * 255)
                cv2.putText(final_image, act_num2text[act], (40, y + 8), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, thickness=3, color=self.social_act_colors[act][[2, 1, 0]] * 255)
        return final_image

    def build_pcd(self,
                  image_path=None,
                  lidar_upper_path=None,
                  lidar_lower_path=None,
                  color_pcd=False,
                  ):
        geo = []
        all_points = None
        for i, pcd_path in enumerate([lidar_upper_path, lidar_lower_path]):
            if pcd_path == None:
                continue
            pcd = o3d.io.read_point_cloud(pcd_path)
            if i == 0:
                translation = np.asarray(self.global_config_dict['calibrated']['lidar_upper_to_rgb']['translation'])
                theta = self.global_config_dict['calibrated']['lidar_upper_to_rgb']['rotation'][-1]
            else:
                translation = np.asarray(self.global_config_dict['calibrated']['lidar_lower_to_rgb']['translation'])
                theta = self.global_config_dict['calibrated']['lidar_lower_to_rgb']['rotation'][-1]
            # Get the points
            points = torch.tensor(pcd.points).type(torch.float32)
            # points = points[::5,:]

            # if all_points is None:
            #     all_points = np.asarray(points)
            # else:
            #     all_points = np.concatenate((all_points, points), axis=0)
            # print(points[::5,:].shape)
            # exit()
            if color_pcd:
                image = cv2.imread(image_path)

                projected_points = self.move_lidar_to_camera_frame(copy.deepcopy(points),
                                                                   upper=True if i == 0 else False)
                projected_points = self.project_ref_to_image_torch(projected_points)
                projected_points = np.floor(np.asarray(projected_points).T).astype(np.int64)

                true_where_x_on_img = (0 <= projected_points[1]) & (projected_points[1] < image.shape[1])
                true_where_y_on_img = (0 <= projected_points[0]) & (projected_points[1] < image.shape[0])
                true_where_point_on_img = true_where_x_on_img & true_where_y_on_img

                points = points[true_where_point_on_img]
                projected_points = projected_points.T[true_where_point_on_img].T

                colors = image[projected_points[1], projected_points[0]]  # BGR
                colors = np.squeeze(cv2.cvtColor(np.expand_dims(colors, 0), cv2.COLOR_BGR2RGB))

            # Get rotation matrix from lidar to camera
            rotation_matrix = torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).type(
                points.type())
            # Start rotating
            points[:, :2] = torch.matmul(rotation_matrix, points[:, :2].unsqueeze(2)).squeeze()
            # Translate the position of points
            points -= translation
            # points = self.filter_points(points, range=[-100,100,-100,100,-0.5,1])
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            if color_pcd:
                pc.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)
            geo.append(pc)

        return geo

    def draw_box_2d(self, image, box, color=(0, 0, 255), thickness=2, text=None, occluded=0,
                    show_action=False, social_action=None
                    ):
        style = 'dashed' if occluded == 1 else None
        x0, y0, x1, y1 = list(map(int, box))
        self.draw_line(image, (x0, y0), (x0, y1), color, thickness, style)
        self.draw_line(image, (x1, y0), (x1, y1), color, thickness, style)
        self.draw_line(image, (x0, y0), (x1, y0), color, thickness, style)
        self.draw_line(image, (x0, y1), (x1, y1), color, thickness, style)
        if show_action:
            image = self.draw_social_act(image, (x0, y0), (x0, y1), social_action)
        # if text:
        #     cv2.putText(image, f"{text}", (x0, y0), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=0.7, thickness=2, color=(0, 0, 255))
        return image

    def get_annos(self,
                  ts,
                  with_pose=False,
                  score=None):
        empty_anno = {"id": None,
                      "bbox_2d": None,
                      "occlusion_2d": None,
                      "bbox_3d": None,
                      "pose": None,
                      "pose_occlusion": None,
                      "social_group": None,
                      "social_action": [],
                      "individual_action": [],
                      }

        # 3D
        annos = {}
        # if self.labels_3d is not None and len(self.labels_2d) != 5:
        # pp.pprint(self.labels_3d)
        # exit()
        # if self.labels_3d is not None and sorted(self.labels_2d.keys()) != sorted(self.cam_ids):
        # TODO: handle individual image with 3D boxes
        # if self.labels_3d is not None and self.show_individual_image:
        #     print("Warning: individual image with 3D boxes is not supported yet")
        # if (self.labels_3d is not None) and not (self.show_individual_image):
        if self.labels_3d is not None:
            # and (sorted(self.labels_2d.keys()) != sorted(self.cam_ids))
            if "label_id" in self.labels_3d[f"{ts}.pcd"][0].keys():
                label_3d_ids = np.asarray(
                    [int(l['label_id'][l['label_id'].find(':') + 1:]) for l in self.labels_3d[f"{ts}.pcd"]])
            else:  # pseudo IDs
                label_3d_ids = np.asarray([i for i in range(len(self.labels_3d[f"{ts}.pcd"]))])

            # social grouping + action
            for idx, p in enumerate(self.labels_3d[f"{ts}.pcd"]):
                if score is not None:
                    if p['score'] < score:
                        continue
                if "label_id" in p.keys():
                    ped_id = int(p['label_id'][p['label_id'].find(':') + 1:])
                else:
                    ped_id = idx
                annos[ped_id] = copy.deepcopy(empty_anno)
                annos[ped_id]['id'] = ped_id
                annos[ped_id]['bbox_3d'] = [p['box']['cx'], p['box']['cy'], p['box']['cz'],
                                            p['box']['w'], p['box']['l'], p['box']['h'],
                                            p['box']['rot_z']]
                if 'social_group' in p.keys():
                    annos[ped_id]['social_group'] = p['social_group']['cluster_ID']
                    annos[ped_id]['social_action'] = [social_act_list[item] for item in p['social_activity'].keys()]
                    annos[ped_id]['individual_action'] = [social_act_list[item] for item in p['action_label'].keys()]
        if self.labels_2d is not None:
            if sorted(self.labels_2d.keys()) != sorted(self.cam_ids):
                for idx, p in enumerate(self.labels_2d[f"{ts}.jpg"]):
                    ped_id = int(p['label_id'][p['label_id'].find(':') + 1:])
                    if ped_id not in annos.keys():
                        annos[ped_id] = copy.deepcopy(empty_anno)
                    annos[ped_id]['id'] = ped_id
                    annos[ped_id]['bbox_2d'] = p['box']
                    annos[ped_id]['occlusion_2d'] = 1 if (
                            p['attributes']['occlusion'] not in ['Fully_visible', 'Mostly_visible']) else 0
                    if 'social_group' in p.keys():
                        annos[ped_id]['social_group'] = p['social_group']['cluster_ID']
                        annos[ped_id]['social_action'] = [social_act_list[item] for item in p['social_activity'].keys()]
                        annos[ped_id]['individual_action'] = [social_act_list[item] for item in
                                                              p['action_label'].keys()]
            else:
                for cam_id, cam_labels in self.labels_2d.items():
                    annos[f"sensor_{cam_id}"] = {}
                    for idx, p in enumerate(cam_labels[f"{ts}.jpg"]):
                        ped_id = int(p['label_id'][p['label_id'].find(':') + 1:])
                        if ped_id not in annos[f"sensor_{cam_id}"].keys():
                            annos[f"sensor_{cam_id}"][ped_id] = copy.deepcopy(empty_anno)
                        annos[f"sensor_{cam_id}"][ped_id]['id'] = ped_id
                        annos[f"sensor_{cam_id}"][ped_id]['bbox_2d'] = p['box']
                        annos[f"sensor_{cam_id}"][ped_id]['occlusion_2d'] = 1 if (
                                p['attributes']['occlusion'] not in ['Fully_visible', 'Mostly_visible']) else 0
                        if 'social_group' in p.keys():
                            annos[f"sensor_{cam_id}"][ped_id]['social_group'] = p['social_group']['cluster_ID']
                            annos[f"sensor_{cam_id}"][ped_id]['social_action'] = [social_act_list[item] for item in
                                                                      p['social_activity'].keys()]
                            annos[f"sensor_{cam_id}"][ped_id]['individual_action'] = [social_act_list[item] for item in
                                                                          p['action_label'].keys()]

        # if score is not None:
        #     scores = np.asarray([l['score'] for l in self.labels_3d[f"{ts}.pcd"]])
        #     mask = scores > score
        #     boxes_3d = boxes_3d[mask]
        #     label_3d_ids = label_3d_ids[mask]

        # # 2D
        # if self.labels_2d is not None:
        #     bboxes_2d = [l['box'] for l in self.labels_2d[f"{ts}.jpg"]]
        #     occlusions_2d = [1 if (l['attributes']['occlusion'] not in ['Fully_visible', 'Mostly_visible']) else 0 for l
        #                      in
        #                      self.labels_2d[f"{ts}.jpg"]]
        #     label_2d_ids = [int(l['label_id'][l['label_id'].find(':') + 1:]) for l in self.labels_2d[f"{ts}.jpg"]]

        # Pose
        if with_pose:
            img_id = int(ts) + 1

            if not sorted(self.labels_2d_pose.keys()) == sorted(self.cam_ids):
                for ped in self.labels_2d_pose['annotations']:
                    if ped['image_id'] != img_id:
                        continue
                    if ped['track_id'] in annos.keys():
                        if "keypoints" in ped.keys():
                            keypoints = ped['keypoints']
                        else:
                            continue
                        pose, pose_occlusion = [], []
                        for i in range(0, 51, 3):
                            pose.append([keypoints[i], keypoints[i + 1]])
                            pose_occlusion.append(keypoints[i + 2])
                        annos[ped['track_id']]['pose'] = pose
                        annos[ped['track_id']]['pose_occlusion'] = pose_occlusion
            else:
                for cam_id, cam_labels in self.labels_2d_pose.items():
                    for ped in cam_labels['annotations']:
                        if ped['image_id'] != img_id:
                            continue
                        if ped['track_id'] in annos[f"sensor_{cam_id}"].keys():
                            if "keypoints" in ped.keys():
                                keypoints = ped['keypoints']
                            else:
                                continue
                            pose, pose_occlusion = [], []
                            for i in range(0, 51, 3):
                                pose.append([keypoints[i], keypoints[i + 1]])
                                pose_occlusion.append(keypoints[i + 2])
                            annos[f"sensor_{cam_id}"][ped['track_id']]['pose'] = pose
                            annos[f"sensor_{cam_id}"][ped['track_id']]['pose_occlusion'] = pose_occlusion
        # annos = []
        # if self.labels_2d is not None:
        #     for (label_2d_id, box_2d, occlusion_2d) in zip(label_2d_ids, bboxes_2d, occlusions_2d):
        #         for (label_3d_id, box_3d) in zip(label_3d_ids, boxes_3d):
        #             if label_2d_id == label_3d_id:
        #                 anno = copy.deepcopy(empty_anno)
        #                 anno["id"] = label_2d_id
        #                 anno["bbox_2d"] = box_2d
        #                 anno["occlusion_2d"] = occlusion_2d
        #                 anno["bbox_3d"] = box_3d
        #
        #                 # Get poses
        #                 if label_2d_id in pose_peds.keys():
        #                     keypoints = pose_peds[label_2d_id]['keypoints']
        #                     pose, pose_occlusion = [], []
        #                     for i in range(0, 51, 3):
        #                         pose.append([keypoints[i], keypoints[i + 1]])
        #                         pose_occlusion.append(keypoints[i + 2])
        #                     anno['pose'] = pose
        #                     anno['pose_occlusion'] = pose_occlusion
        #
        #                 annos.append(anno)
        # elif self.labels_3d is not None:
        #     for (label_3d_id, box_3d) in zip(label_3d_ids, boxes_3d):
        #         anno = copy.deepcopy(empty_anno)
        #         anno["id"] = label_3d_id
        #         anno["bbox_3d"] = box_3d
        #         annos.append(anno)

        return annos

    def build_3d_boxes(self, points, color=(1, 0, 0)):
        # points shape: (8,3)
        # points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
        #           [0, 1, 1], [1, 1, 1]] x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
        points = points[[5, 6, 4, 7, 1, 2, 0, 3], :]
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7], [0, 6], [2, 4]]
        colors = [color for i in range(len(lines))]
        # print(len(colors))
        # exit()
        line_set = o3d.geometry.LineSet()

        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def visualize_2d(self,
                     root_dir,
                     location,
                     velo_loc='both',
                     box_ids=None,
                     except_box_ids=None,
                     area=None,
                     show_velo_points=False,
                     timestamps=None,
                     show_2d_labels=True,
                     show_3d_labels=True,
                     show_2d_poses=True,
                     show_social_group=False,
                     show_social_action=False,
                     show_individual_action=False,
                     shift_velo=0,
                     color_ids=True,
                     show_id=False,
                     score_filter=0.1,
                     save_as_vid=False,
                     show_individual_image=False,
                     video_output_dir=None
                     ):
        self.area = area
        self.show_individual_image = show_individual_image


        if timestamps is None:
            timestamps = [ts[:-4] for ts in self.labels_3d.keys()]
        if save_as_vid:
            # assert video_output_dir is not None, "select video output dir"
            if video_output_dir is None:
                video_output_dir = root_dir
                print(f"Video output dir set to {root_dir}")
            folder_name = "_"
            for k, v in {"Ind_Img": show_individual_image,
                         "2D": show_2d_labels,
                         "3D": show_3d_labels,
                         "Pose": show_2d_poses,
                         "Group": show_social_group,
                         "Soc_Action": show_social_action,
                         "Ind_Action": show_individual_action}.items():
                if v == True:
                    folder_name += f"{k}_" if k != "Action" else k
            os.makedirs(f'{video_output_dir}/output_videos_2D/{folder_name}', exist_ok=True)
            out = cv2.VideoWriter(f'{video_output_dir}/output_videos_2D/{folder_name}/{location}.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'),
                                  20, (window_width, window_height))
        else:
            print("Press any key to continue")
        for i, ts in tqdm(enumerate(timestamps)):
            # if int(ts) > 20:
            #     break
            label_3d_path, label_2d_path, img_path, \
            velo_upper_path, velo_lower_path, calib_folder, label_length, \
            label_2d_pose_path = get_infos(root_dir, location, ts, velo_loc,
                                           shift_velo=shift_velo,
                                           label_length=len(self.labels_3d),
                                           shift_type=shift_type,
                                           show_individual_image=show_individual_image,
                                           )

            annos = []
            # if show_2d_poses:
            #     peds = [ped for ped in self.labels_2d_pose['annotations'] if ped['image_id'] == int(ts)]
            #     for ped in peds:
            #         anno = copy.deepcopy(empty_anno)
            #         anno['id'] = ped['id']
            #         anno['bbox_2d'] = ped['bbox']
            #         anno['keypoints'] = ped['keypoints'] if "keypoints" in ped.keys() else None
            #         annos.append(anno)

            if not self.kitti_submission:
                annos = self.get_annos(ts, with_pose=show_2d_poses)
            else:
                annos = self.get_annos(ts, score=score_filter)
            if len(annos) == 0:
                continue
            hsv_image = self.show_stitched_image_and_pcd(img_path,
                                                         lidar_upper_path=velo_upper_path,
                                                         lidar_lower_path=velo_lower_path,
                                                         file_idx=ts,
                                                         show_velo_points=show_velo_points,
                                                         show_2d_bboxes=show_2d_labels,
                                                         show_3d_bboxes=show_3d_labels,
                                                         show_2d_poses=show_2d_poses,
                                                         show_social_group=show_social_group,
                                                         show_social_action=show_social_action,
                                                         show_individual_action=show_individual_action,
                                                         annos=annos,
                                                         color_ids=color_ids,
                                                         show_id=show_id)

            if save_as_vid:
                cv2.imwrite(f'{video_output_dir}/output_videos_2D/{folder_name}/temp.jpg', hsv_image)
                out.write(cv2.resize(cv2.imread(f'{video_output_dir}/output_videos_2D/{folder_name}/temp.jpg'),
                                     (window_width, window_height)))
                # time.sleep(0.05)
            else:
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', window_width, window_height)
                cv2.imshow(f'image', hsv_image)
                cv2.waitKey(0)

        if save_as_vid:
            out.release()
            os.remove(f"{video_output_dir}/output_videos_2D/{folder_name}/temp.jpg")
        print("Completed")

    def text_3d(self, text, pos, direction=None, degree=180.0, font='DejaVuSansMono.ttf', font_size=16):
        if direction is None:
            direction = (0., 0., 1.)

        from PIL import Image, ImageFont, ImageDraw
        from pyquaternion import Quaternion

        font_obj = ImageFont.truetype(font, font_size)
        font_dim = font_obj.getsize(text)

        img = Image.new('RGB', font_dim, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
        img = np.asarray(img)
        img_mask = img[:, :, 0] < 128
        indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
        pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

        raxis = np.cross([0.0, 0.0, 1.0], direction)
        if np.linalg.norm(raxis) < 1e-6:
            raxis = (0.0, 0.0, 1.0)
        trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
                 Quaternion(axis=direction, degrees=degree)).transformation_matrix
        trans[0:3, 3] = np.asarray(pos)
        pcd.transform(trans)
        return pcd

    def create_sphere(self, point, color, radius):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)  # create a small sphere to represent point
        sphere.translate(point)  # translate this sphere to point
        sphere.paint_uniform_color(color)
        return sphere

    # def create_geometry_at_points(points):
    #     geometries = o3d.geometry.TriangleMesh()
    #     for point in points:
    #         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # create a small sphere to represent point
    #         sphere.translate(point)  # translate this sphere to point
    #         geometries += sphere
    #     geometries.paint_uniform_color([1.0, 0.0, 0.0])
    #     return geometries
    def update_spheres(self, sphere_geos, box_corner, cur_sphere, action):
        (x1, y1, z1), (x2, y2, z2) = box_corner[1], box_corner[5]
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, dist / 15):  # max 15 actions
            # print(i)
            r = i / dist
            x = (x1 * r + x2 * (1 - r))
            y = (y1 * r + y2 * (1 - r))
            z = (z1 * r + z2 * (1 - r))
            p = (x, y, z)
            pts.append(p)

        for idx, act in enumerate(sorted(action)):
            cur_center = sphere_geos[cur_sphere + idx].get_center()
            sphere_geos[cur_sphere + idx].translate(-cur_center)
            sphere_geos[cur_sphere + idx].translate(pts[idx + 1])
            sphere_geos[cur_sphere + idx].paint_uniform_color(self.social_act_colors[act])
            # print(sphere_geos[cur_sphere + idx].get_center())
            # print(-sphere_geos[cur_sphere + idx].get_center())
            # exit()
        return sphere_geos

    def visualize_3d(self,
                     root_dir,
                     location,
                     velo_loc='both',
                     box_ids=None,
                     except_box_ids=None,
                     timestamps=None,
                     show_3d_labels=True,
                     show_social_group=False,
                     show_social_action=False,
                     show_individual_action=False,
                     show_as_vid=False,
                     front=[0.01988731276770269, 0.9464119395195808, -0.32234908953751495],
                     lookat=[7.3005555531276221, 1.5176243040374471, 1.9742039533310078],
                     up=[-0.015490803158388902, 0.32266582920794568, 0.94638617787827872],
                     zoom=0.029999999999999988,
                     show_id=False,
                     save_as_vid=False,
                     color_velo=False,
                     shift_velo=0,
                     score_filter=0.2,
                     video_output_dir=None
                     ):
        if video_output_dir == None:
            video_output_dir = root_dir
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=window_width, height=window_height, visible=False if save_as_vid else True)
        ctr = vis.get_view_control()
        opt = vis.get_render_option()
        opt.background_color = np.asarray(background_color)
        empty_box = o3d.geometry.LineSet()
        empty_box.lines = o3d.utility.Vector2iVector(np.array([[0, 1]] * 14))
        empty_box.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]] * 8))
        empty_box.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1]] * 14))

        empty_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.06, resolution=10)
        empty_sphere.paint_uniform_color(background_color)
        # empty_sphere.translate([0,0,-1])
        sphere_geos = [copy.deepcopy(empty_sphere) for _ in range(400)]

        # points = points[[5, 6, 4, 7, 1, 2, 0, 3], :]
        # lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
        #          [0, 4], [1, 5], [2, 6], [3, 7], [0, 6], [2, 4]]
        # colors = [color for i in range(len(lines))]
        # line_set = o3d.geometry.LineSet()
        geo = [copy.deepcopy(empty_box) for _ in range(300)]
        if save_as_vid:
            folder_name = "_"
            for k, v in {"Ind_Img": show_individual_image,
                         "2D": show_2d_labels,
                         "3D": show_3d_labels,
                         "Pose": show_2d_poses,
                         "Group": show_social_group,
                         "Soc_Action": show_social_action,
                         "Ind_Action": show_individual_action}.items():
                if v:
                    folder_name += f"{k}_" if k != "Ind_Action" else k
            os.makedirs(f'{video_output_dir}/output_videos_3D/{folder_name}', exist_ok=True)
            out = cv2.VideoWriter(f'{video_output_dir}/output_videos_3D/{folder_name}/{location}.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'),
                                  20, (window_width, window_height))

        for idx, ts in tqdm(enumerate(timestamps)):
            # if int(ts) > 20:
            #     break
            label_3d_path, label_2d_path, img_path, \
            velo_upper_path, velo_lower_path, calib_folder, label_length, \
            label_2d_pose_path = get_infos(
                root_dir, location, ts, velo_loc,
                # labels_ver=labels_ver,
                shift_velo=shift_velo, shift_type=shift_type,
                label_length=len(self.labels_3d))
            pcd_name = f'{ts}.pcd'
            if not self.kitti_submission:
                annos = self.get_annos(ts)
            else:
                annos = self.get_annos(ts, score=score_filter)
            if len(annos) == 0:
                continue
            bboxes, label_ids, social_groups, actions = [], [], [], []
            for anno in annos.values():
                if not "bbox_3d" in anno.keys():
                    break
                if anno["bbox_3d"] is not None:
                    bboxes.append(anno['bbox_3d'])
                    label_ids.append(anno['id'])
                    social_groups.append(anno['social_group'])
                    if show_social_action:
                        actions.append(anno['social_action'])
                    else:
                        actions.append(anno['individual_action'])

            bbox_corners = self.box_to_3d_corners(bboxes)
            cur_sphere = 0

            if show_3d_labels:
                for box_corner, box, label_id, social_group, action in zip(bbox_corners, bboxes,
                                                                           label_ids, social_groups,
                                                                           actions):
                    if show_id and not self.kitti_submission:
                        geo.append(self.text_3d(str(label_id), pos=box[:3],
                                                font_size=50))
                    # print(label_id, self.ped_colors[label_id])
                    if self.kitti_submission:
                        color = [1, 0, 0]
                    elif show_social_group:
                        if social_group is not None:
                            color = self.cluster_colors[social_group - 1]
                        else:
                            color = [0, 0, 0]
                    else:
                        color = self.ped_colors[label_id]

                    # color = self.ped_colors[label_id] if not self.kitti_submission else [1, 0, 0]
                    new_box = self.build_3d_boxes(box_corner, color=color)
                    box_corner = box_corner[[5, 6, 4, 7, 1, 2, 0, 3], :]
                    geo[label_id].lines = new_box.lines
                    geo[label_id].points = new_box.points
                    geo[label_id].colors = new_box.colors
                    if show_social_action or show_individual_action:
                        sphere_geos = self.update_spheres(sphere_geos, box_corner, cur_sphere=cur_sphere,
                                                          action=action)
                        cur_sphere += len(action)

                for sphere in sphere_geos[cur_sphere:]:
                    cur_center = sphere.get_center()
                    sphere.translate(-cur_center)
                    sphere.paint_uniform_color(background_color)
                    # sphere.translate([0,0,0])
                    # sphere.paint_uniform_color(background_color)

                for k in range(len(geo) - 2):
                    if k not in label_ids:
                        geo[k].colors = o3d.utility.Vector3dVector([background_color] * len(geo[k].colors))

            pcd_geo = self.build_pcd(image_path=img_path,
                                     lidar_upper_path=velo_upper_path,
                                     lidar_lower_path=velo_lower_path,
                                     color_pcd=color_velo
                                     )
            for j in range(2):
                if idx == 0:
                    pcd_geos = pcd_geo
                else:
                    pcd_geos[j].points = pcd_geo[j].points
                    pcd_geos[j].colors = pcd_geo[j].colors

            if idx == 0:
                for g in sphere_geos:
                    vis.add_geometry(g)
                for g in geo:
                    vis.add_geometry(g)
                for g in pcd_geos:
                    vis.add_geometry(g)

                ctr.set_zoom(zoom)
                ctr.set_up(up)
                ctr.set_lookat(lookat)
                ctr.set_front(front)

            else:
                for g in sphere_geos:
                    vis.update_geometry(g)
                for g in geo:
                    vis.update_geometry(g)
                for g in pcd_geos:
                    vis.update_geometry(g)
                vis.poll_events()
                vis.update_renderer()
            if not show_as_vid and not save_as_vid:
                vis.run()

            if save_as_vid:
                vis.capture_screen_image(f'{video_output_dir}/output_videos_3D/{folder_name}/temp.jpg', do_render=True)
                out.write(cv2.resize(cv2.imread(f'{video_output_dir}/output_videos_3D/{folder_name}/temp.jpg'),
                                     (window_width, window_height)))
        if save_as_vid:
            out.release()
            # shutil.rmtree(f"{video_output_dir}/output_videos_3D/temp")
            os.remove(f"{video_output_dir}/output_videos_3D/{folder_name}/temp.jpg")
            print("Video saved")
        vis.destroy_window()


def get_infos(root_dir="/home/tho/datasets/JRDB/train_dataset_with_activity",
              location="nvidia-aud-2019-04-18_0",
              file_index="001001",
              velo_loc="both",  # upper, lower, both
              labels_ver="",
              shift_velo=0,
              shift_type='image',
              label_length=None,
              show_individual_image=False,
              ):
    assert velo_loc in ['both', 'upper', 'lower']
    velo_upper_path, velo_lower_path = None, None
    label_3d_path = f"{root_dir}/{labels_ver}labels/labels_3d/{location}.json"

    calib_folder = f"{root_dir}/calibration"

    if label_length == None:
        with open(label_3d_path, 'r') as f:
            labels_3d = json.load(f)
            labels_3d = labels_3d['labels']
        label_length = len(labels_3d)

    if show_individual_image:
        label_2d_path = [f"{root_dir}/{labels_ver}labels/labels_2d/{location}_image{cam}.json" for cam in
                         [0, 2, 4, 6, 8]]
        label_2d_pose_path = [f"{root_dir}/{labels_ver}labels/labels_2d_pose_coco/{location}_image{cam}.json" for cam in
                              [0, 2, 4, 6, 8]]
        if shift_type == 'image':
            temp_index = "{:06d}".format(min(label_length - 1, max(0, int(file_index) + shift_velo)))
            img_path = [f"{root_dir}/images/image_{cam}/{location}/{temp_index}.jpg" for cam in [0, 2, 4, 6, 8]]
        else:
            img_path = [f"{root_dir}/images/image_{cam}/{location}/{file_index}.jpg" for cam in [0, 2, 4, 6, 8]]
    else:
        label_2d_path = f"{root_dir}/{labels_ver}labels/labels_2d_stitched/{location}.json"
        label_2d_pose_path = f"{root_dir}/{labels_ver}labels/labels_2d_pose_stitched_coco/{location}.json"
        if shift_type == 'image':
            temp_index = "{:06d}".format(min(label_length - 1, max(0, int(file_index) + shift_velo)))
            img_path = f"{root_dir}/images/image_stitched/{location}/{temp_index}.jpg"
        else:
            img_path = f"{root_dir}/images/image_stitched/{location}/{file_index}.jpg"

    if shift_type == 'velo':
        temp_index = "{:06d}".format(max(0, int(file_index) - shift_velo))
        if velo_loc in ['both', 'upper']:
            velo_upper_path = f"{root_dir}/pointclouds/upper_velodyne/{location}/{temp_index}.pcd"
        if velo_loc in ['both', 'lower']:
            velo_lower_path = f"{root_dir}/pointclouds/lower_velodyne/{location}/{temp_index}.pcd"
    else:
        if velo_loc in ['both', 'upper']:
            velo_upper_path = f"{root_dir}/pointclouds/upper_velodyne/{location}/{file_index}.pcd"
        if velo_loc in ['both', 'lower']:
            velo_lower_path = f"{root_dir}/pointclouds/lower_velodyne/{location}/{file_index}.pcd"

    return label_3d_path, label_2d_path, img_path, \
           velo_upper_path, velo_lower_path, calib_folder, label_length, \
           label_2d_pose_path


# def show_test_images(root_path, location):
#     imgs_dir = os.path.join(root_path, 'images', 'image_stitched', location)
#     img_names = os.listdir(imgs_dir)
#     for img_name in tqdm(img_names):
#         image = cv2.imread(os.path.join(imgs_dir, img_name))
#         cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('image', window_width, 500)
#         cv2.imshow(f'image', image)
#         cv2.waitKey(0)


if __name__ == '__main__':
    velo_loc = "both"  # both, upper, lower
    root_location = "/home/lk/workspase/python/dection/Sparse4D/data/JRDB2019/jrdb19_train_norosbag"  # path to JRDB

    viz_kitti_submissions = False  # change the predictions_dir if set this to True
    predictions_dir = '/location/of/predictions/in/KITTI/format'
    # for prediction visualisation
    # cubberly-auditorium-2019-04-22_1/000000.txt
    #                                 /000001.txt
    #                                 /000002.txt
    #                                 ...
    #                                 /001078.txt
    # ...
    # tressider-2019-04-26_3/000000.txt
    #                       /000001.txt
    #                       /000002.txt
    #                       ...
    #                       /001658.txt

    # configs = json.loads('vis_configs.json')
    
    with open('vis_configs.json', 'r') as f:
        configs = json.load(f)

    # Directory to save videos if save_as_vid == True
    video_output_dir = '/location/to/save/videos'
    # Save visualisation as a video, set it to False to inspect, Once you're happy with it, set to True.
    save_as_vid = False

    ##################################################
    ############# VISUALISATION OPTIONS ##############
    ##################################################

    # Choose the location/sequence to visualise
    locations = [
        'bytes-cafe-2019-02-07_0',
        'stlc-111-2019-04-19_0',
        'svl-meeting-gates-2-2019-04-08_0',
    ]

    viz_2D = True  # True is viz 2D, False if viz 3D
    show_individual_image   = True  # Show individual cameras
    show_2d_labels          = False  # Show 2D bboxes
    show_3d_labels          = True  # show 3d bboxes
    show_2d_poses           = False  # show 2d poses
    show_social_group       = False  # show social grouping
    show_social_action      = False  # show social actions
    show_individual_action  = False  # show individual actions
    show_velo_points        = False  # Project 3D lidar on to STITCHED IMAGE, Used when visualise 2D stitched image
    color_velo = True  # Color the pointcloud, Used when visualise 3D point cloud

    start_ts = 0  # starting at specific timestamp
    score_filter = 0.4  # Filter the predictions based on confident score

    viz_settings = [
        [viz_2D],  # viz2D
        [show_individual_image],  # indi 2D
        [show_2d_labels],  # 2D box
        [show_3d_labels],  # 3D box
        [show_2d_poses],  # 2D Pose
        [show_social_group],  # soc group
        [show_social_action],  # soc act
        [show_individual_action],  # indi act
    ]

    # shift_velo = 0  # Just ignore, my persional stuff
    # shift_type = 'velo'  # Just ignore, my persional stuff
    # labels_ver = ""  # ignore this, just my personal stuff

    monitor = get_monitors()
    window_width, window_height = int(monitor[0].width * 0.95), int(monitor[0].height * 0.95)

    for location in locations:
        if location in TRAIN:
            root_dir = root_location + "/train_dataset_with_activity"
        else:
            root_dir = root_location + "/test_dataset_without_labels"
        print(location)
        for (viz_2D, show_individual_image, show_2d_labels, show_3d_labels,
             show_2d_poses, show_social_group, show_social_action, \
             show_individual_action) in zip(*viz_settings):

            # project points inside the boxes only, check synchron of pc and image -> can be ts lag.
            if location in configs.keys():
                config = json.loads(configs[location])
            else:
                config = json.loads(configs[list(configs.keys())[-2]])
                print("config not found:", location)
            front = config['trajectory'][0]['front']
            lookat = config['trajectory'][0]['lookat']
            up = config['trajectory'][0]['up']
            zoom = config['trajectory'][0]['zoom']

            if not viz_kitti_submissions:
                label_3d_path, label_2d_path, img_path, \
                velo_upper_path, velo_lower_path, calib_folder, label_length, \
                label_2d_pose_path = get_infos(root_dir,
                                               location,
                                               velo_loc=velo_loc,
                                               # labels_ver=labels_ver,
                                               # shift_velo=shift_velo,
                                               # shift_type=shift_type,
                                               show_individual_image=show_individual_image,
                                               )
                visualizer = Visualizer(calib_folder,
                                        label_3d_path=label_3d_path,
                                        label_2d_path=label_2d_path,
                                        label_2d_pose_path=label_2d_pose_path if show_2d_poses else None)
                timestamps = sorted(list(visualizer.labels_3d.keys()))
            else:
                label_3d_path = f'{predictions_dir}/{location}'
                calib_folder = f'{root_dir}/calibration'
                visualizer = Visualizer(calib_folder,
                                        label_3d_path=label_3d_path,
                                        label_2d_path=None,
                                        kitti_submission=viz_kitti_submissions)
                timestamps = sorted(list(visualizer.labels_3d.keys()))

            ################################################
            ### VISUALIZE 2D STITCHED IMAGES WITH BBOXES ###
            ################################################
            if viz_2D:
                shift_type = 'image'
                if save_as_vid:
                    window_width, window_height = 1920, 246
                else:
                    window_height = 500
                visualizer.visualize_2d(root_dir,
                                        location,
                                        velo_loc=velo_loc,  # can be 'both', 'upper', 'lower'
                                        box_ids=None,  # to show specific boxes ID, eg. [1,2,3,4]
                                        except_box_ids=None,  # show all boxes EXCEPT specific boxes ID, eg. [1,2,3,4]
                                        show_velo_points=show_velo_points,  # show projected pointcloud on image
                                        timestamps=get_timestamps(timestamps, start_ts, None),
                                        # Set timestamps to visualize
                                        show_2d_labels=show_2d_labels,  # Show 2D labels
                                        show_3d_labels=show_3d_labels,
                                        show_2d_poses=show_2d_poses,
                                        show_social_group=show_social_group,
                                        show_social_action=show_social_action,
                                        show_individual_action=show_individual_action,
                                        show_individual_image=show_individual_image,
                                        # shift_velo=shift_velo,
                                        show_id=False,
                                        score_filter=score_filter,
                                        save_as_vid=save_as_vid,
                                        video_output_dir=video_output_dir)  # Show 3D labels
            else:  # viz 3D
                ###########################################
                ## VISUALIZE 3D POINTCLOUDS WITH BBOXES ###
                ###########################################
                if save_as_vid:
                    window_width, window_height = 1920, 834
                shift_type = 'velo'
                visualizer.visualize_3d(root_dir,
                                        location,
                                        box_ids=None,
                                        except_box_ids=None,
                                        velo_loc='both',  # can be 'both', 'upper', 'lower'
                                        timestamps=get_timestamps(timestamps, start_ts, None),
                                        # Set timestamps to visualize
                                        show_3d_labels=show_3d_labels,
                                        show_social_group=show_social_group,
                                        show_social_action=show_social_action,
                                        show_individual_action=show_individual_action,
                                        show_as_vid=False,
                                        front=front,
                                        lookat=lookat,
                                        up=up,
                                        zoom=zoom,
                                        show_id=False,
                                        save_as_vid=save_as_vid,
                                        color_velo=color_velo,
                                        # shift_velo=shift_velo,
                                        score_filter=score_filter,
                                        video_output_dir=video_output_dir
                                        )  # Show 3D labels
            time.sleep(5)
