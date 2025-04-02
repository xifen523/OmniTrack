""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1280, 1024))
display.start()
import mayavi.mlab as mlab
mlab.options.offscreen = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))
sys.path.append('/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/kitti_object_vis/')
import kitti_util as utils
import argparse
import yaml
import gc
import shutil


try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]])

class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, image_dir, label_dir, lidar_upper_dir=None,lidar_lower_dir=None, calib_dir=None,args=None):
        """root_dir contains training and testing folders"""
        # self.root_dir = root_dir
        # self.split = split
        #print(root_dir, split)
        # self.split_dir = os.path.join(root_dir, split)

        # if split == "training":
        #     self.num_samples = 7481
        # elif split == "testing":
        #     self.num_samples = 7518
        # else:
        #     print("Unknown split: %s" % (split))
        #     exit(-1)

        # lidar_dir = "velodyne"
        # depth_dir = "depth"
        # pred_dir = "pred"
        # if args is not None:
        #     lidar_dir = args.lidar
        #     depth_dir = args.depthdir
        #     pred_dir = args.preddir

        self.image_dir =image_dir
        self.label_dir = label_dir
        self.calib_dir = calib_dir
        self.lidar_upper_dir=lidar_upper_dir
        self.lidar_lower_dir=lidar_lower_dir

        # self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        # self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        # self.depth_dir = os.path.join(self.split_dir, depth_dir)
        # self.pred_dir = os.path.join(self.split_dir, pred_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        #assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%06d.jpg" % (idx))
        #print(' the image:',img_filename)
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        #assert idx < self.num_samples
        #print(self.lidar_lower_dir,idx)
        lidar_filename_lower = os.path.join(self.lidar_lower_dir,"%06d.pcd" % (idx))
        lidar_filename_upper = os.path.join(self.lidar_upper_dir ,"%06d.pcd" % (idx))
        #print(lidar_filename)
        lower=utils.load_velo_scan(lidar_filename_lower, dtype, n_vec)
        upper=utils.load_velo_scan(lidar_filename_upper, dtype, n_vec)
        return np.concatenate((lower,upper),axis=0)

    def get_calibration(self, idx):
        #assert idx < self.num_samples
        #print(self.calib_dir)
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx,frame_idx):
        #assert idx < self.num_samples and self.split == "training"
        #label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        label_filename = os.path.join(self.label_dir, "%04d.txt" % (idx))
        #print('===============',label_filename)
        #return []
        return utils.read_label(label_filename,frame_idx)
    def get_label_objects_group(self, seq_index,frame_idx):
        label_filename = os.path.join(self.label_dir, "det_group.txt")
        #label_filename =os.path.join("/pvol2/jrdb_dev/jrdb_website_dev/backend/groundtruths/","gt_group.txt")
        return utils.read_label_group(label_filename,seq_index,frame_idx)
    
    def get_label_objects_action(self, seq_index,frame_idx):
        label_filename = os.path.join(self.label_dir, "det_action.txt")
        return utils.read_label_action(label_filename,seq_index,frame_idx)

    def get_label_objects_activity(self, seq_index,frame_idx):
        label_filename = os.path.join(self.label_dir, "det_activity.txt")
        return utils.read_label_activity(label_filename,seq_index,frame_idx)
    
    def get_label_objects_det(self,frame_idx):
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (frame_idx))
        return utils.read_label_det(label_filename)
    def get_pred_objects(self, idx):
        #assert idx < self.num_samples
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        #assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        #assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        #assert idx < self.num_samples
        lidar_filename = os.path.join(self.depthpc_dir, "%06d.bin" % (idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist
        # print(lidar_filename, is_exist)
        # return utils.load_velo_scan(lidar_filename), is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        #assert idx < self.num_samples and self.split == "training"
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        #assert idx < self.num_samples and self.split == "training"
        depth_filename = os.path.join(self.depth_dir, "%06d.txt" % (idx))
        return os.path.exists(depth_filename)


class kitti_object_video(object):
    """ Load data for KITTI videos """

    def __init__(self, img_dir, lidar_dir, calib_dir):
        #print('calib_dir:',calib_dir)
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted(
            [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        )
        self.lidar_filenames = sorted(
            [os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)]
        )
        #print(len(self.img_filenames))
        #print(len(self.lidar_filenames))
        # assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = self.img_filenames[idx]
        print('image name',img_filename)
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert idx < self.num_samples
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib


def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, "dataset/2011_09_26/")
    dataset = kitti_object_video(
        os.path.join(video_path, "2011_09_26_drive_0023_sync/image_02/data"),
        os.path.join(video_path, "2011_09_26_drive_0023_sync/velodyne_points/data"),
        video_path,
    )
    #print(len(dataset))
    for _ in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        cv2.imshow("video", img)
        draw_lidar(pc)
        raw_input()
        pc[:, 0:3] = dataset.get_calibration().project_velo_to_rect(pc[:, 0:3])
        draw_lidar(pc)
        raw_input()
    return


def show_image_with_boxes(img, objects, calib,obj_colors, show3d=True, depth=None, sub_type='2dt'):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    #img3 = np.copy(img)  # for 3d bbox
    #TODO: change the color of boxes
    for obj in objects:
        if sub_type=='2dt' or sub_type=='3dt':
            if obj.id in obj_colors:
                color=obj_colors[obj.id]
            else:
                color=np.random.choice(range(256), size=3)
                obj_colors[obj.id]=color
        else:
            color=(int(0),int(0),int(255))

        img1=cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (int(color[0]),int(color[1]),int(color[2])),
            2,
        )
        #print(color)
        if sub_type=='action' or sub_type=='group' or sub_type=='activity':
            x, y = int(obj.xmin), int(obj.ymin)
            text_size, _ = cv2.getTextSize(str(int(obj.id)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0)
            text_w, text_h = text_size
            img1=cv2.rectangle(img1, (x,y), (x + text_w, y + text_h), (255,255,255), -1)
            img1=cv2.putText(img1,str(int(obj.id)), (x, y + text_h + 1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 0, cv2.LINE_AA)
        if sub_type=='activity':
            x, y = int(obj.xmax), int(obj.ymin)+text_h
            text_size, _ = cv2.getTextSize(str(int(obj.id)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0)
            text_w, text_h = text_size
            img1=cv2.rectangle(img1, (x,y), (x + text_w, y + text_h), (255,255,255), -1)
            img1=cv2.putText(img1,str(int(obj.group_id)), (x, y + text_h + 1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 0, cv2.LINE_AA)
        #putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
        # box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        # if obj.type == "Car":
        #     img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        # elif obj.type == "Pedestrian":
        #     img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        # elif obj.type == "Cyclist":
        #     img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))


        # project
        # box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # box3d_pts_32d = utils.box3d_to_rgb_box00(box3d_pts_3d_velo)
        # box3d_pts_32d = calib.project_velo_to_image(box3d_pts_3d_velo)
        # img3 = utils.draw_projected_box3d(img3, box3d_pts_32d)
    # print("img1:", img1.shape)
    #cv2.imshow("2dbox", img1)
    # print("img3:",img3.shape)
    # Image.fromarray(img3).show()
    # show3d = True
    # if show3d:
    #     # print("img2:",img2.shape)
    #     cv2.imshow("3dbox", img2)
    # if depth is not None:
    #     cv2.imshow("depth", depth)
    
    return img1, img2


def show_image_with_boxes_3type(img, objects, calib, objects2d, name, objects_pred):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    type_list = ["Pedestrian", "Car", "Cyclist"]
    # draw Label
    color = (0, 255, 0)
    for obj in objects:
        if obj.type not in type_list:
            continue
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            color,
            3,
        )
    startx = 5
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [obj.type for obj in objects if obj.type in type_list]
    text_lables.insert(0, "Label:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    # draw 2D Pred
    color = (0, 0, 255)
    for obj in objects2d:
        cv2.rectangle(
            img1,
            (int(obj.box2d[0]), int(obj.box2d[1])),
            (int(obj.box2d[2]), int(obj.box2d[3])),
            color,
            2,
        )
    startx = 85
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [type_list[obj.typeid - 1] for obj in objects2d]
    text_lables.insert(0, "2D Pred:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    # draw 3D Pred
    if objects_pred is not None:
        color = (255, 0, 0)
        for obj in objects_pred:
            if obj.type not in type_list:
                continue
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                color,
                1,
            )
        startx = 165
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [obj.type for obj in objects_pred if obj.type in type_list]
        text_lables.insert(0, "3D Pred:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(
                img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA
            )

    cv2.imshow("with_bbox", img1)
    cv2.imwrite("imgs/" + str(name) + ".png", img1)


def get_lidar_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def get_lidar_index_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=0.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    # fov_inds = (
    #     (pts_2d[:, 0] < xmax)
    #     & (pts_2d[:, 0] >= xmin)
    #     & (pts_2d[:, 1] < ymax)
    #     & (pts_2d[:, 1] >= ymin)
    # )
    fov_inds = (
        (pts_2d[:, 0] < 100000)
        & (pts_2d[:, 0] >= -100000)
        & (pts_2d[:, 1] < 100000)
        & (pts_2d[:, 1] >= -100000)
    )
    #fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    fov_inds = fov_inds
    return fov_inds


def depth_region_pt3d(depth, obj):
    b = obj.box2d
    # depth_region = depth[b[0]:b[2],b[2]:b[3],0]
    pt3d = []
    # import pdb; pdb.set_trace()
    for i in range(int(b[0]), int(b[2])):
        for j in range(int(b[1]), int(b[3])):
            pt3d.append([j, i, depth[j, i]])
    return np.array(pt3d)


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def show_lidar_with_depth(
    image,
    pc_velo,
    objects,
    calib,
    fig,
    img_fov=False,
    img_width=None,
    img_height=None,
    sub_type='3dt',
    objects_pred=None,
    depth=None,
    cam_img=None,
    constraint_box=False,
    pc_label=False,
    save=False,
    obj_colors=None
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """

    #print(("All point num: ", pc_velo.shape[0]))
    #if False:
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        # pc_velo_index = get_lidar_index_in_image_fov(
        # pc_velo[:, :3], calib, -1000, -1000, img_width*100, img_height*100
        # )
        pc_velo = pc_velo[pc_velo_index, :]
        #print(("FOV point num: ", pc_velo.shape))
    
    #print("pc_velo", pc_velo[:2])
    #print(asd)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    # # Draw depth
    # if depth is not None:
    #     depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

    #     indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
    #     depth_pc_velo = np.hstack((depth_pc_velo, indensity))
    #     # print("depth_pc_velo:", depth_pc_velo.shape)
    #     # print("depth_pc_velo:", type(depth_pc_velo))
    #     # print(depth_pc_velo[:5])
    #     draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

    #     if save:
    #         data_idx = 0
    #         vely_dir = "data/object/training/depth_pc"
    #         save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))
    #         # np.save(save_filename+".npy", np.array(depth_pc_velo))
    #         depth_pc_velo = depth_pc_velo.astype(np.float32)
    #         depth_pc_velo.tofile(save_filename)

    # color = (0, 1, 0)
    color_list=[]
    for obj in objects:
        if sub_type=='3dt':
            if obj.id in obj_colors:
                color=obj_colors[obj.id]
            else:
                color=np.random.choice(range(256), size=3)/256.0
                obj_colors[obj.id]=color
        else:
            color=(int(0),int(0),int(1))
        color_list.append(tuple(color))
    boxes3d_pts_3d = utils.compute_boxes_3d(objects, calib.P)
    boxes3d_pts_3d_img = boxes3d_pts_3d.reshape((-1, 3))
    boxes3d_pts_3d_img = project_velo_to_ref(boxes3d_pts_3d_img)
    boxes3d_pts_3d_img=project_ref_to_image_torch(boxes3d_pts_3d_img)
    boxes3d_pts_3d_img = boxes3d_pts_3d_img.reshape(-1, 8, 2)
    draw_gt_boxes3d(boxes3d_pts_3d, fig=fig, color=(0,0,1), label=None,color_list=color_list)
    image=draw_projected_boxes3d(image, boxes3d_pts_3d_img, thickness=2, text=None, z_loc=None,color_list=color_list)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=35.0,
        figure=fig,
    )
    return image
def draw_projected_boxes3d(image, qs, thickness=2, text=None, z_loc=None,color_list=None):
    for i,box in enumerate(qs):
        image=draw_projected_box3d(image, box,color_list[i], thickness=2, text=None, z_loc=None)
    return image
def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2, text=None, z_loc=None):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    color=(int(color[0]*255),int(color[1]*255),int(color[2]*255))
    qs = qs.astype(np.int32)
    # special = [4, 8, 13, 21, 24, 26, 27, 37, 38, 39, 42, 46, 48, 49, 55, 61]
    special = None
    if text is not None:
        ID = int(text[text.find(':') + 1:])
        if special:
            if ID in special:
                cv2.putText(image, f"{ID}",
                            (round((qs[2, 0] + qs[6, 0]) / 2), round((qs[2, 1] + qs[6, 1]) / 2)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7, thickness=3, color=(0, 255, 0))  # BGR
        else:
            cv2.putText(image, f"{ID}",
                        (round((qs[2, 0] + qs[6, 0]) / 2), round((qs[2, 1] + qs[6, 1]) / 2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7, thickness=2, color=(0, 255, 0))  # BGR    
    front_color = (0, 255, 0)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        if abs(qs[j, 0] - qs[i, 0]) > 800:
            (a, b) = (i, j) if (qs[i, 0] < qs[j, 0]) else (j, i)
            cv2.line(image, (0, round((qs[b, 1] + qs[a, 1]) / 2)), (qs[a, 0], qs[a, 1]),
                     color, thickness, cv2.LINE_AA)
            cv2.line(image, (qs[b, 0], qs[b, 1]), (image.shape[1], round((qs[b, 1] + qs[a, 1]) / 2)),
                     color, thickness,
                     cv2.LINE_AA)
        else:
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1],), color,
                     thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        if abs(qs[j, 0] - qs[i, 0]) > 800:
            (a, b) = (i, j) if (qs[i, 0] < qs[j, 0]) else (j, i)
            cv2.line(image, (0, round((qs[b, 1] + qs[a, 1]) / 2)), (qs[a, 0], qs[a, 1]),
                     color, thickness, cv2.LINE_AA)
            cv2.line(image, (qs[b, 0], qs[b, 1]), (image.shape[1], round((qs[b, 1] + qs[a, 1]) / 2)),
                     color, thickness,
                     cv2.LINE_AA)
        else:
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1],), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        # cv2.putText(image, f"({qs[i, 0]}, {qs[i, 1]})", (qs[i, 0], qs[i, 1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5, thickness=2, color=(0, 255, 0))
        # cv2.putText(image, f"({qs[j, 0]}, {qs[j, 1]})", (qs[j, 0], qs[j, 1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5, thickness=2, color=(0, 255, 0))

    return image
def draw_gt_boxes3d(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=2,
    draw_text=False,
    text_scale=(1, 1, 1),
    color_list=None,
    label=""
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    """
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        # if draw_text:
        #     mlab.text3d(
        #         b[4, 0],
        #         b[4, 1],
        #         b[4, 2],
        #         label,
        #         scale=text_scale,
        #         color=color,
        #         figure=fig,
        #     )
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
    # mlab.view(
    #     azimuth=180,
    #     elevation=70,
    #     focalpoint=[12.0909996, -1.04700089, -2.03249991],
    #     distance=35.0,
    #     figure=fig,
    # )
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig
def save_depth0(
    data_idx,
    pc_velo,
    calib,
    img_fov,
    img_width,
    img_height,
    depth,
    constraint_box=False,
):

    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        type = np.zeros((pc_velo.shape[0], 1))
        pc_velo = np.hstack((pc_velo, type))
        print(("FOV point num: ", pc_velo.shape))
    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))

        type = np.ones((depth_pc_velo.shape[0], 1))
        depth_pc_velo = np.hstack((depth_pc_velo, type))
        print("depth_pc_velo:", depth_pc_velo.shape)

        depth_pc = np.concatenate((pc_velo, depth_pc_velo), axis=0)
        print("depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def save_depth(
    data_idx,
    pc_velo,
    calib,
    img_fov,
    img_width,
    img_height,
    depth,
    constraint_box=False,
):

    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc = np.hstack((depth_pc_velo, indensity))

        print("depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def show_lidar_with_boxes(
    pc_velo,
    objects,
    calib,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb(width=1920, height=1080)
        vdisplay.start()
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        #print("box3d_pts_3d_velo:")
        #print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)

        # Draw depth
        if depth is not None:
            # import pdb; pdb.set_trace()
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))

        # Draw heading arrow
        _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            # print("box3d_pts_3d_velo:")
            # print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )


def box_min_max(box3d):
    box_min = np.min(box3d, axis=0)
    box_max = np.max(box3d, axis=0)
    return box_min, box_max


def get_velo_whl(box3d, pc):
    bmin, bmax = box_min_max(box3d)
    ind = np.where(
        (pc[:, 0] >= bmin[0])
        & (pc[:, 0] <= bmax[0])
        & (pc[:, 1] >= bmin[1])
        & (pc[:, 1] <= bmax[1])
        & (pc[:, 2] >= bmin[2])
        & (pc[:, 2] <= bmax[2])
    )[0]
    # print(pc[ind,:])
    if len(ind) > 0:
        vmin, vmax = box_min_max(pc[ind, :])
        return vmax - vmin
    else:
        return 0, 0, 0, 0


def stat_lidar_with_boxes(pc_velo, objects, calib):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """

    # print(('All point num: ', pc_velo.shape[0]))

    # draw_lidar(pc_velo, fig=fig)
    # color=(0,1,0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        v_l, v_w, v_h, _ = get_velo_whl(box3d_pts_3d_velo, pc_velo)
        print("%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    """ Project LiDAR points to image """
    img =  np.copy(img)
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            2,
            color=tuple(color),
            thickness=-1,
        )
    cv2.imshow("projection", img)
    return img


def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    top_view = utils.lidar_to_top(pc_velo)
    top_image = utils.draw_top_image(top_view)
    print("top_image:", top_image.shape)
    # gt

    def bbox3d(obj):
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = utils.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    cv2.imshow("top_image", top_image)
    return top_image


def dataset_viz(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    ## load 2d detection results
    #objects2ds = read_det_file("box2d.list")

    if args.show_lidar_with_depth:
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb(width=1920, height=1080)
        vdisplay.start()
        import mayavi.mlab as mlab

        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
        )
    for data_idx in range(len(dataset)):
        if args.ind > 0:
            data_idx = args.ind
        # Load data from dataset
        if args.split == "training":
            objects = dataset.get_label_objects(data_idx)
        else:
            objects = []
        #objects2d = objects2ds[data_idx]

        objects_pred = None
        if args.pred:
            # if not dataset.isexist_pred_objects(data_idx):
            #    continue
            objects_pred = dataset.get_pred_objects(data_idx)
            if objects_pred == None:
                continue
        if objects_pred == None:
            print("no pred file")
            # objects_pred[0].print_object()

        n_vec = 4
        if args.pc_label:
            n_vec = 5

        dtype = np.float32
        if args.dtype64:
            dtype = np.float64
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        if args.depth:
            depth, _ = dataset.get_depth(data_idx)
            print(data_idx, "depth shape: ", depth.shape)
        else:
            depth = None

        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))

        if args.stat:
            stat_lidar_with_boxes(pc_velo, objects, calib)
            continue
        print("======== Objects in Ground Truth ========")
        n_obj = 0
        for obj in objects:
            if obj.type != "DontCare":
                print("=== {} object ===".format(n_obj + 1))
                obj.print_object()
                n_obj += 1

        # Draw 3d box in LiDAR point cloud
        if args.show_lidar_topview_with_boxes:
            # Draw lidar top view
            show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred)

        # show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)
        if args.show_image_with_boxes:
            # Draw 2d and 3d boxes on image
            show_image_with_boxes(img, objects, calib, True, depth)
        if args.show_lidar_with_depth:
            # Draw 3d box in LiDAR point cloud
            show_lidar_with_depth(
                pc_velo,
                objects,
                calib,
                fig,
                args.img_fov,
                img_width,
                img_height,
                objects_pred,
                depth,
                img,
                constraint_box=args.const_box,
                save=args.save_depth,
                pc_label=args.pc_label,
            )
            # show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height, \
            #    objects_pred, depth, img)
        if args.show_lidar_on_image:
            # Show LiDAR points on image.
            show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width, img_height)
        input_str = raw_input()

        mlab.clf()
        if input_str == "killall":
            break


def depth_to_lidar_format(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    for data_idx in range(len(dataset)):
        # Load data from dataset

        pc_velo = dataset.get_lidar(data_idx)[:, 0:4]
        calib = dataset.get_calibration(data_idx)
        depth, _ = dataset.get_depth(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        print(data_idx, "depth shape: ", depth.shape)
        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))
        save_depth(
            data_idx,
            pc_velo,
            calib,
            args.img_fov,
            img_width,
            img_height,
            depth,
            constraint_box=args.const_box,
        )
        #input_str = raw_input()


def read_det_file(det_filename):
    """ Parse lines in 2D detection output files """
    #det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    objects = {}
    with open(det_filename, "r") as f:
        for line in f.readlines():
            obj = utils.Object2d(line.rstrip())
            if obj.img_name not in objects.keys():
                objects[obj.img_name] = []
            objects[obj.img_name].append(obj)
        # objects = [utils.Object2d(line.rstrip()) for line in f.readlines()]

    return objects




def visualise_jrdb(submission_path,sub_type,seqs):
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0') # FourCC is a 4-byte code used to specify the video codec.
    width = 3760
    height = 480
    # gt_dir=''
    label_dir=os.path.join(submission_path.replace('.zip',''),'CIWT','data')
    # /pvol2/jrdb_dev/jrdb_website_dev/media/submissions/u6361796@anu.edu.au/2dt/2021-12-07 04:20:42+00:00_inyoung/
    # /pvol2/jrdb_dev/jrdb_website_dev/static/videos/leaderboards_videos/
    submission_path=submission_path.replace('.zip','')
    video_dir=submission_path.replace('media/submissions','static/videos/leaderboards_videos')
    #video_dir=os.path.join(video_path,'videos')
    os.makedirs(video_dir)
    for i in range(len(test)):
        if test[i] not in seqs:
            continue
        image_dir=os.path.join(gt_dir,test[i])
        dataset = kitti_object(image_dir, label_dir)
        obj_colors=dict()
        video = cv2.VideoWriter(os.path.join(video_dir,test[i]+'.webm'), fourcc, float(15), (width, height))
        end_idx=len(list(os.walk(image_dir+"/"))[0][-1])
        for frame_idx in range(end_idx):
            img = dataset.get_image(frame_idx)
            objects = dataset.get_label_objects(i,frame_idx)
            img_bbox2d, img_bbox3d = show_image_with_boxes(img, objects, None,obj_colors,sub_type=sub_type)
            video.write(img_bbox2d)
        video.release()
train = [
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

test = [
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
def visualise_jrdb_3d(submission_path,sub_type,seqs):
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0') # FourCC is a 4-byte code used to specify the video codec.
    width = 900
    height = 500
    # gt_dir=''
    gt_image_dir='/pvol2/jrdb_dev/jrdb_website_dev/static/downloads/jrdb_test/test_dataset_without_labels/images/image_stitched/'
    label_dir=os.path.join(submission_path.replace('.zip',''),'CIWT','data')
    submission_path=submission_path.replace('.zip','')
    video_dir=submission_path.replace('media/submissions','static/videos/leaderboards_videos')
    #video_dir=os.path.join(video_path,'videos')
    os.makedirs(video_dir)
    fig_3d = mlab.figure(bgcolor=(1, 1, 1), size=(width, height))
    for i in range(len(test)):
        if test[i] not in seqs:
            continue
        lidar_lower_dir=os.path.join(gt_dir,'lower_velodyne',test[i])
        lidar_upper_dir=os.path.join(gt_dir,'upper_velodyne',test[i])
        img_dir=os.path.join(gt_image_dir,test[i])
        
        dataset = kitti_object(image_dir=img_dir,label_dir=label_dir,lidar_lower_dir=lidar_lower_dir,lidar_upper_dir=lidar_upper_dir,calib_dir='/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/kitti_object_vis/data/object_jrdb_3d/training/calib/')
        obj_colors=dict()
        video = cv2.VideoWriter(os.path.join(video_dir,test[i]+'.webm'), fourcc, float(15), (width, height+120))
        end_idx=len(list(os.walk(lidar_lower_dir+"/"))[0][-1])
        for frame_idx in range(end_idx):
            img=dataset.get_image(frame_idx)
            pc_velo = dataset.get_lidar(frame_idx)
            objects = dataset.get_label_objects(i,frame_idx)
            calib = dataset.get_calibration(1)
            image=show_lidar_with_depth(img,pc_velo, objects, calib, fig_3d, True, width, height,obj_colors=obj_colors)
            image=cv2.resize(image,(900,120))            
            test_vis=np.array(mlab.screenshot())
            concat=np.concatenate((image,test_vis),axis=0)
            video.write(concat)
            mlab.clf()
        video.release()
def visualise_jrdb_det(submission_path,sub_type,seqs):
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0') # FourCC is a 4-byte code used to specify the video codec.
    width = 3760
    height = 480
    # gt_dir=''
    submission_path=submission_path.replace('.zip','')
    #video_dir=submission_path.replace('media/submissions','static/videos/leaderboards_videos')
    video_dir='./'
    #os.mkdir(video_dir)
    for i in range(len(test)):
        if test[i] not in seqs:
            continue
        label_dir=os.path.join(submission_path.replace('.zip',''),test[i],'image_stitched')
        label_dir_gt=os.path.join('/pvol2/jrdb_dev/jrdb_website_dev/backend/groundtruths/KITTI/',test[i])
        image_dir=os.path.join(gt_dir,test[i])
        dataset = kitti_object(image_dir, label_dir)
        dataset_gt = kitti_object(image_dir, label_dir_gt)
        obj_colors=dict()
        video = cv2.VideoWriter(os.path.join(video_dir,test[i]+'.webm'), fourcc, float(15), (width, height))
        end_idx=len(list(os.walk(image_dir+"/"))[0][-1])
        for frame_idx in range(end_idx):
            img = dataset.get_image(frame_idx)
            objects = dataset.get_label_objects_det(frame_idx)
            objects_gt = dataset_gt.get_label_objects_det(frame_idx)
            # print(objects_gt)
            # print(objects)
            objects=filter_preds(objects,objects_gt,0.7)
            img_bbox2d, img_bbox3d = show_image_with_boxes(img, objects, None,obj_colors,sub_type=sub_type)
            video.write(img_bbox2d)
        video.release()
def filter_preds(preds,gts,thre):
    res=[]
    bboxes1=np.zeros((len(preds),4))
    bboxes2=np.zeros((len(gts),4))
    for i in range(len(preds)):
        tmp=preds[i]
        bboxes1[i]=np.array([tmp.xmin,tmp.ymin,tmp.xmax,tmp.ymax])
    for i in range(len(gts)):
        tmp=gts[i]
        bboxes2[i]=np.array([tmp.xmin,tmp.ymin,tmp.xmax,tmp.ymax])
    pass
    ious=calculate_box_ious(bboxes1, bboxes2)
    ious=ious.max(axis=1)
    ious=list(np.where(ious>thre)[0])
    res=[pred for i,pred in enumerate(preds) if i in ious]
    return res
def calculate_box_ious(bboxes1, bboxes2, box_format="xywh", do_ioa=False):
    """ Calculates the IOU (intersection over union) between two arrays of boxes.
    Allows variable box formats ('xywh' and 'x0y0x1y1').
    If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
    used to determine if detections are within crowd ignore region.
    """

    # layout: (x0, y0, x1, y1)
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(
        min_[..., 3] - max_[..., 1], 0
    )
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1]
    )
    if do_ioa:
        ioas = np.zeros_like(intersection)
        valid_mask = area1 > 0 + np.finfo("float").eps
        ioas[valid_mask, :] = (
            intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]
        )

        return ioas
    else:
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1]
        )
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
        intersection[area1 <= 0 + np.finfo("float").eps, :] = 0
        intersection[:, area2 <= 0 + np.finfo("float").eps] = 0
        intersection[union <= 0 + np.finfo("float").eps] = 0
        union[union <= 0 + np.finfo("float").eps] = 1
        ious = intersection / union
        return ious
def visualise_jrdb_3d_det(submission_path,sub_type,seqs,old_submission=None):
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0') # FourCC is a 4-byte code used to specify the video codec.
    width = 900
    height = 500
    # gt_dir=''
    gt_image_dir='/pvol2/jrdb_dev/jrdb_website_dev/static/downloads/jrdb_test/test_dataset_without_labels/images/image_stitched/'
    
    if old_submission==None:
        submission_path=submission_path.replace('.zip','')
        video_dir=submission_path.replace('media/submissions','static/videos/leaderboards_videos')
        os.mkdir(video_dir)
    else:
        video_dir=os.path.join('old_submission',old_submission)
    #for i in range(1):
    fig_3d = mlab.figure(bgcolor=(1, 1, 1), size=(width, height))
    for i in range(len(test)):
        if test[i] not in seqs:
            continue
        #print(test[i])
        lidar_lower_dir=os.path.join(gt_dir,'lower_velodyne',test[i])
        lidar_upper_dir=os.path.join(gt_dir,'upper_velodyne',test[i])
        img_dir=os.path.join(gt_image_dir,test[i])
        label_dir=os.path.join(submission_path.replace('.zip',''),test[i])
        label_dir_gt=os.path.join('/pvol2/jrdb_dev/jrdb_website_dev/backend/groundtruths/KITTI/',test[i])
        obj_colors=dict()
        video = cv2.VideoWriter(os.path.join(video_dir,test[i]+'.webm'), fourcc, float(15), (width, height+120))
        #video = cv2.VideoWriter(os.path.join(test[i]+'.webm'), fourcc, float(15), (width, height+120))
        end_idx=len(list(os.walk(lidar_lower_dir+"/"))[0][-1])
        end_idx=100
        for frame_idx in range(end_idx):
            #print(frame_idx)
            dataset = kitti_object(image_dir=img_dir,label_dir=label_dir,lidar_lower_dir=lidar_lower_dir,lidar_upper_dir=lidar_upper_dir,calib_dir='/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/kitti_object_vis/data/object_jrdb_3d/training/calib/')
            dataset_gt = kitti_object(image_dir=img_dir,label_dir=label_dir_gt,lidar_lower_dir=lidar_lower_dir,lidar_upper_dir=lidar_upper_dir,calib_dir='/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/kitti_object_vis/data/object_jrdb_3d/training/calib/')
            img=dataset.get_image(frame_idx)
            # #print('img',img)
            pc_velo = dataset.get_lidar(frame_idx)
            objects = dataset.get_label_objects_det(frame_idx)
            #print('before',len(objects))
            objects_gt = dataset_gt.get_label_objects_det(frame_idx)
            objects=filter_preds_3d(objects,objects_gt,0.3)
            #print('after',len(objects))
            calib = dataset.get_calibration(1)
            image=show_lidar_with_depth(img,pc_velo, objects, calib, fig_3d, True, width, height,obj_colors=obj_colors,sub_type=sub_type)
            image=cv2.resize(image,(900,120))            
            test_vis=np.array(mlab.screenshot())
            concat=np.concatenate((image,test_vis),axis=0)
            video.write(concat)
            mlab.clf()
        video.release()
def filter_preds_3d(preds,gts,thre):
    res=[]
    bboxes1=np.zeros((len(preds),7))
    bboxes2=np.zeros((len(gts),7))
    for i in range(len(preds)):
        tmp=preds[i]
        bboxes1[i]=np.array([tmp.h,tmp.w,tmp.l,-tmp.t[1],-tmp.t[2]+tmp.h/2,tmp.t[0],tmp.ry])
    for i in range(len(gts)):
        tmp=gts[i]
        bboxes2[i]=np.array([tmp.h,tmp.w,tmp.l,-tmp.t[1],-tmp.t[2]+tmp.h/2,tmp.t[0],tmp.ry])
    ious=1-iou_matrix_3d(bboxes1, bboxes2)
    # print(ious)
    ious=ious.max(axis=1)
    ious=list(np.where(ious>thre)[0])
    res=[pred for i,pred in enumerate(preds) if i in ious]
    return res   
def iou_matrix_3d(objs, hyps, max_iou=1.):
    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0,0))
    assert objs.shape[1] == 7
    assert hyps.shape[1] == 7

    C = np.empty((objs.shape[0], hyps.shape[0]))
    for o in range(objs.shape[0]):
        for h in range(hyps.shape[0]):
            obj=objs[o]
            hyp=hyps[h]
            if np.all(obj==-np.ones(7)):
                C[o, h] = 0
                continue
            elif np.all(hyp==-np.ones(7)):
                C[o, h] = 1
                continue
            obj=np.array([obj[3],obj[5],obj[4],obj[1],obj[0],obj[2],obj[6]])
            hyp=np.array([hyp[3],hyp[5],hyp[4],hyp[1],hyp[0],hyp[2],obj[6]])
            base_area = find_area(clip_polygon(obj, hyp))
            height = max(obj[2], hyp[2]) - min(obj[2] - obj[4], hyp[2]-hyp[4])
            intersect = base_area*height
            union = obj[3]*obj[4]*obj[5] + hyp[3]*hyp[4]*hyp[5] - intersect
            if union != 0:
                if (1. - intersect / union) < -0.1:
                    C[o, h] = max_iou
                else:
                    C[o, h] = 1. - intersect / union
            else:
                C[o, h] = max_iou
    C[C > max_iou] = max_iou
    return C
def find_area(vertices):
    area = 0
    # print(vertices)
    # print(asd)
    for i in range(len(vertices)):
        area += vertices[i][0]*(vertices[(i+1)%len(vertices)][1] - vertices[i-1][1])
    return 0.5*abs(area)
def clip_polygon(box1, box2):
    #clips box 1 by the edges in box2
    x,y,z,l,h,w,theta = box2
    theta = -theta

    box2_edges = np.asarray([(-np.cos(theta), -np.sin(theta), l/2-x*np.cos(theta)-z*np.sin(theta)),
                    (-np.sin(theta), np.cos(theta), w/2-x*np.sin(theta)+z*np.cos(theta)),
                    (np.cos(theta), np.sin(theta), l/2+x*np.cos(theta)+z*np.sin(theta)),
                    (np.sin(theta), -np.cos(theta), w/2+x*np.sin(theta)-z*np.cos(theta))])
    x,y,z,l,h,w,theta = box1
    theta = -theta

    box1_vertices = [(x+l/2*np.cos(theta)-w/2*np.sin(theta), z+l/2*np.sin(theta)+w/2*np.cos(theta)),
                        (x+l/2*np.cos(theta)+w/2*np.sin(theta), z+l/2*np.sin(theta)-w/2*np.cos(theta)),
                        (x-l/2*np.cos(theta)-w/2*np.sin(theta), z-l/2*np.sin(theta)+w/2*np.cos(theta)),
                        (x-l/2*np.cos(theta)+w/2*np.sin(theta), z-l/2*np.sin(theta)-w/2*np.cos(theta))]
    out_vertices = sort_points(box1_vertices, (x, z))
    for edge in box2_edges:
        vertex_list = out_vertices.copy()
        out_vertices = []
        for idx, current_vertex in enumerate(vertex_list):
            previous_vertex = vertex_list[idx-1]
            if point_inside_edge(current_vertex, edge):
                if not point_inside_edge(previous_vertex, edge):
                    out_vertices.append(compute_intersection_point(previous_vertex, current_vertex, edge))
                out_vertices.append(current_vertex)
            elif point_inside_edge(previous_vertex, edge):
                out_vertices.append(compute_intersection_point(previous_vertex, current_vertex, edge))
    to_remove = []
    for i in range(len(out_vertices)):
        if i in to_remove:
            continue
        for j in range(i+1, len(out_vertices)):
            if abs(out_vertices[i][0] - out_vertices[j][0]) < 1e-6 and abs(out_vertices[i][1] - out_vertices[j][1]) < 1e-6:
                to_remove.append(j)
    out_vertices = sorted([(v[0]-x, v[1]-z) for i,v in enumerate(out_vertices) if i not in to_remove], key = lambda p: get_angle((p[0],p[1])))
    return out_vertices
def sort_points(pts, center):
    x, z = center
    sorted_pts = sorted([(i, (v[0]-x, v[1]-z)) for i,v in enumerate(pts)], key = lambda p: get_angle((p[1][0],p[1][1])))
    idx, _ = zip(*sorted_pts)
    return [pts[i] for i in idx]
def compute_intersection_point(pt1, pt2, line1):
    if pt1[0] == pt2[0]:
        slope = np.inf
    else:
        slope = (pt1[1]-pt2[1])/(pt1[0] - pt2[0])
    if np.isinf(slope):
        line2 = (1, 0, pt1[0])
    else:
        line2 = (slope, -1, pt1[0]*slope-pt1[1])
    # print("Line1:", line1)
    # print("Line2:", line2)
    if line1[1] == 0:
        x = line1[2]/line1[0]
        y = (line2[2] - line2[0]*x)/line2[1]
    elif line1[0] == 0:
        y = line1[2]/line1[1]
        x = (line2[2] - line2[1]*y)/line2[0]
    elif line2[1] == 0:
        x = pt1[0]
        y = (line1[2]-x*line1[0])/line1[1]
    else:
        tmp_line = (line2 - line1*(line2[1]/line1[1]))
        x = tmp_line[2]/tmp_line[0]
        y = (line2[2] - line2[0]*x)/line2[1]
    return (x,y)
def get_angle(p):
    x, y = p
    angle = np.arctan2(y,x)
    if angle < 0:
        angle += np.pi*2
    return angle
def point_inside_edge(pt, edge):
    lhs = pt[0]*edge[0] + pt[1]*edge[1]
    if lhs < edge[2] - 1e-6:
        return True
    else:
        return False
def visualise_jrdb_group(submission_path,sub_type,seqs,is_evaluated_all):
    print(is_evaluated_all)
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0') # FourCC is a 4-byte code used to specify the video codec.
    width = 3760
    height = 480
    # gt_dir=''
    label_dir=submission_path.replace('.zip','')
    submission_path=submission_path.replace('.zip','')
    video_dir=submission_path.replace('media/submissions','static/videos/leaderboards_videos')
    os.mkdir(video_dir)
    for i in range(len(test)):
        if test[i] not in seqs:
            continue
        image_dir=os.path.join(gt_dir,test[i])
        dataset = kitti_object(image_dir, label_dir)
        obj_colors=dict()
        video = cv2.VideoWriter(os.path.join(video_dir,test[i]+'.webm'), fourcc, float(0.25), (width, height))
        end_idx=(len(list(os.walk(image_dir+"/"))[0][-1])//15)*15
        objects=[]
        for frame_idx in range(15,end_idx):
            img=dataset.get_image(frame_idx)
            if frame_idx%15==0:
                objects = dataset.get_label_objects_group(i,frame_idx)
                #print('obj',len(objects),'--------------------------')
                tmp="%s,%04d" % (i, int(frame_idx))
                if tmp in is_evaluated_all:
                    #print('all',len(is_evaluated_all[tmp]))
                    eva_objects=[]
                    all_objs=is_evaluated_all[tmp]
                    for j in range(len(all_objs)):
                        if all_objs[j]:
                            eva_objects.append(objects[j])
                    #print(len(eva_objects))
                    objects=eva_objects
                else:
                    objects=[]
            else:
                objects=[]
            if frame_idx%15==0:
                for j in range(10):
                    img_bbox2d, img_bbox3d = show_image_with_boxes(img, objects, None,obj_colors,sub_type=sub_type)
                    video.write(img_bbox2d)
                    mlab.clf()
            else:
                video.write(img)
                mlab.clf()
        video.release()

def visualise_jrdb_action(submission_path,sub_type,seqs,is_evaluated_all):
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0') # FourCC is a 4-byte code used to specify the video codec.
    width = 3760
    height = 480+113
    # gt_dir=''
    label_dir=submission_path.replace('.zip','')
    submission_path=submission_path.replace('.zip','')
    video_dir=submission_path.replace('media/submissions','static/videos/leaderboards_videos')
    os.mkdir(video_dir)
    #print('at least hereeeeeeeeeeeeeeeeeeeeeee',len(test))
    legend=cv2.imread('/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/kitti_object_vis/legend_resize.png')
    for i in range(len(test)):
        if test[i] not in seqs:
            continue
        image_dir=os.path.join(gt_dir,test[i])
        dataset = kitti_object(image_dir, label_dir)
        obj_colors=dict()
        video = cv2.VideoWriter(os.path.join(video_dir,test[i]+'.webm'), fourcc, float(15), (width, height))
        end_idx=(len(list(os.walk(image_dir+"/"))[0][-1])//15)*15
        objects=[]
        for frame_idx in range(end_idx):
            img=dataset.get_image(frame_idx)
            if frame_idx%15==0:
                objects = dataset.get_label_objects_action(i,frame_idx)
                tmp="%s,%04d" % (i, int(frame_idx))
                if tmp in is_evaluated_all:
                    #print('all',len(is_evaluated_all[tmp]))
                    eva_objects=[]
                    all_objs=is_evaluated_all[tmp]
                    for j in range(len(all_objs)):
                        if all_objs[j]:
                            eva_objects.append(objects[j])
                    #print(len(eva_objects))
                    objects=eva_objects
                else:
                    objects=[]
            else:
                objects=[]
            if frame_idx%15==0:
                for j in range(10):
                    img_bbox2d, img_bbox3d = show_image_with_boxes(img, objects, None,obj_colors,sub_type=sub_type)
                    concat=np.concatenate((np.array(legend),img_bbox2d),axis=0)
                    video.write(concat)
                    mlab.clf()
            else:
                concat=np.concatenate((np.array(legend),img),axis=0)
                video.write(concat)
                mlab.clf()
        video.release()
def visualise_jrdb_activity(submission_path,sub_type,seqs,is_evaluated_all):
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0') # FourCC is a 4-byte code used to specify the video codec.
    width = 3760
    height = 480+113
    # gt_dir=''
    label_dir=submission_path.replace('.zip','')
    submission_path=submission_path.replace('.zip','')
    video_dir=submission_path.replace('media/submissions','static/videos/leaderboards_videos')
    #os.mkdir(video_dir)
    #print('at least hereeeeeeeeeeeeeeeeeeeeeee',len(test))
    legend=cv2.imread('/pvol2/jrdb_dev/jrdb_website_dev/backends_dev/kitti_object_vis/legend_resize.png')
    for i in range(len(test)):
        if test[i] not in seqs:
            continue
        image_dir=os.path.join(gt_dir,test[i])
        dataset = kitti_object(image_dir, label_dir)
        obj_colors=dict()
        #video = cv2.VideoWriter(os.path.join(video_dir,test[i]+'.webm'), fourcc, float(15), (width, height))
        video = cv2.VideoWriter(os.path.join(test[i]+'.webm'), fourcc, float(15), (width, height))
        end_idx=(len(list(os.walk(image_dir+"/"))[0][-1])//15)*15
        objects=[]
        for frame_idx in range(end_idx):
            img=dataset.get_image(frame_idx)
            if frame_idx%15==0:
                objects = dataset.get_label_objects_activity(i,frame_idx)
                tmp="%s,%04d" % (i, int(frame_idx))
                if tmp in is_evaluated_all:
                    #print('all',len(is_evaluated_all[tmp]))
                    eva_objects=[]
                    all_objs=is_evaluated_all[tmp]
                    for j in range(len(all_objs)):
                        if all_objs[j]:
                            eva_objects.append(objects[j])
                    #print(len(eva_objects))
                    #objects=eva_objects
                else:
                    #objects=[]
                    pass
            else:
                objects=[]
            if frame_idx%15==0:
                #for j in range(10):
                for j in range(1):
                    img_bbox2d, img_bbox3d = show_image_with_boxes(img, objects, None,obj_colors,sub_type=sub_type)
                    concat=np.concatenate((np.array(legend),img_bbox2d),axis=0)
                    video.write(concat)
                    mlab.clf()
            else:
                continue
                concat=np.concatenate((np.array(legend),img),axis=0)
                video.write(concat)
                mlab.clf()
        video.release()
def draw_lidar(
    pc,
    color=None,
    fig=None,
    bgcolor=(255, 255, 255),
    pts_scale=1.0,
    pts_mode="sphere",
    pts_color=None,
    color_by_intensity=True,
    pc_label=False,
):
    """ Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """
    # ind = (pc[:,2]< -1.65)
    # pc = pc[ind]
    pts_mode = "2dthick_cross"
    pts_scale = 0.01
    #print("====================", pc.shape)
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
        )
    if color is None:
        color = pc[:, 2]
    if pc_label:
        color = pc[:, 4]
    if color_by_intensity:
        color = pc[:, 2]

    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color,
        color=pts_color,
        mode=pts_mode,
        colormap="gnuplot",
        scale_factor=pts_scale,
        figure=fig,
    )
    return fig   
def project_ref_to_image_torch(pointcloud):

    theta = (np.arctan2(pointcloud[:, 0], pointcloud[:, 2]) + np.pi) % (2 * np.pi)
    horizontal_fraction = theta / (2 * np.pi)
    x = (horizontal_fraction * 3760) % 3760
    median_focal_length_y = calculate_median_param_value(param='f_y')
    median_optical_center_y = calculate_median_param_value(param='t_y')
    y = -median_focal_length_y * (
            pointcloud[:, 1] * np.cos(theta) / pointcloud[:, 2]) + median_optical_center_y
    pts_2d = np.stack([x, y], axis=1)

    return pts_2d
def project_velo_to_ref(box3d_pts_3d_img):
    box3d_pts_3d_img = box3d_pts_3d_img[:, [1, 2, 0]]
    box3d_pts_3d_img[:, 0] *= -1
    box3d_pts_3d_img[:, 1] *= -1
    return box3d_pts_3d_img
def calculate_median_param_value( param):
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
    camera_config_dict = yaml.safe_load(open('/pvol2/jrdb_dev/jrdb_website_dev/static/downloads/jrdb_test/test_dataset_without_labels/calibration/cameras.yaml'))
    #print(camera_config_dict)
    omni_camera = ['sensor_0', 'sensor_2', 'sensor_4', 'sensor_6', 'sensor_8']
    parameter_list = []
    for sensor, camera_params in camera_config_dict['cameras'].items():
        if sensor not in omni_camera:
            continue
        K_matrix = camera_params['K'].split(' ')
        parameter_list.append(float(K_matrix[idx]))
    return np.median(parameter_list)
if __name__ == "__main__":
    # eg. call visualise_jrdb_3d_det
    pass
