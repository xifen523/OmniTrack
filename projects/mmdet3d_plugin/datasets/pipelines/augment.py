import torch

import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from PIL import Image


@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img, mat = self._img_transform(
                np.uint8(imgs[i]), aug_config,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]
                # results["cam_intrinsic"][i][:3, :3] = (
                #     mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]
                # )

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix


@PIPELINES.register_module()
class BBoxRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                results["lidar2img"][view] @ rot_mat_inv
            )
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(
                results["gt_bboxes_3d"], angle
            )
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str




@PIPELINES.register_module()
class ResizeCropFlipImageJRDB2D(object):
    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img, mat = self._img_transform(
                np.uint8(imgs[i]), aug_config,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            # if "lidar2img" not in results:
            #     results["lidar2img"] = {i:mat}
            # results["lidar2img"].append(mat)

            if "aug_mat" not in results:
                results["aug_mat"] = []
            results["aug_mat"].append(mat)

            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]
                # results["cam_intrinsic"][i][:3, :3] = (
                #     mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]
                # )

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        return img, transform_matrix
    

@PIPELINES.register_module()
class ExtendStitchedImageJRDB2D(object):

    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        stitchedImg = results["img"][0]

        extend_size = aug_config["crop"][2] - stitchedImg.shape[1]
        img_left = np.zeros((stitchedImg.shape[0], extend_size, 3), dtype=np.uint8)
        img_left[:, :extend_size, :] = stitchedImg[:, :extend_size, :]
        new_img = np.concatenate((stitchedImg,img_left), axis=1)

        results["img"] = [new_img]
        results["img_shape"] = [new_img.shape[:2]]
        
        if "aug_mat" not in results:
            results["aug_mat"] = []
        results["aug_mat"].append(np.eye(3))
        return results

@PIPELINES.register_module()
class BBoxAugJRDB2D(object):
    def __call__(self, results):
        
        assert "gt_bboxes_2d" in results, "train 2d bbox required"
        mat = results['aug_mat'][0]
        bbox2d = results["gt_bboxes_2d"]   # [N,4] (x1,y1,w,h)
        mask = (bbox2d[:,2]>0) & (bbox2d[:,3]>0)
        bbox2d = bbox2d[mask]

        results["gt_labels_2d"] = results["gt_labels_2d"][mask]
        results["gt_names"] = results["gt_names"][mask]
        results["instance_inds"] = results["instance_inds"][mask]
        
        nums_bbox = len(bbox2d)

        # convert cx,cy,w,h to x1,y1,x2,y2
        bbox2d_ = bbox2d.copy().astype(np.float32)
        half_wh = bbox2d_[:,2:]/2
        bbox2d_[:,0] -= half_wh[:,0]
        bbox2d_[:,1] -= half_wh[:,1]
        bbox2d_[:,2] = bbox2d_[:,0] + half_wh[:,0]*2
        bbox2d_[:,3] = bbox2d_[:,1] + half_wh[:,1]*2

        bbox2d_xyxy = bbox2d_.reshape(nums_bbox*2,2)
        bbox2d_xyxy = np.concatenate([bbox2d_xyxy, np.ones((nums_bbox*2,1))], axis=1)
        bbox2d_xyxy = np.matmul(mat, bbox2d_xyxy.T).T
        bbox2d_xyxy = bbox2d_xyxy[:,:2]
        bbox2d_ = bbox2d_xyxy.reshape(nums_bbox,4)

        if results["aug_config"]['flip']:
            bbox2d_ = bbox2d_[:, [2,1,0,3]]
        
        bbox2d_[:,2] -= bbox2d_[:,0]
        bbox2d_[:,3] -= bbox2d_[:,1]

        bbox2d_[:,0] += bbox2d[:,2]/2
        bbox2d_[:,1] += bbox2d[:,3]/2

        bbox2d_[:,2:] = np.clip(bbox2d_[:,2:], 1e-6, None)
        results["gt_bboxes_2d"] = bbox2d_

        return results
    
@PIPELINES.register_module()
class BBoxAugJRDB2DDETR(object):
    def __call__(self, results):
        
        assert "gt_bboxes_2d" in results, "train 2d bbox required"
        mat = results['aug_mat'][0]
        bbox2d = results["gt_bboxes_2d"]   # [N,4] (x1,y1,w,h)
        mask = (bbox2d[:,2]>0) & (bbox2d[:,3]>0)
        bbox2d = bbox2d[mask]

        results["gt_labels_2d"] = results["gt_labels_2d"][mask]
        results["gt_names"] = results["gt_names"][mask]
        results["instance_inds"] = results["instance_inds"][mask]
        
        nums_bbox = len(bbox2d)

        # convert cx,cy,w,h to x1,y1,x2,y2
        bbox2d_ = bbox2d.copy().astype(np.float32)
        half_wh = bbox2d_[:,2:]/2
        bbox2d_[:,0] -= half_wh[:,0]
        bbox2d_[:,1] -= half_wh[:,1]
        bbox2d_[:,2] = bbox2d_[:,0] + half_wh[:,0]*2
        bbox2d_[:,3] = bbox2d_[:,1] + half_wh[:,1]*2

        bbox2d_xyxy = bbox2d_.reshape(nums_bbox*2,2)
        bbox2d_xyxy = np.concatenate([bbox2d_xyxy, np.ones((nums_bbox*2,1))], axis=1)
        bbox2d_xyxy = np.matmul(mat, bbox2d_xyxy.T).T
        bbox2d_xyxy = bbox2d_xyxy[:,:2]
        bbox2d_ = bbox2d_xyxy.reshape(nums_bbox,4)

        if results["aug_config"]['flip']:
            bbox2d_ = bbox2d_[:, [2,1,0,3]]
        
        bbox2d_[:,2] -= bbox2d_[:,0]
        bbox2d_[:,3] -= bbox2d_[:,1]

        bbox2d_[:,0] += bbox2d[:,2]/2
        bbox2d_[:,1] += bbox2d[:,3]/2

        bbox2d_[:,2:] = np.clip(bbox2d_[:,2:], 1e-6, None)
        results["gt_bboxes_2d"] = bbox2d_   # cx,cy,w,h

        return results
    

@PIPELINES.register_module()
class BBoxExtendJRDB2DDETR(object):
    def __call__(self, results):
        
        assert "gt_bboxes_2d" in results, "train 2d bbox required"
        bbox2d = results["gt_bboxes_2d"]   # [N,4] (cx,cy,w,h)
        mask = (bbox2d[:,2]>0) & (bbox2d[:,3]>0)
        bbox2d = bbox2d[mask]

        results["gt_labels_2d"] = results["gt_labels_2d"][mask]
        results["gt_names"] = results["gt_names"][mask]
        results["instance_inds"] = results["instance_inds"][mask]
        results["gt_bboxes_2d"] = bbox2d   # cx,cy,w,h

        labels = results["gt_labels_2d"]
        gt_names = results["gt_names"]
        instance_inds = results["instance_inds"]
        bboxes_2d = results["gt_bboxes_2d"]

        img_size = results["img_shape"][0]
        extend_size = img_size[1] - results['pad_shape'][1]
        orign_shape = results['pad_shape'][:2]

        new_bboxes_2d = []
        new_gt_names = []
        new_instance_inds = []
        new_labels = []
        for i, box in enumerate(bboxes_2d):
            if box[0] < extend_size:  # extend to right
                new_bboxes_2d.append(np.array([box[0]+orign_shape[1], box[1], box[2], box[3]]))
                new_gt_names.append(gt_names[i])
                new_instance_inds.append(-instance_inds[i])
                new_labels.append(labels[i])
            if box[0] + (box[2]/2) > orign_shape[1] + 20  : # bbox out of right extend_pixel
                left_ = box[0] - box[2]/2
                w_r = orign_shape[1] - left_  # width of right bbox
                w_l = box[2] - w_r # width of left bbox
                new_bboxes_2d.append(np.array([w_l/2, box[1], w_l, box[3]]))
                new_gt_names.append(gt_names[i])
                new_instance_inds.append(-instance_inds[i])
                new_labels.append(labels[i])
            new_bboxes_2d.append(box)
            new_gt_names.append(gt_names[i])
            new_instance_inds.append(instance_inds[i])
            new_labels.append(labels[i])

        
        results["gt_bboxes_2d"] = np.array(new_bboxes_2d)
        results["gt_names"] = np.array(new_gt_names)
        results["instance_inds"] = np.array(new_instance_inds)
        results["gt_labels_2d"] = np.array(new_labels)

        return results