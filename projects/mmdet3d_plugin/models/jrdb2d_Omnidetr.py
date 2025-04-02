# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch
import numpy as np

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["JRDB2DOMNIDETR"]


@DETECTORS.register_module()
class JRDB2DOMNIDETR(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(JRDB2DOMNIDETR, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            # feature_maps = list(self.img_neck(feature_maps))
            feature_maps = self.img_neck(feature_maps)
        # for i, feat in enumerate(feature_maps):
        #     feature_maps[i] = torch.reshape(
        #         feat, (bs, num_cams) + feat.shape[1:]
        #     )
        # if return_depth and self.depth_branch is not None:
        #     depths = self.depth_branch(feature_maps, metas.get("focal"))
        # else:
        #     depths = None
        # if self.use_deformable_func:
        #     feature_maps = feature_maps_format(feature_maps)
        # if return_depth:
        #     return feature_maps, depths
        return feature_maps,None

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        batch = self.convert_batch(data)
        model_outs = self.head(feature_maps, batch)

        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, dec_qt = model_outs if self.training else model_outs[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            if 'temp_dn_pos_idx' in dn_meta:
                temp_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["temp_dn_num_split"], dim=2)
                temp_scores, dec_scores = torch.split(dec_scores, dn_meta["temp_dn_num_split"], dim=2)
                temp_qt, dec_qt = torch.split(dec_qt, dn_meta["temp_dn_num_split"], dim=2)
            else:
                temp_bboxes, temp_scores, temp_qt = None, None, None
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
            dn_qt, dec_qt = torch.split(dec_qt, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss, (idx, gt_idx) = self.head.loss((dec_bboxes, dec_scores), batch, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta, temp_info=(temp_bboxes,temp_scores), temp_match_indices = dn_meta.get('temp_match_indices', None), qt=(temp_qt, dn_qt, dec_qt))
        
        # cache query for temp denoising
        if hasattr(self.head.instance_bank, "cache"):
            self.head.instance_bank.cache((dec_bboxes, dec_scores), batch, idx, gt_idx, dn_info=(dn_bboxes, dn_scores), temp_info=(temp_bboxes,temp_scores), dn_meta=dn_meta)
        # loss = self.criterion
        #     (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        # )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        # return sum(loss.values()), torch.as_tensor(
        #     [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        # )


        
        # if depths is not None and "gt_depth" in data:
        #     output["loss_dense_depth"] = self.depth_branch.loss(
        #         depths, data["gt_depth"]
        #     )
        return loss

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps, _ = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
    
    def convert_batch(self, data):
        # convert batch data to RTDETRDecoder input format

        # batch.(['cls', 'bboxes', 'batch_idx', 'gt_groups'])
        cls_labels = torch.cat(data["gt_labels_2d"])
        batch_idx = torch.cat([torch.full_like(t, i) for i, t in enumerate(data['gt_labels_2d'])])
        gt_groups = [(batch_idx == i).sum().item() for i in range(len(data["gt_labels_2d"]))]
        bboxes = torch.cat(data["gt_bboxes_2d"])  # normalized bboxes
        instance_ids = np.concatenate([i["instance_id"] for i in data['img_metas']])
        instance_ids = torch.tensor(instance_ids, dtype=torch.int64).to(bboxes.device)
        wh = data['image_wh'][0][0]
        bboxes[:,[0,2]] /= wh[0]
        bboxes[:,[1,3]] /= wh[1]

        batch = {
            "cls": cls_labels,
            "bboxes": bboxes,
            "batch_idx": batch_idx,
            "gt_groups": gt_groups,
            "instance_ids": instance_ids,
            'timestamp': data['timestamp']
            }
        return batch
