import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES

from projects.mmdet3d_plugin.core.box2d import *


@LOSSES.register_module()
class SparseBox2DLoss(nn.Module):
    def __init__(
        self,
        loss_box,
        loss_centerness=None,
        loss_yawness=None,
    ):
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_box = build(loss_box, LOSSES)
        self.loss_cns = build(loss_centerness, LOSSES)
        self.loss_yns = build(loss_yawness, LOSSES)


    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        output = {}
        box_loss = self.loss_box(
            box, box_target, weight=weight, avg_factor=avg_factor
        )
        output[f"loss_box{suffix}"] = box_loss

        if quality is not None:
            cns = quality[..., CNS_2D]

            # NOTE: use bbox center to calculate quality loss
            box_target_center = box_target[..., [X_2D, Y_2D]]
            box_center = box[..., [X_2D, Y_2D]]
            cns_target = torch.norm( box_center - box_target_center, p=2, dim=-1)

            # cns_target = torch.norm(
            #     box_target[..., [X_2D, Y_2D]] - box[..., [X_2D, Y_2D]], p=2, dim=-1
            # )
            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
            output[f"loss_cns{suffix}"] = cns_loss

            # yns_target = (
            #     torch.nn.functional.cosine_similarity(
            #         box_target[..., [SIN_YAW, COS_YAW]],
            #         box[..., [SIN_YAW, COS_YAW]],
            #         dim=-1,
            #     )
            #     > 0
            # )
            # yns_target = yns_target.float()
            # yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
            # output[f"loss_yns{suffix}"] = yns_loss
        return output




@LOSSES.register_module()
class SparseBox2DLossIOU(nn.Module):
    def __init__(
        self,
        loss_box,
        loss_centerness=None,
    ):
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_box = build(loss_box, LOSSES)
        self.loss_cns = build(loss_centerness, LOSSES)



    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        output = {}
        # decode box w and h
        box_exp = box.clone()
        box_exp[..., [W_2D, H_2D]] = box_exp[..., [W_2D, H_2D]].exp()
        box_target_exp = box_target.clone()
        box_target_exp[..., [W_2D, H_2D]] = box_target_exp[..., [W_2D, H_2D]].exp()
        box_loss = self.loss_box(
            box_exp, box_target_exp, weight=weight, avg_factor=avg_factor
        )
        output[f"loss_box{suffix}"] = box_loss

        if quality is not None:
            cns = quality[..., CNS_2D]

            # NOTE: use bbox center to calculate quality loss
            box_target_center = box_target[..., [X_2D, Y_2D]] + (box_target_exp[..., [W_2D, H_2D]])/2
            box_center = box[..., [X_2D, Y_2D]]+ (box_exp[..., [W_2D, H_2D]])/2
            cns_target = torch.norm( box_center - box_target_center, p=2, dim=-1)

            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
            output[f"loss_cns{suffix}"] = cns_loss
        return output