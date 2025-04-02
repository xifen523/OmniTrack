
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from .ops import xywh2xyxy,xyxy2xywh
from .Omnidetr_loss import OmniDETRDetectionLoss, DETRLoss
from ..track.strack import STrack
from ..track.kalman_filter import KalmanFilter
from ..track.matching import iou_distance, iou_score
from .query_handler_module import QueryHandler
from .track_handler_module import TrackHandler

__all__ = ["InstanceBackOMNIDETR"]

def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBackOMNIDETR(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        track_thresh = 0.45,      # updata track when the score is greater than track_thresh
        det_thresh = 0.20,        # detection threshold for track
        nms_thresh = 0.05,      # nms threshold for track
        init_thresh = 0.55,  # init a track when the score is greater than init_thresh
        extend = False,
    ):
        super(InstanceBackOMNIDETR, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()
        self.track = None
        self.timestamp = None
        self.temp_count = 0
        self.prev_id = 0
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.starcks = []
        self.max_time_lost = 10
        self.extend = extend

        self.nms_thresh = nms_thresh
        self.track_thresh = track_thresh
        self.det_thresh = det_thresh
        self.init_thresh = init_thresh

        # tracking variables
        self.lost_stracks = []
        self.OCsort = False
        self.Hybridsort = False
        self.ByteTrack = False
        if self.OCsort or self.Hybridsort or self.ByteTrack:
            self.TBD = True
        else:
            self.TBD = False
        
        if self.TBD:
            if self.OCsort:
                self.instance_handler = TrackHandler(self)
            else:
                self.instance_handler = TrackHandler(self)
        else:
                self.instance_handler = QueryHandler(self)


    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size, metas=None, dn_metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )  # init with zeros   (bs, num_anchor, embed_dims)
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:   # project to current frame
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )  # long term memory will be dropped, if time_interval > max_time_interval
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    # def cache(
    #     self,
    #     instance_feature,
    #     anchor,
    #     confidence,
    #     metas=None,
    #     feature_maps=None,
    # ):
    #     if self.num_temp_instances <= 0:
    #         return
    #     instance_feature = instance_feature.detach()
    #     anchor = anchor.detach()
    #     confidence = confidence.detach()

    #     self.metas = metas
    #     confidence = confidence.max(dim=-1).values.sigmoid()
    #     if self.confidence is not None:
    #         confidence[:, : self.num_temp_instances] = torch.maximum(
    #             self.confidence * self.confidence_decay,
    #             confidence[:, : self.num_temp_instances],
    #         )
    #     self.temp_confidence = confidence

    #     (
    #         self.confidence,
    #         (self.cached_feature, self.cached_anchor),
    #     ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def cache(self, preds, batch, idx, gt_idx, dn_info=None, temp_info=None, dn_meta=None):
        """
        Args:
            preds: list, predicted results
            batch: dict, batch data
            idx: 
        """
        pred_bboxes, pred_scores = preds
        instance_ids = batch.get("instance_ids", None)

        # cache for temp query
        self.temp_instance_ids = instance_ids[gt_idx].detach()
        self.temp_pred_bboxes = pred_bboxes[-1][idx].detach()
        self.pred_scores = pred_scores[-1][idx].detach()
        self.gt_bboxes = batch["bboxes"][gt_idx].detach()
        self.batch_idx = batch["batch_idx"][gt_idx].detach()
        self.timestamp = batch['timestamp'].detach()
        self.gt_groups = batch['gt_groups']

        # maybe generate tracklets 
        if dn_meta is not None:
            dn_bboxes, dn_scores = dn_info

            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            dn_match = OmniDETRDetectionLoss.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])
            dn_idx, dn_gt_idx = DETRLoss._get_index(dn_match)
            pred_bboxes, pre_scores = dn_bboxes[-1][dn_idx], dn_scores[-1][dn_idx].detach()
            temp_gt_bboxes = batch["bboxes"][dn_gt_idx]
            temp_gt_instance_ids = instance_ids[dn_gt_idx]


            max_score, cls = pre_scores.sigmoid().max(-1, keepdim=True)
            group_num = [x * dn_num_group for x in batch["gt_groups"]]
            score_group = max_score.squeeze(-1).split(group_num, dim=0)
            dn_idx_group = dn_idx[1].split(group_num, dim=0)

            score_dn_idx_1 = []
            score_dn_idx_2 = []
            dn_score = []
            for i in range(len(score_group)):
                chunks = torch.chunk(score_group[i], dn_num_group, dim=0)
                score = torch.stack(chunks, dim=0)
                chunks = torch.chunk(dn_idx_group[i], dn_num_group, dim=0)
                idx_matrix = torch.stack(chunks, dim=0)
                max_scores, max_indices = torch.max(score, dim=0)
                column_indices = torch.arange(max_indices.size(0), device=max_indices.device)
                rows = max_indices.cpu()
                cols = column_indices.cpu()
                object_idx = idx_matrix[rows, cols]
                bs_idx = torch.tensor([i]*len(object_idx))

                score_dn_idx_1.append(bs_idx)
                score_dn_idx_2.append(object_idx)
                dn_score.append(max_scores)

            
            score_dn_idx_1 = torch.cat(score_dn_idx_1, dim=0)
            score_dn_idx_2 = torch.cat(score_dn_idx_2, dim=0)
            dn_score = torch.cat(dn_score, dim=0)
            score_dn_idx = (score_dn_idx_1, score_dn_idx_2)
            temp_max_query = dn_bboxes[-1][score_dn_idx]
            inv_gt_idx = torch.zeros_like(gt_idx)
            original_idx = torch.arange(gt_idx.size(0))
            inv_gt_idx[gt_idx] = original_idx

            
            self.temp_instance_ids = self.temp_instance_ids[inv_gt_idx].detach()
            self.temp_pred_bboxes = self.temp_pred_bboxes[inv_gt_idx].detach()
            self.pred_scores = self.pred_scores[inv_gt_idx].detach()
            self.gt_bboxes = self.gt_bboxes[inv_gt_idx].detach()
            
            mask = self.pred_scores.squeeze(-1) > dn_score
            self.temp_pred_bboxes = torch.where(mask[:, None], self.temp_pred_bboxes, temp_max_query)
            self.pred_scores = torch.where(mask, self.pred_scores.squeeze(-1), dn_score)



    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )

    def get_temp_group(
            self, batch, num_classes, num_queries, class_embed, temp_groups=3, id_noise_ratio=0.1, motion_noise_scale=0.5, training=False, atten_mask=None, dn_meta=None
    ):
        """
        Args:
            batch: dict, batch data
            num_classes: int, number of classes
            num_queries: int, number of queries
            class_embed: nn.Module, class embedding layer
            num_dn: int, number of distractor objects
            cls_noise_ratio: float, ratio of distractor classes
            box_noise_scale: float, scale of distractor boxes

        """
        self.temp_count +=1

        if (self.timestamp is None) or (not 'timestamp' in batch):
            return None, None, atten_mask, dn_meta

        cur_timestamp = batch['timestamp']
        inter_time = (cur_timestamp - self.timestamp)
        mask = torch.abs(inter_time) < 0.2   # 0.2s
        
        if self.training: # training mode
            if mask.sum() == 0 or len(batch['gt_groups'])!= len(self.gt_groups):
                return None, None, atten_mask, dn_meta
    
            # mask temp query if the instance is not in the current batch
            self._mask_temp_query(batch, mask)

            bs = len(batch['gt_groups'])
            total_temp_num = sum(batch['gt_groups'])
            num_temp_queries = []
            temp_indices_map = []
            idx_groups = torch.as_tensor([0, *batch['gt_groups'][:-1]]).cumsum_(0)
            temp_idx_groups = torch.as_tensor([0, *self.gt_groups[:-1]]).cumsum_(0)

            for bs_idx, (gt_num, temp_num) in enumerate(zip(batch['gt_groups'], self.gt_groups)):
                # num_temp_queries.append(len(self.temp_instance_ids[self.batch_idx == bs_idx]))
                num_temp_queries.append((self.batch_idx == bs_idx).sum().item())

                cur_bs_instance_ids = batch['instance_ids'][idx_groups[bs_idx]:idx_groups[bs_idx]+gt_num]
                temp_bs_instance_ids = self.temp_instance_ids[temp_idx_groups[bs_idx]:temp_idx_groups[bs_idx]+temp_num]
                temp_bs_instance_ids = temp_bs_instance_ids.to(cur_bs_instance_ids.device)

                cur_indices = []
                temp_indice = []            
                for cur_index,cur_id in enumerate(cur_bs_instance_ids):
                    if cur_id in temp_bs_instance_ids:
                        cur_indices.append(cur_index)
                        temp_indice.append(temp_bs_instance_ids.tolist().index(cur_id))


                temp_indices_map.append((torch.tensor(cur_indices), torch.tensor(temp_indice)))


            max_num_temp_queries = max(num_temp_queries)

            gt_cls = torch.cat([batch['cls'][bs[1]] for bs in temp_indices_map if bs[1].numel() > 0]).detach()
            instance_idx = self.temp_instance_ids
            # Each group has positive and negative queries.
            dn_temp_bbox = self.temp_pred_bboxes.repeat(2*temp_groups, 1).detach()
            dn_temp_b_idx = self.batch_idx.repeat(2*temp_groups).view(-1).detach()
            dn_temp_cls = gt_cls.repeat(2*temp_groups).detach()
            instance_idx = instance_idx.repeat(2*temp_groups).detach()

            # Positive and negative mask
            # (bs*num*num_group, ), the second total_num*num_group part as negative samples
            neg_idx = torch.arange(max_num_temp_queries * temp_groups, dtype=torch.long, device=dn_temp_bbox.device) + temp_groups * max_num_temp_queries
            
            # TODO  add id noise
            # if id_noise_ratio > 0:
            #     # id noise
            #     mask = torch.rand(instance_idx.shape) < id_noise_ratio
            #     idx = torch.nonzero(mask).squeeze(-1)

            #     # random select a new instance id
            #     new_ids = torch.randint(0, num_queries, size=idx.shape, device=instance_idx.device)
            #     instance_idx[idx] = new_ids

            if motion_noise_scale > 0:
                known_bbox = xywh2xyxy(dn_temp_bbox)

                diff = (dn_temp_bbox[..., 2:] * 0.5).repeat(1, 2) * motion_noise_scale  # 2*num_group*bs*num, 4

                rand_sign = torch.randint_like(dn_temp_bbox, 0, 2) * 2.0 - 1.0
                rand_part = torch.rand_like(dn_temp_bbox)
                rand_part[neg_idx] += 1.0
                rand_part *= rand_sign
                known_bbox += rand_part * diff
                known_bbox.clip_(min=0.0, max=1.0)
                dn_bbox = xyxy2xywh(known_bbox)
                dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # inverse sigmoid

            num_dn = int(max_num_temp_queries * 2 * temp_groups)  # total denoising queries

            # class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])
            dn_cls_embed = class_embed[dn_temp_cls]  # bs*num * 2 * num_group, 256
            padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=dn_temp_bbox.device)
            padding_bbox = torch.zeros(bs, num_dn, 4, device=dn_temp_bbox.device)

            
            map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in num_temp_queries])

            map_indices = torch.cat([map_indices + max_num_temp_queries * i for i in range(2 * temp_groups)])
            padding_cls[(dn_temp_b_idx, map_indices)] = dn_cls_embed
            padding_bbox[(dn_temp_b_idx, map_indices)] = dn_bbox


            tgt_size = num_dn + num_queries
            attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
            attn_mask1 = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
            # updata atten_mask
            if atten_mask is not None:
                dn_num_query = dn_meta['dn_num_split'][0]
                attn_mask[:num_dn, num_dn:dn_num_query+num_dn] = True    # MASK temp qurey to see dn query
                attn_mask[num_dn:, num_dn:] = atten_mask
            
            for i in range(temp_groups):
                if i == 0:               # first group
                    attn_mask[max_num_temp_queries * i : max_num_temp_queries * (i + 1), max_num_temp_queries * (i + 1) : num_dn] = True

                if i == temp_groups - 1:  # last group
                    attn_mask[max_num_temp_queries * i : max_num_temp_queries * (i + 1), : max_num_temp_queries * i] = True
                
                else:                     # middle group
                    attn_mask[max_num_temp_queries * i : max_num_temp_queries * (i + 1), max_num_temp_queries * (i + 1) : num_dn] = True
                    attn_mask[max_num_temp_queries * i : max_num_temp_queries * (i + 1), : max_num_temp_queries * i] = True

            # generate pos_idx for each group
            temp_pos_idx = []
            for i in range(bs):
                # pos_idx_ = torch.tensor([], dtype=torch.long, device=dn_temp_bbox.device)
                pos_idx_ = torch.tensor([], dtype=torch.long)
                for j in range(temp_groups):
                    pos_idx_ = torch.cat([pos_idx_, temp_indices_map[i][1] + max_num_temp_queries * j])
                temp_pos_idx.append(pos_idx_)

            temp_match_indices = []
            for i, num_gt in enumerate(self.gt_groups):
                if num_gt > 0:
                    gt_idx = temp_indices_map[i][0] + temp_idx_groups[i]
                    gt_idx = gt_idx.repeat(temp_groups)
                    assert len(temp_pos_idx[i]) == len(gt_idx), "Expected the same length, "
                    f"but got {len(temp_pos_idx[i])} and {len(gt_idx)} respectively."
                    temp_match_indices.append((temp_pos_idx[i], gt_idx))
                else:
                    temp_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))

            temp_dn_meta = {
                "temp_dn_pos_idx": temp_pos_idx,
                "temp_dn_num_group": temp_groups,
                "temp_dn_num_split": [num_dn, num_queries],
                "temp_match_indices": temp_match_indices,
            }
            temp_dn_meta.update(dn_meta)

            return (
                padding_cls.to(class_embed.device),
                padding_bbox.to(class_embed.device),
                attn_mask1.to(class_embed.device),
                temp_dn_meta,
            )
        
        else: # inference
            if not self.TBD:
                query = np.array([t.tlwh for t in self.starcks])
                if len(query)==0:
                    return None, None, atten_mask, dn_meta
                query = STrack.tlwh_to_cxcywh(query)
                num_track = len(query)
                cls = [t._cls for t in self.starcks]
                dn_cls_embed = class_embed[cls]
                bs = 1
            elif self.OCsort:
                query = np.array([t.get_state()[0] for t in self.starcks if t.time_since_update < 1])
                if len(query)==0:
                    return None, None, atten_mask, dn_meta
                query = STrack.tlwh_to_cxcywh(STrack.tlbr_to_tlwh_muti(query))
                query[:,[0,2]] /= batch['image_wh'][0][0][0].item()
                query[:,[1,3]] /= batch['image_wh'][0][0][1].item()
                num_track = len(query)
                cls = [0 for t in self.starcks if t.time_since_update < 1 ]
                dn_cls_embed = class_embed[cls]
                bs = 1

            elif self.Hybridsort:
                query = np.array([t.get_state()[0] for t in self.starcks if t.time_since_update < 1])
                if len(query)==0:
                    return None, None, atten_mask, dn_meta
                query = STrack.tlwh_to_cxcywh(STrack.tlbr_to_tlwh_muti(query[:, :4]))
                query[:,[0,2]] /= batch['image_wh'][0][0][0].item()
                query[:,[1,3]] /= batch['image_wh'][0][0][1].item()
                num_track = len(query)
                cls = [0 for t in self.starcks if t.time_since_update < 1 ]
                dn_cls_embed = class_embed[cls]
                bs = 1

            elif self.ByteTrack:
                query = np.array([t.tlwh for t in self.starcks if t.is_activated])
                if len(query)==0:
                    return None, None, atten_mask, dn_meta
                query = STrack.tlwh_to_cxcywh(query)
                query[:,[0,2]] /= batch['image_wh'][0][0][0].item()
                query[:,[1,3]] /= batch['image_wh'][0][0][1].item()
                num_track = len(query)
                cls = [0 for t in self.starcks if t.is_activated]
                dn_cls_embed = class_embed[cls]
                bs = 1

            padding_cls = torch.zeros(bs, num_track, dn_cls_embed.shape[-1], device=dn_cls_embed.device)
            padding_bbox = torch.zeros(bs, num_track, 4, device=dn_cls_embed.device)
            track_bbox = torch.logit(torch.Tensor(query), eps=1e-6).to(padding_bbox.device)

            map_indices = torch.arange(num_track, dtype=torch.long)
            padding_cls[(0, map_indices)] = dn_cls_embed
            padding_bbox[(0, map_indices)] = track_bbox  # inverse sigmoid

            attn_mask = None

            query_meta = {
                "query_num": num_track,
            }


            tgt_size = num_track + num_queries
            attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
            attn_mask[num_track:, :num_track] = True
            
            return (
                padding_cls.to(class_embed.device),
                padding_bbox.to(class_embed.device),
                attn_mask.to(class_embed.device),
                query_meta,
            )


        
    
    def _mask_temp_query(self, batch, vaild_mask):

        mask_instance = (self.temp_instance_ids[:, None] == batch['instance_ids']).any(dim=1)

        mask = torch.ones_like(self.temp_instance_ids, dtype=torch.bool)
        temp_idx_groups = torch.as_tensor([0, *self.gt_groups[:-1]]).cumsum_(0)
        idx_groups = torch.as_tensor([0, *batch['gt_groups'][:-1]]).cumsum_(0)

        for i, (gt_num, temp_num) in enumerate(zip(batch['gt_groups'], self.gt_groups)):
            
            # instance_id mask 
            cur_bs_instance_ids = batch['instance_ids'][idx_groups[i]:idx_groups[i]+gt_num]
            temp_bs_instance_ids = self.temp_instance_ids[temp_idx_groups[i]:temp_idx_groups[i]+temp_num]
            mask_instance = (temp_bs_instance_ids[:, None] == cur_bs_instance_ids).any(dim=1)

            # time mask
            mask[temp_idx_groups[i]:temp_idx_groups[i]+temp_num] = (vaild_mask[i] & mask_instance)
            
        bs = len(self.gt_groups)
        self.temp_instance_ids = self.temp_instance_ids[mask]
        self.temp_pred_bboxes = self.temp_pred_bboxes[mask]
        self.pred_scores = self.pred_scores[mask]
        self.gt_bboxes = self.gt_bboxes[mask]
        self.batch_idx = self.batch_idx[mask]
        self.gt_groups=[]
        for i in range(bs):
            num = len(self.batch_idx[self.batch_idx == i])
            if num > 0:
                self.gt_groups.append(num)
            else:
                self.gt_groups.append(0)


        # self.temp_instance_ids = instance_ids[gt_idx].detach()
        # self.temp_pred_bboxes = pred_bboxes[-1][idx].detach()
        # self.pred_scores =  pred_scores[-1][idx].detach()
        # self.gt_bboxes = batch["bboxes"][gt_idx].detach()
        # self.batch_idx = batch["batch_idx"][gt_idx].detach()
        # self.timestamp = batch['timestamp'].detach()

    def query_handler(self, bbox, score, meta, qt=None):
        return self.instance_handler.query_handler(bbox, score, meta, qt)
        


