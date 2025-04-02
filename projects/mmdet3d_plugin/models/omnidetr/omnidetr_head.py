
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmcv.cnn.bricks.registry import (PLUGIN_LAYERS,)
from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer

from .utils import bias_init_with_prob, linear_init
from .ops import convert_torch2numpy_batch, xywh2xyxy
import projects.mmdet3d_plugin.models.trackers.byte_tracker.byte_tracker as byte_tracker

from mmdet.models import (
    build_loss,
)


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

@HEADS.register_module()
class OmniETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        loss=dict(type='RTDETRDetectionLoss'),
        post_conf = 0.1,
        classes = None,
        sampler: dict = None,
        instance_bank: dict = None,
        temp_group_queries: int = 3, # temp group queries
        id_noise_ratio: float = 0.1,  # id noise ratio
        motion_noise_scale = 0.5,  # motion noise scale
        with_quality_estimation = True,
        temp_query = False,
        is_track = True,

    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_head = nn.ModuleList([nn.Sequential(*linear_relu_ln(hd, 1, 2), Linear(hd, 1),) for _ in range(ndl)])

        self._reset_parameters()

        self.loss = build_loss(loss)
        self.post_conf = post_conf
        self.classes = classes        
        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.temp_group_queries = temp_group_queries
        self.id_noise_ratio = id_noise_ratio
        self.motion_noise_scale = motion_noise_scale
        self.is_track = is_track
        self.temp_query = temp_query 

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from .ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        if hasattr(self.instance_bank, "get_temp_group") and self.temp_query :
            temp_embed, temp_motion_bbox, attn_mask, dn_meta = self.instance_bank.get_temp_group(
                    batch,
                    self.nc,
                    attn_mask.shape[0] if attn_mask is not None else self.num_queries, # current num queries
                    self.denoising_class_embed.weight,
                    self.temp_group_queries, # temp group queries
                    self.id_noise_ratio,
                    self.motion_noise_scale,
                    self.training,
                    attn_mask,
                    dn_meta
                )
        else:
            temp_embed = temp_motion_bbox = None

            
        # embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox, temp_embed, temp_motion_bbox)

        # Decoder
        dec_bboxes, dec_scores, quality = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            quality_head=self.quality_head if self.with_quality_estimation else None,
        )

        #cache temp bbox

        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, quality
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid(), quality.squeeze(0)), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            # gid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None, temp_embed=None, temp_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        if temp_embed is not None:
            refer_bbox = torch.cat([temp_bbox, refer_bbox], 1)
            embeddings = torch.cat([temp_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


    def post_process(self, preds, metas): #meta
        """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`.

        Args:
            preds (list): List of [predictions, extra] from the model.
            orig_imgs (list or torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
                and class labels.
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        nd = preds[0].shape[-1]
        if nd == 6:
            bboxes, scores, quality = preds[0].split((4, 1, 1), dim=-1)
            quality = quality.sigmoid()
        else:
            bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
            quality = None

        results = []
        if not self.is_track:
            for i , (bbox, score) in enumerate(zip(bboxes, scores)):  # (300, 4)
                bbox = xywh2xyxy(bbox)
                max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
                idx = max_score.squeeze(-1) > self.post_conf  # (300, )
                # if self.classes is not None:
                #     idx = (cls == torch.tensor(self.classes, device=cls.device)).any(1) & idx
                pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]  # filter
                score = score[idx]
                cls = cls[idx]
                ow, oh = metas['image_wh'][i][0]
                # ori_w = metas['ori_shape'][1]
                # extend = ow - ori_w
                
                pred[..., [0, 2]] *= ow
                pred[..., [1, 3]] *= oh
                # to w h
                pred[..., [2, 3]] -= pred[..., [0, 1]]
                # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
                qt = quality[i][idx] if quality is not None else None

                # mask box over right border
                if ow-metas['ori_shape'][1] > 0 :
                    cx = pred[:,0] + pred[:,2]/2
                    mask = cx < metas['ori_shape'][1]
                    pred = pred[mask]
                    score = score[mask]
                    cls = cls[mask]
                    qt = qt[mask] if qt is not None else None
                results.append(
                        {
                    "boxes_2d": pred[:,:4].cpu(), # cx,cy w, h
                    "scores_2d": score.cpu(), 
                    "labels_2d": cls.cpu(),
                    "quality": qt.cpu(),
                    }
                )
            
        if hasattr(self.instance_bank, "query_handler") and self.is_track:
            online_targets, dets = self.instance_bank.query_handler(bboxes, scores, metas, quality)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            online_qt = []
            results = []
            ow, oh = metas['image_wh'][0][0]
            ow = ow.cpu().numpy()
            oh = oh.cpu().numpy()
            for t in online_targets:
                if isinstance(t, np.ndarray):
                    tlwh = [t[0], t[1], t[2]-t[0], t[3]-t[1]]
                    online_tlwhs.append(tlwh)
                    online_ids.append(t[4])
                    online_scores.append(1)
                    online_cls.append(0)
                    online_qt.append(0)

                else:
                    tlwh = t.tlwh
                    if not isinstance(t, byte_tracker.STrack):
                        tlwh[[0, 2]] *= ow
                        tlwh[[1, 3]] *= oh

                    online_tlwhs.append(tlwh)
                    online_ids.append(t.track_id)
                    online_scores.append(t.score)
                    if hasattr(t, "_cls"):
                        online_cls.append(t._cls)
                    else:
                        online_cls.append(0)
                    if hasattr(t, "qt"):
                        online_qt.append(t.qt)
                    else:
                        online_qt.append(0)
            
            det_tlwhs = []
            det_scores = []
            if dets is not None:
                for d in dets:
                    tlwh = d.tlwh
                    tlwh[[0, 2]] *= ow
                    tlwh[[1, 3]] *= oh
                    det_tlwhs.append(tlwh)
                    det_scores.append(d.score)


            results.append(
                    {
                "boxes_2d": online_tlwhs, 
                "scores_2d": online_scores, 
                "labels_2d": online_cls,
                "instance_ids": online_ids,
                "quality": online_qt,
                "det_boxes_2d": det_tlwhs, 
                "det_scores_2d": det_scores, 
                }
            )

                

        return results