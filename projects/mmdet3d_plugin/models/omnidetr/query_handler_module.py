# query_handler_module.py
import torch
from torch import nn
import numpy as np
from .ops import xywh2xyxy,xyxy2xywh
from ..track.strack import STrack
from ..track.kalman_filter import KalmanFilter
from ..track.matching import iou_distance, iou_score

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class QueryHandler:
    def __init__(self, instance_bank):
        self.instance_bank = instance_bank
        # self.instance_bank.nms_thresh = 0.05
        # self.instance_bank.track_thresh = 0.5
        # self.instance_bank.det_thresh = 0.20
        # self.instance_bank.init_thresh = 0.9

        self.instance_bank.nms_thresh = 0.05
        self.instance_bank.init_thresh = 0.315
        self.instance_bank.det_thresh = 0.10
        self.instance_bank.track_thresh = 0.39

    def __getattr__(self, name):
        # 
        return getattr(self.instance_bank, name)

    def __setattr__(self, name, value):
        # 
        if name != 'instance_bank':
            setattr(self.instance_bank, name, value)
        else:
            super().__setattr__(name, value)

    def query_handler(self, bbox, score, meta, qt):
        self.img_wh = meta['image_wh'][0][0].cpu().numpy()
        self.ori_shape = np.array([meta['ori_shape'][1][0].cpu().numpy(), meta['ori_shape'][0][0].cpu().numpy()])
        scale_w = self.ori_shape[0] / self.img_wh[0]


        if self.timestamp is None or abs(self.timestamp - meta['timestamp']) >100:
            self.frame_id = 0
        from torchvision.ops import nms
        """ Step 1: initialize tracks"""
        if self.frame_id == 0:
            mask = score > self.det_thresh
            bbox = bbox[mask.squeeze(-1)]
            score = score[mask]
            bbox_tensor = STrack.cxcywh_to_tlbr_to_tensor(bbox)
            keep_indices = nms(bbox_tensor, score, iou_threshold=0.5)  # nms 
            bbox = bbox[keep_indices]
            score = score[keep_indices]

            bbox = STrack.cxcywh_to_tlwh(bbox,)
            cls = torch.zeros_like(score)

            
            track = [STrack(tlwh, s.cpu().numpy(), c.cpu().numpy()) for (tlwh, s, c) in zip(bbox, score, cls)]
            for t in track:
                t.activate(self.kalman_filter, self.frame_id)
                self.starcks.append(t)

            detections = track

        else:
            """ Step 2: Init new stracks"""
            strack_pool = [t for t in self.starcks]
            STrack.multi_predict(strack_pool)
            num_track = len(strack_pool)
            # predict the current location with KF
            track_bboxes, query_bboxes = torch.split(bbox, [num_track, bbox.size(1) - num_track], dim=1)
            track_scores, query_scores = torch.split(score, [num_track, score.size(1) - num_track], dim=1)
            if qt is not None:
                track_qt, query_qt = torch.split(qt, [num_track, qt.size(1) - num_track], dim=1)
            
            tracks = []
            lost_stracks = []
            # update the track with the predicted location

            track_scores = track_scores.squeeze(0)
            track_qt = track_qt.squeeze(0)
            track_bboxes = track_bboxes.squeeze(0)
            
            # nms track   due to  query may be overlap with track
            track_bboxes_ltbr = STrack.cxcywh_to_tlbr_to_tensor(track_bboxes)
            iou_s = iou_score(STrack.cxcywh_to_tlbr(track_bboxes), [t.tlbr for t in strack_pool] )  # IOU score
            refind_score = track_scores.squeeze(1) * torch.from_numpy(iou_s.copy()).to(track_scores.device)
            
            keep_indices = nms(track_bboxes_ltbr, refind_score, iou_threshold=0.5)  # nms 
            remove_indices = [i for i in range(len(track_scores)) if i not in keep_indices]
            lost_stracks = [strack_pool[i] for i in remove_indices]
            strack_pool = [strack_pool[i] for i in keep_indices]
            track_bboxes = track_bboxes[keep_indices]
            track_scores = track_scores[keep_indices]
            cls = torch.zeros_like(track_scores)
            track_qt = track_qt[keep_indices]


            # update track
            track_bboxes = STrack.cxcywh_to_tlwh(track_bboxes)
            track_bboxes = [STrack(tlwh, s.cpu().numpy(), c.cpu().numpy(), qt.cpu().numpy()) for (tlwh, s, c, qt) in zip(track_bboxes, track_scores, cls, track_qt)]
            
            mask = query_scores > self.det_thresh
            score = query_scores[mask]
            det_bbox = query_bboxes[mask.squeeze(-1)]


            # filter ovelap bbox and low score bbox
            cls = torch.zeros_like(score)
            det_bbox = STrack.cxcywh_to_tlwh(det_bbox)
            query_qt = query_qt.squeeze(0)
            detections = [STrack(tlwh, s.cpu().numpy(), c.cpu().numpy(), qt.cpu().numpy()) for (tlwh, s, c, qt) in zip(det_bbox, score, cls, query_qt)]
            
            for t, d in zip(strack_pool, track_bboxes):
                # conf = d.score * d.qt if d.qt > 0 else d.score
                conf = d.score 
                if conf > self.track_thresh:
                # if t.score - d.score < 0.2:
                    t.update(d, self.frame_id)
                    tracks.append(t)
                else:
                    lost_stracks.append(t)
 
            # step 2.1: init new track
            # remove detections which is overlap with track query
            if len(strack_pool) > 0 and len(detections) > 0:
                # detection nms
                dists = iou_distance(detections, strack_pool)
                min_values = np.min(dists, axis=1)
                mask = min_values > self.nms_thresh
                u_detection = [detections[i] for i in range(len(detections)) if mask[i]]
            
            else:
                u_detection = detections
            
            if len(u_detection) > 0: # init new tracks
                # nms detections
                u_bbox_ltbr = STrack.ltwh_to_tlbr_to_tensor(np.array([d.tlwh for d in u_detection]))
                score = torch.tensor(np.array([d.score for d in u_detection]), dtype=u_bbox_ltbr.dtype)
                keep_ub = nms(u_bbox_ltbr, score, iou_threshold=0.5).numpy()
               
                u_bbox = np.array([d.tlwh for d in u_detection]) 
                u_bbox = np.asarray(u_bbox, dtype=np.float32) 
                center_x = u_bbox[:, 0] + u_bbox[:, 2] / 2
                mask = center_x < scale_w
                u_detection = [u for i, u in enumerate(u_detection) if mask[i] and i in keep_ub]   # mask the bbox in the overlap area of track
                for det in u_detection:
                    if det.score > self.init_thresh:
                        det.activate(self.kalman_filter, self.frame_id)
                        tracks.append(det)

            # Step 3: remove lost tracks
            keep_tracks = []
            for t in lost_stracks:
                if self.frame_id - t.end_frame < self.max_time_lost:
                    t.mark_lost()
                    keep_tracks.append(t)  

            # tracks = joint_stracks(tracks, keep_tracks)
            self.starcks = tracks

        self.timestamp = meta['timestamp']
        self.frame_id += 1

        output_stracks = [track for track in self.starcks if track.is_activated and track.state != TrackState.Lost]
        return output_stracks, detections
    

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb