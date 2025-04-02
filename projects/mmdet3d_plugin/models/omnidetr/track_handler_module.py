# query_handler_module.py
import argparse
import torch
import pickle
from torch import nn
import numpy as np
from .ops import xywh2xyxy,xyxy2xywh
from ..track.strack import STrack
from ..track.kalman_filter import KalmanFilter
from ..track.matching import iou_distance, iou_score
from ..trackers.ocsort_tracker.ocsort import OCSort
from ..trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
from ..trackers.sort_tracker.sort import Sort
from ..trackers.byte_tracker.byte_tracker import BYTETracker
from ..trackers.args import make_parser
from torchvision.ops import nms



class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackHandler:
    def __init__(self, instance_bank):
        self.instance_bank = instance_bank
        self.instance_bank.nms_thresh = 0.05
        self.instance_bank.track_thresh = 0.45
        self.instance_bank.det_thresh = 0.10
        self.instance_bank.init_thresh = 0.55

        # self.tracker = OCSort(det_thresh=0.6, iou_threshold=0.15, use_byte=True)
        args = make_parser().parse_args()
        self.tracker = Hybrid_Sort(args, det_thresh=args.track_thresh,
                                    iou_threshold=args.iou_thresh,
                                    asso_func=args.asso,
                                    delta_t=args.deltat,
                                    inertia=args.inertia,
                                    use_byte=args.use_byte)
        # args.track_thresh = 0.2
        # self.tracker = Sort(args, det_thresh=args.track_thresh)
        self.save_data = {}

    def __getattr__(self, name):
        # use instance_bank attr as normal
        if name in self.instance_bank.__dict__:
            return getattr(self.instance_bank, name)
        else:
            return getattr(self, name)
        
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
        mask = score > self.det_thresh
        bbox = bbox[mask.squeeze(-1)]
        bbox = STrack.cxcywh_to_tlbr_to_tensor(bbox)
        bbox[:,[0,2]] *= self.img_wh[0]
        bbox[:,[1,3]] *= self.img_wh[1]
        score = score[mask]
        bbox = bbox.to(dtype=torch.int, device=bbox.device)
        bbox = bbox.to(dtype=torch.float32, device=bbox.device)
        if meta['image_wh'][0][0][0]-meta['ori_shape'][1] > 0 :
            cx = (bbox[:,0] + bbox[:,2])/2
            mask = cx < meta['ori_shape'][1]
            bbox = bbox[mask]
            score = score[mask]
   
        keep_indices = nms(bbox, score, iou_threshold=0.35)
        bbox = bbox[keep_indices]
        score = score[keep_indices]

        dets = torch.cat([bbox, score.unsqueeze(1)], dim=1)

        count = 0
        if self.timestamp is None or abs(self.timestamp - meta['timestamp']) >100:
            args = make_parser().parse_args()
            self.tracker = Hybrid_Sort(args, det_thresh=args.track_thresh,
                                            iou_threshold=args.iou_thresh,
                                            asso_func=args.asso,
                                            delta_t=args.deltat,
                                            inertia=args.inertia,
                                            use_byte=args.use_byte)
            # args.track_thresh = 0.2
            # self.tracker = Sort(args, det_thresh=args.track_thresh)
            self.frame_id = 0
            # with open("/root/autodl-tmp/sparse4D_track/huang-2-2019-01-25_0.pkl", 'wb') as f:
            #     pickle.dump(self.save_data, f)
            # print("save!!!")


        results, tracklets = self.tracker.update(dets.cpu().numpy())

        # self.save_data.update(
        #     {
        #         f'{self.frame_id}_input': dets,
        #         f'{self.frame_id}_otput': results,
        #     }
        # )

        self.timestamp = meta['timestamp']
        self.frame_id += 1
        self.starcks = tracklets

        return results, []
        