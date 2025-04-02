from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
import numpy as np
import torch

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, cls=0, qt=0):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self._cls = int(cls)
        self.qt = qt

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 0:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.qt = new_track.qt

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    # @jit(nopython=True)
    def tlbr_(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh_muti(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[:, 2:] -= ret[:, :2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    # @jit(nopython=True)
    def cxcywh_to_tlwh(cxcywh):
        ret = np.asarray(cxcywh.cpu().numpy()).copy()
        ret[:, :2] -= ret[:, 2:] / 2
        return ret
    

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_cxcywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:, :2] += ret[:, 2:] / 2
        return ret


    @staticmethod
    # @jit(nopython=True)
    def cxcywh_to_tlbr_to_tensor(cxcywh):
        # tensor(N, 4) cx, cy, w, h  --> tensor(N, 4) tlbr  (x1, y1, x2, y2)
        wh = cxcywh[:, 2:].clone()
        tlbr = torch.cat((cxcywh[:, :2] - wh / 2, cxcywh[:, :2] + wh / 2), dim=1)
        return tlbr

    @staticmethod
    # @jit(nopython=True)
    def cxcywh_to_tlbr(cxcywh):
        ret = np.asarray(cxcywh.cpu().numpy()).copy()
        ret[:, :2] -= ret[:, 2:] / 2
        ret[:, 2:] += ret[:, :2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def ltwh_to_tlbr_to_tensor(tlwh):
        # tensor(N, 4) left, top, width, height  --> tensor(N, 4) tlbr  (x1, y1, x2, y2)
        if isinstance(tlwh, torch.Tensor):
            ret = torch.cat((tlwh[:, :2], tlwh[:, :2] + tlwh[:, 2:]), dim=1)
        else:
            ret = torch.tensor(tlwh)
            ret = torch.cat((ret[:, :2], ret[:, :2] + ret[:, 2:]), dim=1)
        return ret


    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)