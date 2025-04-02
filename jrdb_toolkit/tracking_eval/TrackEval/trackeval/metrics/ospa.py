from ._base_metric import _BaseMetric
from .. import _timing
import numpy as np
from scipy.optimize import linear_sum_assignment

class OSPA(_BaseMetric):
    """Class which simply counts the number of tracker and gt detections and ids."""
    def __init__(self, config=None):
        super().__init__()
        self.loss_fields = ['OSPA','OSPA_CARD','OSPA_LOC']
        self.fields = self.loss_fields
        self.summary_fields = self.fields

    @_timing.time
    def eval_sequence(self, data):
        """Returns counts for one sequence"""
        # Get results
        res = {}
        for field in self.fields:
            res[field] = 0

        counts = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        dist_sum = np.zeros((data['num_gt_ids'],data['num_tracker_ids']))
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):

            dist = 1 - data['similarity_scores'][t]
            dist_t = np.zeros((data['num_gt_ids'], data['num_tracker_ids'],))

            dist_t[gt_ids_t] = 1
            counts[gt_ids_t] += 1
            dist_t[:, tracker_ids_t] = 1
            counts[:, tracker_ids_t] += 1

            dist_t[gt_ids_t[:, None], tracker_ids_t] = dist
            counts[gt_ids_t[:, None], tracker_ids_t] -= 1

            dist_sum += dist_t

        counts[counts == 0] = 1
        trk_dist = dist_sum / counts

        match_rows, match_cols = linear_sum_assignment(trk_dist)
        cost=np.sum(trk_dist[match_rows,match_cols])
        m=data['num_gt_ids']
        n=data['num_tracker_ids']
        ospa2 = np.power(((1 * np.absolute(m-n) + cost) / max(m,n)), 1)
        term1 = np.absolute(m-n) / max(m,n)
        term2 = cost / max(m,n)
        res['OSPA']=ospa2
        res['OSPA_CARD']=term1
        res['OSPA_LOC']=term2
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.loss_fields:
            res[field] = self._combine_average(all_res, field)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=None):
        """Combines metrics across all classes by averaging over the class values"""
        res = {}
        for field in self.loss_fields:
            res[field] = self._combine_sum(all_res, field)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.loss_fields:
            res[field] = self._combine_sum(all_res, field)
        return res