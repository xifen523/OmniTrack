from ._base_metric import _BaseMetric
from .. import _timing
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

class DE(_BaseMetric):
    """Class which simply counts the number of tracker and gt detections and ids."""
    def __init__(self, config=None):
        super().__init__()
        self.loss_fields = ['EFE', 'EFE_CARD', 'EFE_LOC']
        self.fields = self.loss_fields
        self.summary_fields = self.fields

    @_timing.time
    def eval_sequence(self, data):
        """Returns counts for one sequence"""
        # Get results
        res = {}
        for field in self.fields:
            res[field] = 0

        m=data['num_gt_ids']
        n=data['num_tracker_ids']

        c = 5 # the penalty coefficient and cut_off distance

        counts_nop = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        dist_sum_nop = np.zeros((data['num_gt_ids'],data['num_tracker_ids']))

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):

            dist = - data['similarity_scores'][t]
            dist_t_nop = np.zeros((data['num_gt_ids'], data['num_tracker_ids'],))

            dist_c = np.minimum(dist,c) # cut_off distance

            dist_t_nop[gt_ids_t] = c
            counts_nop[gt_ids_t] += 1
            dist_t_nop[gt_ids_t[:, None], tracker_ids_t] = dist_c

            dist_sum_nop += dist_t_nop

        counts_nop[counts_nop == 0] = 1
        trk_dist_nop = dist_sum_nop / counts_nop
        match_rows, match_cols = linear_sum_assignment(trk_dist_nop)
        
        res['EFE_LOC'] = np.sum(trk_dist_nop[match_rows,match_cols]) / max(m,n)
        res['EFE_CARD'] = c * np.absolute(m-n) / max(m,n)
        res['EFE'] = res['EFE_LOC'] + res['EFE_CARD']

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