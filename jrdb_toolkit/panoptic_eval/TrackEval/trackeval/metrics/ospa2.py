from ._base_metric import _BaseMetric
from .. import _timing
import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy


class OSPA2(_BaseMetric):
    def __init__(self):
        self.loss_fields = ['OSPA', 'OSPA_CARD', 'OSPA_LOC',
                            # 'OSPA_S', 'OSPA_M', 'OSPA_L'
                            ]
        self.size_level = {0: 'OSPA_S', 1: 'OSPA_M', 2: 'OSPA_L', 3: 'OSPA'}
        self.fields = self.loss_fields
        self.summary_fields = self.loss_fields

        self.integer_fields = []
        self.integer_array_fields = []
        self.float_array_fields = []
        self.float_fields = self.loss_fields

    # def eval_sequence(self, data):
    #     res = {}
    #     for field in self.fields:
    #         res[field] = 0
    #     gt_mask_sizes = deepcopy(data['gt_mask_sizes'])
    #     tracker_mask_sizes = deepcopy(data['tracker_mask_sizes'])
    #     similarity_scores = deepcopy(data['similarity_scores'])
    #
    #     dist_sum = {i: np.zeros((data['num_gt_ids'], data['num_tracker_ids'], data['num_timesteps'])) for i in range(4)}
    #     count_sum = {i: np.zeros((data['num_gt_ids'], data['num_tracker_ids'], data['num_timesteps'])) for i in range(4)}
    #     masks = {i: np.zeros((data['num_gt_ids'], data['num_tracker_ids'], data['num_timesteps'])) for i in range(4)}
    #
    #     for t, (gt_ids_t, tracker_ids_t,), in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
    #         if len(tracker_ids_t) == 0:
    #             continue
    #         for size_lvl in range(4):
    #             sim_score_t = similarity_scores[t]
    #             dist = (1 - sim_score_t)
    #             dist_t = np.zeros((data['num_gt_ids'], data['num_tracker_ids'],))
    #
    #             dist_t[gt_ids_t] = 1
    #             count_sum[size_lvl][gt_ids_t,:,t] += 1
    #             dist_t[:, tracker_ids_t] = 1
    #             count_sum[size_lvl][:, tracker_ids_t,t] += 1
    #             dist_t[gt_ids_t[:, None], tracker_ids_t,] = dist
    #             count_sum[size_lvl][gt_ids_t[:, None], tracker_ids_t,t] -= 1
    #             if size_lvl < 3:
    #                 # apply mask
    #                 gt_ids_t_lvl = gt_ids_t[gt_mask_sizes[t] == size_lvl]
    #                 tracker_ids_t_lvl = tracker_ids_t[tracker_mask_sizes[t] == size_lvl]
    #                 masks[size_lvl][gt_ids_t_lvl[:, None], tracker_ids_t_lvl,t] = 1
    #                 # dist_t[mask] = 1
    #                 # masks[size_lvl][gt_ids_t[:, None], tracker_ids_t,t] = mask
    #
    #             dist_sum[size_lvl][...,t] += dist_t
    #
    #     trk_dist = np.sum(dist_sum[3], axis=2) / np.sum(count_sum[3], axis=2)
    #     match_gt, match_trk = linear_sum_assignment(trk_dist)
    #
    #     for size_lvl in range(4):
    #         cost_per_occ = dist_sum[size_lvl] / count_sum[size_lvl]
    #         mask_lvl = masks[size_lvl].astype(bool)
    #         cost = np.sum(cost_per_occ[mask_lvl])
            # stuck here, the masks can change in sizes over time

    # def eval_sequence(self, data):
    #     res = {}
    #     for field in self.fields:
    #         res[field] = 0
    #     gt_mask_sizes = deepcopy(data['gt_mask_sizes'])
    #     tracker_mask_sizes = deepcopy(data['tracker_mask_sizes'])
    #     similarity_scores = deepcopy(data['similarity_scores'])
    #     dist_sum = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
    #     dist_per_occl = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
    #     count_sum = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
    #
    #     for t, (gt_ids_t, tracker_ids_t,), in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
    #         if len(tracker_ids_t) == 0:
    #             continue
    #         sim_score_t = similarity_scores[t]
    #         dist = (1 - sim_score_t)
    #         dist_t = np.zeros((data['num_gt_ids'], data['num_tracker_ids'],))
    #
    #         dist_t[gt_ids_t] = 1
    #         count_sum[gt_ids_t] += 1
    #         dist_t[:, tracker_ids_t] = 1
    #         count_sum[:, tracker_ids_t] += 1
    #         dist_t[gt_ids_t[:, None], tracker_ids_t,] = dist
    #         count_sum[gt_ids_t[:, None], tracker_ids_t,] -= 1
    #         dist_sum += dist_t
    #
    #     count_sum[count_sum == 0] = 1
    #     trk_dist = dist_sum / count_sum
    #     match_gt, match_trk = linear_sum_assignment(trk_dist)
    #     cost = np.sum(trk_dist[match_gt, match_trk]) # match_rows, match_cols
    #     m = data['num_gt_ids']
    #     n = data['num_tracker_ids']
    #     ospa2 = np.power(((1 * np.absolute(m - n) + cost) / max(m, n)), 1)
    #     term1 = np.absolute(m - n) / max(m, n)
    #     term2 = cost / max(m, n)
    #     res['OSPA']=ospa2
    #     res['OSPA_CARD']=term1
    #     res['OSPA_LOC']=term2
    #     return res
    def eval_sequence(self, data):
        res = {}
        for field in self.fields:
            res[field] = 0
        similarity_scores = deepcopy(data['similarity_scores'])
        dist_sum = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        count_sum = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))

        for t, (gt_ids_t, tracker_ids_t,), in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(tracker_ids_t) == 0:
                continue
            sim_score_t = similarity_scores[t]
            dist = (1 - sim_score_t)
            dist_t = np.zeros((data['num_gt_ids'], data['num_tracker_ids'],))

            dist_t[gt_ids_t] = 1
            count_sum[gt_ids_t] += 1
            dist_t[:, tracker_ids_t] = 1
            count_sum[:, tracker_ids_t] += 1
            dist_t[gt_ids_t[:, None], tracker_ids_t,] = dist
            count_sum[gt_ids_t[:, None], tracker_ids_t,] -= 1
            dist_sum += dist_t

        count_sum[count_sum == 0] = 1
        trk_dist = dist_sum / count_sum
        match_gt, match_trk = linear_sum_assignment(trk_dist)
        cost = np.sum(trk_dist[match_gt, match_trk]) # match_rows, match_cols
        m = data['num_gt_ids']
        n = data['num_tracker_ids']
        ospa2 = np.power(((1 * np.absolute(m - n) + cost) / max(m, n)), 1)
        term1 = np.absolute(m - n) / max(m, n)
        term2 = cost / max(m, n)
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

    @staticmethod
    def _combine_average(all_res, field):
        """Combine sequence results via sum"""
        results = []
        for seq in all_res.keys():
            res = all_res[seq][field]
            # check for nan
            if not np.isnan(res):
                results.append(res)

        # tmp = [all_res[k][field] for k in all_res.keys()]
        return np.mean(results)
    def _combine_sum(self, all_res, field):
        """Combine sequence results via sum"""
        results = []
        for seq in all_res.keys():
            res = all_res[seq][field]
            # check for nan
            if not np.isnan(res):
                results.append(res)

        return np.sum(results)

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
