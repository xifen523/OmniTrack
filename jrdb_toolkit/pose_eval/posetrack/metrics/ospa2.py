from ._base_metric import _BaseMetric
from .. import _timing
import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
class OSPA2(_BaseMetric):
    def __init__(self):
        self.joint_names = ['head',
                            'right eye',
                            'left eye',
                            'right shoulder',
                            'neck',
                            'left shoulder',
                            'right elbow',
                            'left elbow',
                            'tailbone',
                            'right hand',
                            'right hip',
                            'left hip',
                            'left hand',
                            'right knee',
                            'left knee',
                            'right foot',
                            'left foot']
        self.n_joints = 17
        self.loss_fields = ['OSPA', 'OSPA_CARD', 'OSPA_LOC','OSPA_INVI',
                            'OSPA_OCCL', 'OSPA_VIS']
        self.occl_level = {0:'OSPA_INVI', 1:'OSPA_OCCL', 2:'OSPA_VIS', 3:'OSPA'}
        self.occl_level_avg = {0:'OSPA_INVI_AVG', 1:'OSPA_OCCL_AVG', 2:'OSPA_VIS_AVG'}
        self.fields = self.loss_fields
        self.summary_fields = self.loss_fields

        self.integer_fields = []
        self.integer_array_fields = []
        self.float_array_fields = []
        self.float_fields = self.loss_fields


    def eval_sequence(self, data):
        res = {}
        for field in self.fields:
            res[field] = 0

        keypoint_visibilities = data['keypoint_visibilities']
        # for occ_lvl in range(4):
            # if occ_lvl < 3:
            #     similarity_scores = deepcopy(data['oks_kpts_sims'])
            # else:
        per_kpt_sim = deepcopy(data['oks_kpts_sims'])
        similarity_scores = deepcopy(data['similarity_scores'])
        dist_sum = {i: np.zeros((data['num_gt_ids'], data['num_tracker_ids'])) for i in range(4)}
        dist_per_occl = {i: np.zeros((data['num_gt_ids'], data['num_tracker_ids'])) for i in range(4)}

        counts = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))

        for t, (gt_ids_t, tracker_ids_t,), in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(tracker_ids_t) == 0:
                continue
            for occ_lvl in range(4):
                if occ_lvl < 3:
                    mask = np.stack([keypoint_visibilities[t]==occ_lvl]*len(tracker_ids_t), axis=1)
                    kpts_sim_t = per_kpt_sim[t] * mask
                    dist = (1 - kpts_sim_t) * mask
                    dist = np.sum(dist,axis=-1)/np.maximum(1,np.sum(dist>0,axis=-1))
                    dist_t = np.zeros((data['num_gt_ids'], data['num_tracker_ids'],))

                    dist_t[gt_ids_t] = 1
                    dist_t[:, tracker_ids_t] = 1
                    dist_t[gt_ids_t[:, None], tracker_ids_t] = dist
                    dist_sum[occ_lvl] += dist_t

                else:
                    dist = 1 - similarity_scores[t]
                    dist_t = np.zeros((data['num_gt_ids'],data['num_tracker_ids'],))

                    dist_t[gt_ids_t] = 1
                    counts[gt_ids_t] += 1
                    dist_t[:,tracker_ids_t] = 1
                    counts[:,tracker_ids_t] += 1

                    dist_t[gt_ids_t[:,None],tracker_ids_t] = dist
                    counts[gt_ids_t[:,None],tracker_ids_t] -= 1
                    dist_sum[occ_lvl] += dist_t

        counts[counts == 0] = 1
        trk_dist = dist_sum[3] / counts

        match_rows, match_cols, = linear_sum_assignment(trk_dist)
        for occ_lvl in range(4):
            cost_per_occ = dist_sum[occ_lvl] / counts
            cost = np.sum(cost_per_occ[match_rows, match_cols])
            m=data['num_gt_ids']
            n=data['num_tracker_ids']
            ospa2 = np.power(((1 * np.absolute(m-n) + cost) / max(m,n)), 1)
            term1 = np.absolute(m-n) / max(m,n)
            term2 = cost / max(m,n)
            res[self.occl_level[occ_lvl]]=ospa2
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
        tmp=[all_res[k][field] for k in all_res.keys()]
        return sum(tmp)/len(tmp)

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