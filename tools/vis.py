import cv2
import json
import numpy as np
import os


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

class Visualizer(object):
    def __init__(self, dict_path, data_root, save_dr=None):
        self.dict_path = dict_path
        # self.data = json.load(open(dict_path, 'r'))['results']
        self.data = self.read_pkl(dict_path)
        self.data_root = data_root
        self.save_dr = save_dr
        self.bbox = None

    def draw_bbox(self, img, bboxes, color=(0, 255, 0), thickness=2, ltwh=False, track_id=None, score=None, qt=False):

        if ltwh:
            bboxes = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for bbox in bboxes]
            bboxes = [[bbox[0]+(bbox[2]-bbox[0])/2, bbox[1]+(bbox[3]-bbox[1])/2, bbox[2]+(bbox[2]-bbox[0])/2, bbox[3]+(bbox[3]-bbox[1])/2] for bbox in bboxes]
            

        for idx, bbox in enumerate(bboxes):
            if track_id[idx] != "None" :
                # alph = 0.5
                if track_id[idx] in [2, 22, 15, 10, 16, 44, 49]:
                    continue
                color = get_color(abs(int(track_id[idx])))
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)
                # pass
            else:
                # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), thickness)
                continue
                

            if track_id is not None:
                if track_id[idx] != "None" :
                    cv2.putText(img, f"{track_id[idx]}", (int(bbox[0]+5), int(bbox[1])+65), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

            # if score is not None and qt is not None:
            #     # cv2.putText(img, f"{score[idx]:0.2f}/{qt[idx]:0.3f}", (int(bbox[0]), int(bbox[3])-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            #     cv2.putText(img, f"{score[idx]:0.2f}", (int(bbox[0]), int(bbox[3])-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            # else :
            #     if score is not None:
            #         cv2.putText(img, f"{score[idx]:0.2f}", (int(bbox[0]), int(bbox[3])-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
            #     if qt is not None:
            #         cv2.putText(img, f"{qt[idx]:0.3f}", (int(bbox[0]), int(bbox[3])+13), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

      
            # if track_id[idx] == '2':
            #     if self.bbox is not None:
            #         cv2.rectangle(img, (int(self.bbox[0]), int(self.bbox[1])), (int(self.bbox[2]), int(self.bbox[3])), (200, 125, 70), 2)
            #     self.bbox = bbox
            

        return img

    def read_pkl(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


    def vis_img(self,sorce_thd=0.1,seq=None):
        for K, v in enumerate(self.data):
            # seq_name,frame = k.rsplit('_', 1)
            # if seq_name!=seq and seq_name is not None:
            #     continue
            K = K+1
            frame = f"{K:06d}"
            boxes_2d = [i.tolist() for i in v['img_bbox']['boxes_2d']]
            track_id = v['img_bbox']['instance_ids']
            score = [i.item() for i in v['img_bbox']['scores_2d']]

            # bboxes = [[*i['x1y1'],*i['size']] for i in v if i['detection_score']>sorce_thd or np.isnan(v[-1]['detection_score'])]
            # track_id = [i['tracking_id'] for i in v if i['detection_score']>sorce_thd or np.isnan(v[-1]['detection_score'])]
            # score = [i['detection_score'] for i in v if i['detection_score']>sorce_thd or np.isnan(v[-1]['detection_score'])]
            # qt = [i['cls_score'] for i in v if i['detection_score']>sorce_thd or np.isnan(v[-1]['detection_score'])]

            img = cv2.imread(f"{self.data_root}/{frame}.jpg")
            img = self.draw_bbox(img, boxes_2d, color=(0, 255, 0), thickness=2, ltwh=True, track_id=track_id, score=score)

            if not os.path.exists(os.path.join(self.save_dr, seq_name)):
                os.makedirs(os.path.join(self.save_dr, seq_name))
            cv2.imwrite(os.path.join(self.save_dr, seq_name,  f"{frame}_{sorce_thd}.jpg"), img)
            # cv2.imshow(f"{seq_name}_{frame}_{sorce_thd}", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

if __name__=="__main__":
    # dict_path = "/home/lk/workspase/python/dection/Sparse4D/json/results_jrdb2d.json"
    # dict_path = "./results/submission/results_jrdb2d_track.json"
    # dict_path = "./results/submission/results_jrdb2d.json"
    dict_path = "/root/autodl-tmp/sparse4D_track/results.pkl"

    root_path = "/root/autodl-tmp/sparse4D_track/data/MOT20_v2/test/MOT20-07/img1"
    save_dir = "./results/vis"
    vis = Visualizer(dict_path, root_path, save_dir)
    seq_name = "MOT20-07"
    vis.vis_img(seq=seq_name, sorce_thd=0.3)
