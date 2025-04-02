import os
import sys
import pickle as pkl
import time
import numpy as np


from tqdm import tqdm

class DanceTrackConverter(object):
    def __init__(self, data_root: str, save_root: str):
        self.data_root = data_root
        self.dancetrack_root = os.path.join(data_root, 'dancetrack')
        self.save_root = save_root
        # current time in microseconds
        self.cur_time = int(time.time() * 1_000_000)
        self.global_instance_id = 0

        self.test_pkl=[]
        self.train_pkl=[]
        self.val_pkl=[]

        
        self.sequences = {}
        for folder in ['train', 'test', 'val']:
            seq_names = os.listdir(os.path.join(self.dancetrack_root, folder))
            seq_names = [os.path.join(folder, seq_name) for seq_name in seq_names if ".DS_Store" not in seq_name]
            info = {folder: seq_names}
            self.sequences.update(info)
        
        

        
    def generate_pkl(self, type: str=None, save_root: str=None):
        '''
        generate pkl file for dancetrack dataset
        :param type: 'train', 'test', 'val'
        :param save_root: save path
        :return: None
        '''

        # check type
        if type is None:
            types = ['train', 'test', 'val']
        elif isinstance(type, str) and type in ['train', 'test', 'val']:
            types = [type]
        else:
            raise ValueError('type should be "train", "test", "val" or None')
        
        
        # handle info
        for t in types:
            self.sequences[t].sort()
            seq_names = self.sequences[t]
            for seq_name in tqdm(seq_names, desc=f"handle {t} sequence files"):
                folder = os.path.join(self.dancetrack_root, seq_name)
                if not os.path.isdir(folder):
                    continue
                self.handle_seq(folder,t)

            self.save_pkl(save_root)


    def handle_frame(self, frame_path: str, frame_id: int, seq_id: int, seqinfo: dict, gt_dict: dict) -> dict:
        
        # handle frame
        interval = int(1/float(seqinfo['frameRate'])*1000*1000)  # microseconds
        
        timestamp_us = self.cur_time + seq_id * 600_000_000 + frame_id*interval

        # # 将16位时间戳转换为秒和微秒
        # timestamp_s = timestamp_us // 1_000_000
        # microseconds = timestamp_us % 1_000_000

        # # 转换为datetime对象
        # from datetime import datetime, timedelta
        # # 转换为datetime对象
        # dt_object = datetime.fromtimestamp(timestamp_s) + timedelta(microseconds=microseconds)

        # # 格式化为字符串
        # readable_time = dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')
        # print(readable_time)
        

        if gt_dict is not None and frame_id in gt_dict:
            instance_ids = [gt_dict[frame_id][i][1] for i in range(len(gt_dict[frame_id]))]
            instance_ids = np.array(instance_ids) + self.global_instance_id
            bbox_ltwh = [gt_dict[frame_id][i][2:6] for i in range(len(gt_dict[frame_id]))]
            bbox_ltwh = np.array(bbox_ltwh)
            gt_names = np.array(['pedestrian'] * len(gt_dict[frame_id]))
        else:
            instance_ids = []
            bbox_ltwh = []
            gt_names = []


        cams = { "CAM_FRONT": 
                {
            'data_path': frame_path, 
            'type': "CAM_FRONT", 
            'sample_data_token': str(timestamp_us), 
            'timestamp': timestamp_us, 
            }  
        }

        infos_frame = {
            'cams': cams,
            'sweeps': [],
            

            'gt_boxes': bbox_ltwh,
            'gt_names': gt_names,
            'gt_velocity': [],
            'instance_inds': instance_ids,

            'timestamp': timestamp_us,
            'token': str(timestamp_us),
            'lidar_path': '',
        }

        return infos_frame

    def handle_seq(self, folder: str, type: str) -> dict:
        '''
        handle one sequence
        :param folder: sequence folder path
        :param type: 'train', 'test', 'val'
        :return: None
        '''

        img1 = os.path.join(folder, 'img1')
        # load seqinfo
        seqinfo_path = os.path.join(folder,'seqinfo.ini')
        seqinfo = self.load_seqinfo(seqinfo_path)

        # load gt
        if type=='train' or type=='val':
            gt = os.path.join(folder, 'gt/gt.txt')
            gt_dict = self.load_gt(gt)
        else:
            gt_dict = {}
        

        imgs = os.listdir(img1)
        imgs.sort()
        imgs = [os.path.join(img1, img) for img in imgs]

        # handle frames
        for img_path in imgs:
            relative_path = os.path.relpath(img_path, self.data_root)
            final_path = os.path.join("./data", relative_path)
            seq_name = os.path.basename(folder)
            seq_name_id = int(seq_name.split('dancetrack')[1])
            frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
            frame_dict = self.handle_frame(final_path, frame_id, seq_name_id, seqinfo, gt_dict)

            if type=='train':
                self.train_pkl.append(frame_dict)
            elif type=='test':
                self.test_pkl.append(frame_dict)
            elif type=='val':
                self.val_pkl.append(frame_dict)
            else:
                raise ValueError('type should be "train", "test", "val"')
        

        # update global instance id
        if type=='train' or type=='val':
            # lines = [gt_dict[frame_id][i]  for  i in range(len(gt_dict[frame_id])) for frame_id in gt_dict]
            lines = [gt_dict[frame_id] for frame_id in gt_dict]
            lines = [item for sublist in lines for item in sublist]
            lines = np.array(lines)
            # calculate max instance id in gt
            max_id = np.max(lines[:, 1])
            self.global_instance_id = self.global_instance_id + max_id + 1


    def load_seqinfo(self, ini_path: str) -> dict:
        '''
        load seqinfo.ini file
        :param ini_path: seqinfo.ini path
        :return: seqinfo dict
        '''

        seqinfo = {}
        with open(ini_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                seqinfo[key] = value
        return seqinfo

    def load_gt(self, gt_path: str) -> dict:
        '''
        load gt file
        :param gt_path: gt file path
        :return: gt dict
        '''

        gt = {}

        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):  # comment line
                    continue

                frame_id, obj_id, bb_left, bb_top, bb_width, bb_height, _, _, _ = line.split(',')
                frame_id = int(frame_id)
                obj_id = int(obj_id)
                bb_left = float(bb_left)
                bb_top = float(bb_top)
                bb_width = float(bb_width)
                bb_height = float(bb_height)

                if frame_id not in gt:
                    gt[frame_id] = []
                gt[frame_id].append([frame_id, obj_id, bb_left, bb_top, bb_width, bb_height])

        return gt

    @staticmethod
    def _save_pkl(data, filename, save_root):
        if len(data) > 0:
            info = {
                'infos': data,
                'metadata': dict(version='dancetrack_v1.0')
            }
            with open(os.path.join(save_root, filename), 'wb') as f:
                pkl.dump(info, f)
                print(f"save {filename}")

    def save_pkl(self, save_root: str=None):

        '''
        save pkl file
        :param save_root: save path
        :return: None
        '''
        if save_root is None:
            save_root = self.save_root

        if not os.path.exists(save_root):
            os.makedirs(save_root)

        self._save_pkl(self.train_pkl, 'dancetrack_infos_train.pkl', save_root)
        self._save_pkl(self.test_pkl, 'dancetrack_infos_test.pkl', save_root)
        self._save_pkl(self.val_pkl, 'dancetrack_infos_val.pkl', save_root)




if __name__ == '__main__':
    data_root = '/home/lk/workspase/python/dection/Sparse4D/data'
    save_root = '/home/lk/workspase/python/dection/Sparse4D/data/dancetrack_anno_pkls'
    danctrack=DanceTrackConverter(data_root, save_root)

    danctrack.generate_pkl()
    print("train pkl len:",len(danctrack.train_pkl))
    print("test pkl len:",len(danctrack.test_pkl))
    print("val pkl len:",len(danctrack.val_pkl))

    print("save train pkl")