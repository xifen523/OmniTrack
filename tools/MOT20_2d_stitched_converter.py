import os
import sys
import json
import yaml
import pickle as pkl
import time
import pyquaternion
import numpy as np
import pycocotools

from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask


# use for validation
validation = [
    "clark-center-2019-02-28_1",
    "gates-ai-lab-2019-02-08_0",
    "huang-2-2019-01-25_0",
    "meyer-green-2019-03-16_0",
    "nvidia-aud-2019-04-18_0",
    "tressider-2019-03-16_1",
    "tressider-2019-04-26_2"
]

# file 
FILE = ['calibration','images', 'labels', 'pointclouds', 'timestamps']



class MOT20Converter2D(object):
    '''
    Convert MOT20 dataset to Nuscenes format pkl file.
    '''
    def __init__(self, data_root: str, save_root: str, is_val: bool=False): 
        self.data_root = data_root
        self.MOT20_root = os.path.join(data_root, 'MOT20')
        self.save_root = save_root
        self.is_val = is_val

        self.test_pkl=[]
        self.train_pkl=[]
        self.val_pkl=[]

        self.__version__ = '1.2'
        self.global_id = 0

     
        self.info = {}
        for i in ['train', 'test']:
            # folder = "jrdb19_train_norosbag/train_dataset_with_activity" if i=="train"  else "jrdb19_test_norosbag/test_dataset_without_labels"
            folder = "train" if i=="train"  else "test"
            mot_folder = os.listdir(os.path.join(self.MOT20_root, folder))
            info = {i: mot_folder}
            self.info.update(info)


    def _load_json(self, json_path: str) -> dict:
        '''
        load json file
        :param json_path: json file path
        :return: json data
        '''
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"File {json_path} not found")
            return None
        except PermissionError:
            print(f"Permission error: {json_path}")
            return None
        except json.decoder.JSONDecodeError:
            print(f"JSON decode error: {json_path}")
            return None
        except Exception as e:
            print(f"Error loading json file: {e}")
            return None
        
    def _load_txt(self, txt_path: str):
        '''
        load txt file
        :param txt_path: txt file path
        :return: txt data
        '''
        try:
            with open(txt_path, 'r') as f:
                data = f.readlines()
            return data
        except FileNotFoundError:
            print(f"File {txt_path} not found")
            return None
        except PermissionError:
            print(f"Permission error: {txt_path}")
            return None
        except Exception as e:
            print(f"Error loading txt file: {e}")
            return None

    def _load_yaml(self, yaml_path: str) -> dict:
        '''
        load yaml file
        :param yaml_path: yaml file path
        :return: yaml data
        '''
        with open(yaml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data
    
    def generate_pkl(self, save_root: str=None, is_train: bool=None, ):
        '''
        generate pkl file for MOT20 dataset
        :param save_root: save path
        :param is_train: True for training set, False for testing set
        :return: None
        '''

        # check is train or test
        if is_train is None:
            types = ['train', 'test', 'val'] if self.is_val else ['train', 'test']
        elif is_train:
            types = ['train', 'val'] if self.is_val else ['train']
        else:
            types = ['test']
            
        # handle info
        for t in types:
            seq_names = self.info[t]
            
            seq_names.sort()
     
            for seq_name in tqdm(seq_names):
                self.handle_seq(seq_name, t)

            self.save_pkl(save_root)

    def handle_seq(self, seq_name: str, types: str='train'):
        '''
        handle one sequence
        :param seq_name: sequence name
        :param type: 'train', 'test'
        :return: None
        '''
        imgs = os.listdir(os.path.join(self.MOT20_root, types, seq_name, 'img1'))
        imgs.sort()

        if types=='train' or types=='val':
            lables_txt = os.path.join(self.MOT20_root, types, seq_name, f"gt/gt.txt")
            lables_dict = self._load_txt(lables_txt)
            # lables_dict = self._load_json(lables_json)
            lables_dict = self.prase_annotations(lables_dict)
            # print(lables_dict.keys())
        else:
            lables_dict = {}

        # calib_yaml_paths = os.listdir(self.info[types]['calibration'])
        # calib_dict = {}
        # for calib_yaml_path in calib_yaml_paths:
        #     if calib_yaml_path.endswith('.yaml'):
        #         calib_yaml = self._load_yaml(os.path.join(self.info[types]['calibration'], calib_yaml_path))
        #         calib_type = calib_yaml_path.split('.')[0]
        #         calib_dict[calib_type] = calib_yaml


            
        # pointclouds_lower_path = os.listdir(os.path.join(self.info[types]['pointclouds'], 'lower_velodyne', seq_name))
        # pointclouds_lower_path.sort()
        # pointclouds_upper_path = os.listdir(os.path.join(self.info[types]['pointclouds'], 'upper_velodyne', seq_name))
        # pointclouds_lower_path.sort()

        timestamp_img_dict = self._prase_timestamp(os.path.join(self.info[types]['timestamps'], seq_name, 'frames_img.json'))
        timestamp_pc_dict = self._prase_timestamp(os.path.join(self.info[types]['timestamps'], seq_name, 'frames_pc.json'))

        

        for img in imgs:
            frame_dict = self.handle_singe_frame(img, seq_name, types, calib_dict, lables_dict, pointclouds_lower_path, pointclouds_upper_path, timestamp_img_dict, timestamp_pc_dict)

            if types=='train':
                self.train_pkl.append(frame_dict)
            elif types=='val':
                self.val_pkl.append(frame_dict)
            elif types=='test':
                self.test_pkl.append(frame_dict)
            else:
                raise ValueError('type should be "train", "test"')
        

    def handle_singe_frame(
            self, img: str, 
            seq_name: str, 
            types: str,  
            calib_dict: dict, 
            lables_dict: dict, 
            pointclouds_lower_path: list, 
            pointclouds_upper_path: list, 
            timestamp_img_dict: dict, 
            timestamp_pc_dict: dict
            ):
        ''' 
        handle one frame
        :param img: image name
        :param seq_name: sequence name
        :param types: 'train', 'test'
        :param calib_dict: calibration dict
        :param lables_dict: labels dict
        :param pointclouds_lower_path: lower pointcloud path
        :param pointclouds_upper_path: upper pointcloud path
        :param timestamp_img_dict: images timestamp dict
        :param timestamp_pc_dict: pointclouds timestamp dict
        :return: None
        '''
        sensor_name = ['image_stitched']
        frame_id = int(img.split('.')[0])
        pointcloud_lower_path = os.path.join(self.info[types]['pointclouds'], 'lower_velodyne', seq_name, pointclouds_lower_path[frame_id])
        pointcloud_upper_path = os.path.join(self.info[types]['pointclouds'], 'upper_velodyne', seq_name, pointclouds_upper_path[frame_id])
        cams = {}
        for sensor in sensor_name:
            img_path = os.path.join(self.info[types]['images'], sensor, seq_name, img)
            relative_path = os.path.relpath(img_path, self.data_root)
            final_path = os.path.join("./data", relative_path)

            cams[sensor] = {
                        'data_path':final_path, 
                        'type': sensor, 
                        'sample_data_token': seq_name + '_' + f'{sensor}_' + str(f"{frame_id:06}"), 
                        'timestamp': int(timestamp_img_dict[frame_id]*1_000_000), 
            }  

        if types=='train':
            bboxes = lables_dict[img]['bbox']
            label_id = lables_dict[img]['label_id']

            gt_names = [anno.split(":")[0] for anno in label_id]
            instance_ids =  [int(anno.split(":")[1])+1 for anno in label_id]

        
            instance_ids = np.array(instance_ids)
            bboxes = self.ltwh2cxywh(bboxes)
            ego_bboxes = np.array(bboxes)


        else:
            ego_bboxes = []
            gt_names = []
            instance_ids = np.array([])

        infos_frame = {
            'cams': cams,
            'sweeps': [],
            
            'gt_boxes': np.array(ego_bboxes),
            'gt_names': np.array(gt_names),
            'gt_velocity': [],
            'instance_inds': instance_ids,

            'timestamp': int(timestamp_pc_dict[frame_id]*1_000_000),
            'token': seq_name + '_' + str(f"{frame_id:06}"),

        }
        return infos_frame
        

    def _prase_timestamp(self, timestamp_path: str) -> dict:
        '''
        parse timestamp dict
        :param timestamp_path: timestamp path
        :return: timestamp_us
        '''
        
        timestamp_dict = self._load_json(timestamp_path)

        timestamp_info = {}
        for info in timestamp_dict['data']:
            #  pointclouds is empty, get timestamp from cameras
            if not info['pointclouds'] == []:
                data = info['pointclouds']
            else:
                data = info['cameras']
            
            url = data[0]['url']
            frame_id = int(os.path.basename(url).split('.')[0])
            timestamp_info[frame_id] = data[0]['timestamp']
        
        return timestamp_info

            

    @staticmethod
    def _save_pkl(data, filename, save_root):
        if len(data) > 0:
            info = {
                'infos': data,
                'metadata': dict(version='JRDB_v{self.__version__}')
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

        self._save_pkl(self.train_pkl, f'JRDB_infos_train_v{self.__version__}.pkl', save_root)
        self._save_pkl(self.val_pkl, f'JRDB_infos_val_v{self.__version__}.pkl', save_root)
        self._save_pkl(self.test_pkl, f'JRDB_infos_test_v{self.__version__}.pkl', save_root)


    def prase_annotations(self, lables_dict: dict) -> dict:
        '''
        prase annotations
        :param lables_dict: labels dict
        :return: prased labels dict
        '''
        annos = defaultdict(list)

        occlusion = 0.4
        global_max_id = 0
        for info in lables_dict:
            line = info.strip().split(',')
            if float(line[8]) < occlusion:
                continue
            frame_id = int(line[0])
            label_id = int(line[1])+self.global_id
            bbox = [float(line[2]), float(line[3]), float(line[4]), float(line[5])] # left, top, width, height
            instance = {
                "bbox": bbox,
                "label_id": label_id,
            }
            annos[frame_id].append(instance)

            global_max_id = max(global_max_id, label_id)

        self.global_id = global_max_id

        new = {}
        for frame_id, instances in annos.items():
            new[frame_id] = {
                "bbox": [instance['bbox'] for instance in instances],
                "label_id": [instance['label_id'] for instance in instances],
            }
            
        return new



    @staticmethod
    def rle_to_bbox(rle):
        """
        Convert RLE encoded mask to bounding box.
        
        Args:
        rle (dict): RLE encoding containing 'counts' and 'size' keys.
        height (int): Height of the mask.
        width (int): Width of the mask.
        
        Returns:
        list: Bounding box [x_min, y_min, x_max, y_max].
        """
        # Decode the RLE
        mask_array = mask.decode(rle)
        
        # Find the non-zero coordinates
        non_zero_coords = mask_array.nonzero()
        y_min, x_min = non_zero_coords[0].min(), non_zero_coords[1].min()
        y_max, x_max = non_zero_coords[0].max(), non_zero_coords[1].max()
        
        # Return the bounding box
        return [x_min, y_min, x_max, y_max]
    
    def ltwh2cxywh(self, ltwh):
        """
        Convert bounding box from left, top, width, height to center x, center y, width, height.
        
        Args:
        ltwh (list): Bounding box [left, top, width, height].

        Returns:
        list: Bounding box [center_x, center_y, width, height].
        """
        ltwh = np.array(ltwh)
            
        center_x = ltwh[:,0] + ltwh[:,2] / 2
        center_y = ltwh[:,1] + ltwh[:,3] / 2
        
        bbox_ = np.stack([center_x, center_y, ltwh[:,2], ltwh[:,3]], axis=1)
        return bbox_

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = f'{root}/data'
    save_root = f'{root}/data/MOT20_2d_stitched_anno_pkls'
    mot20 = MOT20Converter2D(data_root, save_root, is_val=False)

    mot20.generate_pkl()
