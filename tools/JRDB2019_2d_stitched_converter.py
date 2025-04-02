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



class JRDB2019Converter2D(object):
    '''
    Convert JRDB dataset to Nuscenes format pkl file.
    '''
    def __init__(self, data_root: str, save_root: str, is_val: bool=False): 
        self.data_root = data_root
        self.JRDB_root = os.path.join(data_root, 'JRDB2019')
        self.save_root = save_root
        self.is_val = is_val

        self.test_pkl=[]
        self.train_pkl=[]
        self.val_pkl=[]

        self.__version__ = '1.2'

     
        self.info = {}
        for i in ['train', 'test']:
            # folder = "jrdb19_train_norosbag/train_dataset_with_activity" if i=="train"  else "jrdb19_test_norosbag/test_dataset_without_labels"
            folder = "train_dataset_with_activity" if i=="train"  else "test_dataset_without_labels"
            JRDB_folder = os.listdir(os.path.join(self.JRDB_root, folder))
            JRDB_folder = [sub_folder for sub_folder in JRDB_folder if (not sub_folder.endswith('.zip'))]  # remove zip files
            JRDB_folder = [sub_folder for sub_folder in JRDB_folder if any(file_name in sub_folder for file_name in FILE)]  # save only folders with required files
            file_path = [{seq_name : os.path.join(self.JRDB_root, folder, seq_name)} for seq_name in JRDB_folder]
            file_path = {k:v for d in file_path for k,v in d.items()}
            info = {i: file_path}
            self.info.update(info)
            if self.is_val and i=='train':
                self.info['val'] = self.info['train']


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
        generate pkl file for JRDB2D dataset
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
            seq_names = os.listdir(os.path.join(self.info[t]['images'], "image_stitched"))
            
            if t == 'train':
                seq_names = [seq for seq in seq_names if seq not in validation]
            if t == 'val':
                seq_names = [seq for seq in seq_names if seq in validation]

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
        imgs = os.listdir(os.path.join(self.info[types]['images'], 'image_stitched' , seq_name))
        imgs.sort()

        if types=='train' or types=='val':
            lables_json = os.path.join(self.info[types]['labels'], 'labels_2d_stitched', f"{seq_name}.json")
            lables_dict = self._load_json(lables_json)
            lables_dict = self.prase_annotations(lables_dict)
            # print(lables_dict.keys())
        else:
            lables_dict = {}

        calib_yaml_paths = os.listdir(self.info[types]['calibration'])
        calib_dict = {}
        for calib_yaml_path in calib_yaml_paths:
            if calib_yaml_path.endswith('.yaml'):
                calib_yaml = self._load_yaml(os.path.join(self.info[types]['calibration'], calib_yaml_path))
                calib_type = calib_yaml_path.split('.')[0]
                calib_dict[calib_type] = calib_yaml
            
        pointclouds_lower_path = os.listdir(os.path.join(self.info[types]['pointclouds'], 'lower_velodyne', seq_name))
        pointclouds_lower_path.sort()
        pointclouds_upper_path = os.listdir(os.path.join(self.info[types]['pointclouds'], 'upper_velodyne', seq_name))
        pointclouds_lower_path.sort()

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
                print(f"Save {filename} to {save_root}")

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
        annos = defaultdict(dict)

        occlusion = ['Mostly_visible', 'Fully_visible']
        for frame, lable in lables_dict["labels"].items():
            bbox = [l['box'] for l in lable if l['attributes']['occlusion'] in occlusion]
            label_id = [l['label_id'] for l in lable if l['attributes']['occlusion'] in occlusion]
            info = {
                "bbox": bbox,
                "label_id": label_id,
            }
            annos[frame]=info

        
        return annos



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
    save_root = f'{root}/data/JRDB2019_2d_stitched_anno_pkls'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    danctrack = JRDB2019Converter2D(data_root, save_root, is_val=True)

    danctrack.generate_pkl()
