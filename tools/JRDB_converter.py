import os
import sys
import json
import yaml
import pickle as pkl
import time
import pyquaternion
import numpy as np

# file names
FILE = ['calibration','images', 'labels', 'pointclouds', 'timestamps']


from tqdm import tqdm

class JRDBConverter(object):
    '''
    Convert JRDB dataset to Nuscenes format pkl file.
    '''
    def __init__(self, data_root: str, save_root: str):
        self.data_root = data_root
        self.JRDB_root = os.path.join(data_root, 'JRDB')
        self.save_root = save_root
        # current time in microseconds
        self.cur_time = int(time.time() * 1_000_000)
        self.global_instance_id = 0

        self.test_pkl=[]
        self.train_pkl=[]
     
        self.info = {}
        for folder in ['train', 'test']:
            JRDB_folder = os.listdir(os.path.join(self.JRDB_root, folder))
            JRDB_folder = [sub_folder for sub_folder in JRDB_folder if (not sub_folder.endswith('.zip'))]  # remove zip files
            JRDB_folder = [sub_folder for sub_folder in JRDB_folder if any(file_name in sub_folder for file_name in FILE)]  # save only folders with required files
            file_path = [{seq_name.split("_")[1] : os.path.join(self.JRDB_root, folder, seq_name, os.listdir(os.path.join(self.JRDB_root, folder, seq_name))[0])} for seq_name in JRDB_folder]
            file_path = {k:v for d in file_path for k,v in d.items()}
            info = {folder: file_path}
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
        generate pkl file for JRDB dataset
        :param save_root: save path
        :param is_train: True for training set, False for testing set
        :return: None
        '''

        # check is train or test
        if is_train is None:
            types = ['train', 'test']
        elif is_train==True:
            types = ['train']
        else:
            types = ['test']
            
        # handle info
        for t in types:
            seq_names = os.listdir(os.path.join(self.info[t]['images'], os.listdir(self.info[t]['images'])[0]))
            seq_names.sort()
            for seq_name in tqdm(seq_names):
                self.handle_seq(seq_name, t)

            self.save_pkl(save_root)


    def handle_frame(self, frame_path: str, frame_id: int, seq_id: int, seqinfo: dict, gt_dict: dict) -> dict:
        
        # handle frame
        interval = int(1/float(seqinfo['frameRate'])*1000*1000)  # microseconds
        
        timestamp_us = self.cur_time + seq_id * 600_000_000 + frame_id*interval
        

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

    def handle_seq(self, seq_name: str, types: str='train'):
        '''
        handle one sequence
        :param seq_name: sequence name
        :param type: 'train', 'test'
        :return: None
        '''
        imgs = os.listdir(os.path.join(self.info[types]['images'], 'image_0' , seq_name))
        imgs.sort()

        if types=='train':
            lables_json = os.path.join(self.info[types]['labels'], 'labels_3d', f"{seq_name}.json")
            lables_dict = self._load_json(lables_json)
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
        sensor_name = ['image_0', 'image_2', 'image_4', 'image_6', 'image_8']
        frame_id = int(img.split('.')[0])
        pointcloud_lower_path = os.path.join(self.info[types]['pointclouds'], 'lower_velodyne', seq_name, pointclouds_lower_path[frame_id])
        pointcloud_upper_path = os.path.join(self.info[types]['pointclouds'], 'upper_velodyne', seq_name, pointclouds_upper_path[frame_id])
        cams = {}
        for sensor in sensor_name:
            img_path = os.path.join(self.info[types]['images'], sensor, seq_name, img)
            relative_path = os.path.relpath(img_path, self.data_root)
            final_path = os.path.join("./data", relative_path)

            # maybe not right
            cam_calib = calib_dict['lidars'][f'{sensor.replace("image", "sensor")}']
            cam2ego = np.array(cam_calib['cam2ego'])
            sensor2ego_translation = list(cam2ego[:3, 3])

            # 3*3 rataion matrix to quaternion
            sensor2ego_rotation = cam2ego[:3, :3]

            upper2cam = np.array(cam_calib['upper2cam'])
            # invert upper2cam
            cam2upper = np.linalg.inv(upper2cam)
            sensor2lidar_translation = list(cam2upper[:3, 3])
            sensor2lidar_rotation = cam2upper[:3, :3]

            distortion = np.array(cam_calib['D'])
            distorted_img_K = np.array(cam_calib['distorted_img_K'])
            undistorted_img_K = np.array(cam_calib['undistorted_img_K'])
            cams[sensor] = {
                        'data_path':final_path, 
                        'type': sensor, 
                        'sample_data_token': seq_name + '_' + f'{sensor}' + str(f"{frame_id:06}"), 
                        'sensor2ego_translation': sensor2ego_translation, 
                        'sensor2ego_rotation': sensor2ego_rotation, 
                        'ego2global_translation': [0, 0, 0], 
                        'ego2global_rotation' : [1, 0, 0, 0], 
                        'timestamp': int(timestamp_img_dict[frame_id]*1_000_000), 
                        'sensor2lidar_rotation': sensor2lidar_rotation, 
                        'sensor2lidar_translation': sensor2lidar_translation, 
                        'cam_intrinsic': undistorted_img_K, 
                        'distortion_coefficient': distortion,
                        'distorted_img_K': distorted_img_K,
                        'undistorted_img_K': undistorted_img_K,
            }  

        if types=='train':
            annos_list = lables_dict['labels'][f"{frame_id:06}.pcd"]
            gt_names = [anno['label_id'].split(":")[0] for anno in annos_list]
            instance_ids =  [anno['label_id'].split(":")[1] for anno in annos_list]

            bboxes = [anno['box'] for anno in annos_list]

            ego_bboxes = []
            for bbox in bboxes:
                ego_bboxes.append([bbox['cx'], bbox['cy'], bbox['cz'], bbox['l'],  bbox['w'],  bbox['h'],  bbox['rot_z']])

            instance_ids = np.array(instance_ids)


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
            'lidar_path': [pointcloud_lower_path, pointcloud_upper_path],

            'lidar2ego_translation': [0, 0, 0],   # the bbox  in ego coordinate system, so the translation is [0, 0, 0]
            'lidar2ego_rotation': [1, 0, 0, 0],   # don`t  need to set rotation, because the bbox  in ego coordinate system, so the rotation is [1, 0, 0, 0]
            'ego2global_translation': [0, 0, 0], 
            'ego2global_rotation': np.array([1, 0, 0, 0]), 

            'num_lidar_pts': np.full(instance_ids.shape, 200),  # not to filter the points, so set it to 200
            'valid_flag': np.full(instance_ids.shape, True),

            'lower2upper': calib_dict['lidars']['lidar']['lower2upper'],
            'upper2ego': calib_dict['lidars']['lidar']['upper2ego'],

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
                'metadata': dict(version='JRDB_v1.0')
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

        self._save_pkl(self.train_pkl, 'JRDB_infos_train.pkl', save_root)
        self._save_pkl(self.test_pkl, 'JRDB_infos_test.pkl', save_root)




if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = f'{root}/data'
    save_root = f'{root}/data/JRDB_anno_pkls'
    danctrack = JRDBConverter(data_root, save_root)

    danctrack.generate_pkl()
