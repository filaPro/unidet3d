import os
from concurrent import futures as futures
from os import path as osp
import mmengine
import numpy as np
from typing import List, Optional


class ARKitScenesOfflineData:
    """ARKitScenesOfflineData
    Generate arkitscenes infos (offline benchmark) for indoor_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str): Stplit type 'train' or 'val'.
    """
    def __init__(self, root_path: str, split: str):
        self.split = split
        raw_path = os.path.join(root_path, '3dod')
        self.data_path = os.path.join(root_path, 'offline_prepared_data')
        assert split in ['train', 'val']
        class_names = [
            'cabinet', 'refrigerator', 'shelf', 'stove', 'bed',
            'sink', 'washer', 'toilet', 'bathtub', 'oven',
            'dishwasher', 'fireplace', 'stool', 'chair', 'table',
            'tv_monitor', 'sofa'
        ]
        self.name2class = {
            name: i
            for i, name in enumerate(class_names)
        }
        all_id_list = set(
            map(lambda x: x.split('_')[0],
            os.listdir(self.data_path)))
        split_dir = 'Training' if split == 'train' else 'Validation'
        split_id_list = set(os.listdir(osp.join(raw_path, split_dir)))
        self.sample_id_list = all_id_list & split_id_list
        print(f'{split}, raw ids: {len(split_id_list)}, '
              f'processed ids: {len(self.sample_id_list)}')

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.sample_id_list)
    
    def get_infos(self,
                  num_workers: int = 4,
                  has_label: bool = True,
                  sample_id_list: Optional[List[str]] = None) -> dict:
        """Get data infos.
        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[str], optional): Index list of the sample.
                Default: None.

        Returns:
            dict: Information of the raw data.
        """
        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}', end='\r')
            info = {
                'lidar_points': {
                    'num_pts_feats': 6,
                    'lidar_path': f'{sample_idx}_point.bin'
                }
            }
            boxes = np.load(
                osp.join(self.data_path, f'{sample_idx}_bbox.npy'))
            labels = np.load(
                osp.join(self.data_path, f'{sample_idx}_label.npy'))
            instances = []
            for box, label in zip(boxes, labels):
                # follow heading angle of DepthInstance3DBoxes
                box[-1] = -box[-1]
                instances.append({
                    'bbox_3d': box.tolist(),
                    'bbox_label_3d': self.name2class[label]
                })
            info['instances'] = instances
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, list(sample_id_list))
        
        infos = {
            'metainfo': {
                'categories': self.name2class,
                'dataset': 'arkitscenes_offline',
                'info_version': '1.0'
            },
            'data_list': list(infos)
        }
        return infos


# do not want to add create_annotations.py to projects
if __name__ == '__main__':
    root_path = '/opt/project/data/arkitscenes'
    out_path = '/opt/project/work_dirs/tmp'
    infos_train = ARKitScenesOfflineData(
        root_path=root_path, split='train').get_infos()
    train_filename = osp.join(out_path, 'arkitscenes_offline_infos_train.pkl')
    mmengine.dump(infos_train, train_filename, 'pkl')
    infos_val = ARKitScenesOfflineData(
        root_path=root_path, split='val').get_infos()
    val_filename = osp.join(out_path, 'arkitscenes_offline_infos_val.pkl')
    mmengine.dump(infos_val, val_filename, 'pkl')
