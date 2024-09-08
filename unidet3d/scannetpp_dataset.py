from typing import Union
import numpy as np
from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes
import os.path as osp
from mmengine.logging import print_log
import logging
import numpy as np

@DATASETS.register_module()
class Scannetpp(Det3DDataset):
    """Scannetpp dataset.

    Args:
        data_prefix (dict): Prefix for data. Defaults to
            dict(pts='points', pts_instance_mask='instance_mask',
                     pts_semantic_mask='semantic_mask').
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth'.
    """
    METAINFO = {
        'classes':
        ('wall',  'ceiling',  'floor',  'table',  'door',  'ceiling lamp',  'cabinet',  'blinds',  'curtain',  'chair',  'storage cabinet',  'office chair',  'bookshelf',  'whiteboard',  'window',  'box',  'window frame',  'monitor',  'shelf',  'doorframe',  'pipe',  'heater',  'kitchen cabinet',  'sofa',  'windowsill',  'bed',  'shower wall',  'trash can',  'book',  'plant',  'blanket',  'tv',  'computer tower',  'kitchen counter',  'refrigerator',  'jacket',  'electrical duct',  'sink',  'bag',  'picture',  'pillow',  'towel',  'suitcase',  'backpack',  'crate',  'keyboard',  'rack',  'toilet',  'paper',  'printer',  'poster',  'painting',  'microwave',  'board',  'shoes',  'socket',  'bottle',  'bucket',  'cushion',  'basket',  'shoe rack',  'telephone',  'file folder',  'cloth',  'blind rail',  'laptop',  'plant pot',  'exhaust fan',  'cup',  'coat hanger',  'light switch',  'speaker',  'table lamp',  'air vent',  'clothes hanger',  'kettle',  'smoke detector',  'container',  'power strip',  'slippers',  'paper bag',  'mouse',  'cutting board',  'toilet paper',  'paper towel',  'pot',  'clock',  'pan',  'tap',  'jar',  'soap dispenser',  'binder',  'bowl',  'tissue box',  'whiteboard eraser',  'toilet brush',  'spray bottle',  'headphones',  'stapler',  'marker')
    }
    
    def __init__(self,
                 data_prefix=dict(
                     pts='points',
                     pts_instance_mask='instance_mask',
                     pts_semantic_mask='semantic_mask'),
                 box_type_3d='Depth',
                 **kwargs):
        super().__init__(
            data_prefix=data_prefix, box_type_3d=box_type_3d, **kwargs)

    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)
        
        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            origin=(0.5, 0.5, 0.5), box_dim=6,
            with_yaw=False).convert_to(self.box_mode_3d)

        return ann_info


@DATASETS.register_module()
class Scannetpp_(Scannetpp):
    """Scannetpp dataset with partition.

    Args:
        partition(float): Defaults to 1, the part of 
            the dataset that will be used.
    """
    METAINFO = {
        'classes':
        ('table', 'door', 'ceiling lamp', 'cabinet', 'blinds', 'curtain', 
         'chair', 'storage cabinet', 'office chair', 'bookshelf', 'whiteboard', 
         'window', 'box', 'monitor', 'shelf', 'heater', 'kitchen cabinet', 
         'sofa', 'bed', 'trash can', 'book', 'plant', 'blanket', 'tv', 
         'computer tower', 'refrigerator', 'jacket', 'sink', 'bag', 'picture', 
         'pillow', 'towel', 'suitcase', 'backpack', 'crate', 'keyboard', 'rack', 
         'toilet', 'printer', 'poster', 'painting', 'microwave', 'shoes', 
         'socket', 'bottle', 'bucket', 'cushion', 'basket', 'shoe rack', 
         'telephone', 'file folder', 'laptop', 'plant pot', 'exhaust fan', 
         'cup', 'coat hanger', 'light switch', 'speaker', 'table lamp', 
         'kettle', 'smoke detector', 'container', 'power strip', 'slippers', 
         'paper bag', 'mouse', 'cutting board', 'toilet paper', 'paper towel', 
         'pot', 'clock', 'pan', 'tap', 'jar', 'soap dispenser', 'binder', 'bowl', 
         'tissue box', 'whiteboard eraser', 'toilet brush', 'spray bottle', 
         'headphones', 'stapler', 'marker'), 
        'valid_class_ids': (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 
                            21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 34, 35, 37, 
                            38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51,
                            52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 65, 66, 67, 
                            68, 69, 70, 71, 72, 75, 76, 77, 78, 79, 80, 81, 82, 
                            83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 
                            96, 97, 98, 99)
    }

    def __init__(self,
                 partition: float = 1,
                 **kwargs) -> None:
        self.partition = partition
        super().__init__(**kwargs)

    def parse_ann_info(self, info: dict) -> Union[dict, None]:
        """Process the `instances` in data info to `ann_info`.

        In `Custom3DDataset`, we simply concatenate all the field
        in `instances` to `np.ndarray`, you can do the specific
        process in subclass. You have to convert `gt_bboxes_3d`
        to different coordinates according to the task.

        Args:
            info (dict): Info dict.

        Returns:
            dict or None: Processed `ann_info`.
        """
        ids = {c: i for i, c in enumerate(self.metainfo['valid_class_ids'])}
        instances = []
        for instance in info['instances']:
            if instance['bbox_label_3d'] in ids:
                instance['bbox_label_3d'] = ids[instance['bbox_label_3d']]
                instances.append(instance)
        info['instances'] = instances
        return super().parse_ann_info(info)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['super_pts_path'] = osp.join(
            self.data_prefix.get('sp_pts_mask', ''), 
            info['lidar_points']['lidar_path']) #info['super_pts_path']

        info = super().parse_data_info(info)

        return info

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.

        if not self.test_mode:
            if self.serialize_data:
                dataset_len = len(self.data_address)
            else:
                dataset_len = len(self.data_list)
            idx = np.random.randint(0, dataset_len)
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """

        if self.serialize_data:
            dataset_len = len(self.data_address)
        else:
            dataset_len = len(self.data_list)
        if not self.test_mode:
            return int(self.partition * dataset_len)
        else:
            return dataset_len
