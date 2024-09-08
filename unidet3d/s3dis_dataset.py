from mmdet3d.registry import DATASETS
from mmdet3d.datasets.s3dis_dataset import S3DISDataset
import os.path as osp
from mmengine.logging import print_log
import logging
import numpy as np

@DATASETS.register_module()
class S3DISSegDetDataset(S3DISDataset):
    """S3DISSegDetDataset dataset.

    Args:
        partition(float): Defaults to 1, the part of 
            the dataset that will be used.
    """
    def __init__(self,
                 partition: float = 1,
                 **kwargs) -> None:
        self.partition = partition
        super().__init__(**kwargs)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['super_pts_path'] = osp.join(
            self.data_prefix.get('sp_pts_mask', ''), info['super_pts_path'])

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
