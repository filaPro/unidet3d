# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from .indoor_eval import indoor_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import get_box_type
from .show_results import show_result_v2
from pathlib import Path

@METRICS.register_module()
class IndoorMetric_(BaseMetric):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 datasets,
                 datasets_classes,
                 vis_dir: str = None,
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(IndoorMetric_, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.datasets = datasets
        self.datasets_classes = datasets_classes
        self.vis_dir = vis_dir

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            pred_3d['dataset'] = self.get_dataset(data_sample['lidar_path'])
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        ann_infos = [[] for _ in self.datasets]
        pred_results = [[] for _ in self.datasets]

        for eval_ann, sinlge_pred_results in results:
            idx = self.datasets.index(sinlge_pred_results['dataset'])
            ann_infos[idx].append(eval_ann)
            pred_results[idx].append(sinlge_pred_results)
            if self.vis_dir is not None:
                self.vis_results(eval_ann, sinlge_pred_results)

        # some checkpoints may not record the key "box_type_3d"
        box_type_3d, box_mode_3d = get_box_type(
            self.dataset_meta.get('box_type_3d', 'depth'))

        ret_dict = {}
        for i in range(len(self.datasets)):
            ret_dict[self.datasets[i]] = indoor_eval(
                                                ann_infos[i],
                                                pred_results[i],
                                                self.iou_thr,
                                                self.datasets_classes[i],
                                                logger=logger,
                                                box_mode_3d=box_mode_3d)

        return ret_dict

    def get_dataset(self, lidar_path):
        for dataset in self.datasets:
            if dataset in lidar_path.split('/'):
                return dataset

    def vis_results(self, eval_ann, sinlge_pred_results):
        pts = sinlge_pred_results['points'].numpy()
        pts[:, 3:] *= 127.5
        pts[:, 3:] += 127.5
        show_result_v2(pts, eval_ann['gt_bboxes_3d'].corners, 
                    eval_ann['gt_labels_3d'],
                    sinlge_pred_results['bboxes_3d'].corners, 
                    sinlge_pred_results['labels_3d'],
                    Path(self.vis_dir) / sinlge_pred_results['dataset'],
                    eval_ann['lidar_idx'])