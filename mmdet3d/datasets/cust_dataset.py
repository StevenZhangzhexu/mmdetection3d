import copy
import mmcv
import numpy as np
import os
import tempfile
import torch

from os import path as osp

# from ..core import show_result
# from mmcv.utils import print_log

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset

@DATASETS.register_module()
class CustomDataset(Det3DDataset):
    r"""Custom Dataset.

    Dataset class for datasets comprising of pointclouds and annotations.
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        data_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    METAINFO = { 'classes':( 
                'Bollard',
                # 'Building',
                #'Bus Stop',
                'ControlBox',
                #'Ground',
                'LampPost',
                # 'Pole',
                # 'Railing',
                # 'Road',
                # 'Shrub',
                'Sign',
                'BusStop',
                'TrafficLight'
                # ,'dontcare'
                )}

    def __init__(self,
                 data_root,
                 ann_file,
                #  pts_prefix='velodyne',
                 data_prefix='velodyne',
                 pipeline=None,
                #  classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pcd_limit_range= [30264,29862, 1.7, 34040, 41990, 25.7], #[-20, -20, 0, 20, 20, 20],
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            # classes=classes,
            data_prefix=data_prefix,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        # self.pts_prefix = pts_prefix


    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)

        if ann_info is None:
            # empty instance
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # filter the gt classes not used in training
        ann_info = self._remove_dontcare(ann_info)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info

    

    # def evaluate(self,
    #              results,
    #              metric=None,
    #              logger=None,
    #              pklfile_prefix=None,
    #              submission_prefix=None,
    #              show=False,
    #              out_dir=None):
    #     """Evaluation in KITTI protocol.

    #     Args:
    #         results (list[dict]): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated.
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #         pklfile_prefix (str | None): The prefix of pkl files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #         submission_prefix (str | None): The prefix of submission datas.
    #             If not specified, the submission data will not be generated.
    #         show (bool): Whether to visualize.
    #             Default: False.
    #         out_dir (str): Path to save the visualization results.
    #             Default: None.

    #     Returns:
    #         dict[str, float]: Results of each evaluation metric.
    #     """
    #     return dict()

    # def show(self, results, out_dir):
    #     """Results visualization.

    #     Args:
    #         results (list[dict]): List of bounding boxes results.
    #         out_dir (str): Output directory of visualization result.
    #     """
    #     assert out_dir is not None, 'Expect out_dir, got none.'
    #     for i, result in enumerate(results):
    #         example = self.prepare_test_data(i)
    #         data_info = self.data_infos[i]
    #         pts_path = data_info['point_cloud']['velodyne_path']
    #         file_name = osp.split(pts_path)[-1].split('.')[0]
    #         # for now we convert points into depth mode
    #         points = example['points'][0]._data.numpy()
    #         points = points[..., [1, 0, 2]]
    #         points[..., 0] *= -1
    #         gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor
    #         gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
    #                                       Box3DMode.DEPTH)
    #         gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2
    #         pred_bboxes = result['boxes_3d'].tensor.numpy()
    #         pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
    #                                         Box3DMode.DEPTH)
    #         pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
    #         show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name)