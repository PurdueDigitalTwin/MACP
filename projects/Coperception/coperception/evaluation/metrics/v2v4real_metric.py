from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS
from shapely.geometry import Polygon

from mmdet3d.structures import LiDARInstance3DBoxes
from ..utils import calculate_tp_fp, eval_final_results


@METRICS.register_module()
class V2V4RealMetric(BaseMetric):
    RangeMapping: dict[str, tuple] = {
        "overall": (-float("inf"), float("inf")),
        "short": (0, 30),
        "middle": (30, 50),
        "long": (50, 100),
    }
    """dict[str, tuple]: The mapping from the range name to the range."""
    default_prefix: str
    """str: The default prefix of the metric."""
    threshold: float
    """float: The threshold of the metric."""
    data_infos: list[dict[str, Any]]
    """list[dict[str, Any]]: The dataset information dictionary."""
    ann_file: str
    """str: The annotation file path."""
    classes: list[str]
    """list[str]: The class names."""
    metrics: list[str]
    """list[str]: The metrics to be evaluated."""
    subsampler_indices: Optional[Sequence[int]]
    """Optional[Sequence[int]]: The indices of the subsampler."""

    def __init__(
        self,
        ann_file: str,
        score_threshold: float = 0.2,
        nms_iou_threshold: float = 0.15,
        metrics: Union[str, List[str]] = ["overall", "short", "middle", "long"],
        pcd_limit_range: List[float] = [-100.0, -40.0, -5.0, 100.0, 40.0, 3.0],
        collect_device: str = "cpu",
        subsampler_indices: List[int] = None,
    ) -> None:
        assert (
            isinstance(score_threshold, float) and 0.0 <= score_threshold <= 1.0
        ), "score_threshold must be in [0, 1]"
        assert (
            isinstance(nms_iou_threshold, float) and 0.0 <= nms_iou_threshold <= 1.0
        ), "nms_iou_threshold must be in [0, 1]"

        self.default_prefix = ""
        super().__init__(collect_device)
        self.data_infos = None
        self.ann_file = ann_file
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.pcd_limit_range = pcd_limit_range
        self.classes = ["car"]
        self.subsampler_indices = subsampler_indices

    @staticmethod
    def post_process(
        pred_box3d_tensor: torch.Tensor,
        pred_score: torch.Tensor,
        score_threshold: float,
        nms_iou_threshold: float,
        pcd_limit_range: List[float],
    ):
        # filter by score
        valid_mask = pred_score > score_threshold
        pred_box3d_tensor = pred_box3d_tensor[valid_mask]
        pred_score = pred_score[valid_mask]

        # remove large boxes
        keep_index_1 = V2V4RealMetric.remove_large_bbx(pred_box3d_tensor)
        # remove abnormal z boxes
        keep_index_2 = V2V4RealMetric.remove_abnormal_z_bbx(pred_box3d_tensor)

        keep_index = torch.logical_and(keep_index_1, keep_index_2)
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        pred_score = pred_score[keep_index]

        # nms
        keep_index = V2V4RealMetric.nms_rotated(
            bbx_3d=pred_box3d_tensor, scores=pred_score, threshold=nms_iou_threshold
        )
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        pred_score = pred_score[keep_index]

        # filter out boxes outside the range
        mask = V2V4RealMetric.get_mask_for_boxes_within_range(
            pred_box3d_tensor, pcd_limit_range
        )
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        pred_score = pred_score[mask]

        assert pred_box3d_tensor.shape[0] == pred_score.shape[0]
        return pred_box3d_tensor, pred_score

    @staticmethod
    def get_mask_for_boxes_within_range(box3d_tensor: torch.Tensor, pcd_limit_range):
        """
        Args:
            box3d_tensor: torch.Tensor, shape (N, 8, 3)
        Returns:
            mask: torch.Tensor, shape (N, )
                The mask for bounding boxes: True if the box is within the range,
                False otherwise.
        """
        assert box3d_tensor.shape[1:] == (8, 3)
        boundary_lower_range = torch.tensor(
            pcd_limit_range[0:2], device=box3d_tensor.device
        )
        boundary_upper_range = torch.tensor(
            pcd_limit_range[3:5], device=box3d_tensor.device
        )
        mask = torch.all(
            torch.all(box3d_tensor[:, :, 0:2] >= boundary_lower_range, dim=-1)
            & torch.all(box3d_tensor[:, :, 0:2] <= boundary_upper_range, dim=-1),
            dim=-1,
        )
        return mask

    def format_result(self, result: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        sample_idx = result["sample_idx"]
        pred_box3d_tensor: torch.Tensor = result["pred_instances_3d"][
            "bboxes_3d"
        ].corners  # shape [N, 8, 3]
        pred_score = result["pred_instances_3d"]["scores_3d"]  # shape [N]
        # post process
        pred_box3d_tensor, pred_score = self.post_process(
            pred_box3d_tensor,
            pred_score,
            self.score_threshold,
            self.nms_iou_threshold,
            self.pcd_limit_range,
        )

        gt_info = self.data_infos[
            self.subsampler_indices.index(sample_idx)
            if self.subsampler_indices is not None
            else sample_idx
        ]
        assert sample_idx == int(gt_info["sample_idx"])
        gt_box3d_tensor = torch.tensor(
            [instance["bbox_3d"] for instance in gt_info["instances"]]
        )
        gt_box3d_tensor = LiDARInstance3DBoxes(
            gt_box3d_tensor, origin=(0.5, 0.5, 0.5)
        ).corners
        # filter the gt bboxes to make sure all the gt bboxes are within the range
        gt_mask = self.get_mask_for_boxes_within_range(
            gt_box3d_tensor, self.pcd_limit_range
        )
        gt_box3d_tensor = gt_box3d_tensor[gt_mask, :, :]

        return pred_box3d_tensor, pred_score, gt_box3d_tensor

    @staticmethod
    def _create_counter_dict():
        return {
            0.5: {"tp": [], "fp": [], "gt": 0},
            0.7: {"tp": [], "fp": [], "gt": 0},
        }

    def compute_metrics(self, results: List[Dict[str, torch.Tensor]]) -> dict:
        """Compute the metrics from processed results.

        Args:
        results(List[dict]): The processed results of each batch.

        Returns:
        Dict[str, float]: The computed metrics.
            The keys are the names of the metrics, and the values are the
            corresponding results.
        """

        _: MMLogger = MMLogger.get_current_instance()
        pkl_infos = load(self.ann_file)
        self.data_infos = pkl_infos["data_list"]
        if self.subsampler_indices is not None:
            self.data_infos = [self.data_infos[i] for i in self.subsampler_indices]

        assert len(results) == len(self.data_infos)
        # Create the dictionary for evaluation
        metrics = {m: self._create_counter_dict() for m in self.metrics}

        for idx in range(len(results)):
            assert results[idx]["sample_idx"] == (
                self.subsampler_indices[idx]
                if self.subsampler_indices is not None
                else idx
            )
            pred_box_tensor, pred_score, gt_box_tensor = self.format_result(
                results[idx]
            )

            for rg, result_stat in metrics.items():
                calculate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    result_stat,
                    0.5,
                    left_range=self.RangeMapping[rg][0],
                    right_range=self.RangeMapping[rg][1],
                )
                calculate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    result_stat,
                    0.7,
                    left_range=self.RangeMapping[rg][0],
                    right_range=self.RangeMapping[rg][1],
                )

        metric_dict = {}
        for rg, result_stat in metrics.items():
            ap_dict = eval_final_results(result_stat, rg)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]
        return metric_dict

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch(dict): A batch of data from the dataloader.
            data_samples(Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample["pred_instances_3d"]
            pred_2d = data_sample["pred_instances"]
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to("cpu")
            result["pred_instances_3d"] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to("cpu")
            result["pred_instances"] = pred_2d
            sample_idx = data_sample["sample_idx"]
            result["sample_idx"] = sample_idx
            self.results.append(result)

    # ---------------------Utility functions--------------------- #
    @staticmethod
    def remove_large_bbx(bbx_3d):
        """Remove large bounding box.

        Args:
            bbx_3d: torch.Tensor
                The bounding boxes, shape(N, 8, 3).
        Returns:
            keep_index: torch.Tensor
                The keep index.
        """

        bbx_x_max = torch.max(bbx_3d[:, :, 0], dim=1)[0]
        bbx_x_min = torch.min(bbx_3d[:, :, 0], dim=1)[0]
        x_len = bbx_x_max - bbx_x_min

        bbx_y_max = torch.max(bbx_3d[:, :, 1], dim=1)[0]
        bbx_y_min = torch.min(bbx_3d[:, :, 1], dim=1)[0]
        y_len = bbx_y_max - bbx_y_min

        bbx_z_max = torch.max(bbx_3d[:, :, 2], dim=1)[0]
        bbx_z_min = torch.min(bbx_3d[:, :, 2], dim=1)[0]
        z_len = bbx_z_max - bbx_z_min

        keep_index = torch.logical_and(x_len <= 6, y_len <= 6)
        keep_index = torch.logical_and(keep_index, z_len)

        return keep_index

    @staticmethod
    def remove_abnormal_z_bbx(bbx_3d):
        """Remove abnormal bounding box that has abnormal z value.

        Args:
            bbx_3d: torch.Tensor
                The bounding boxes, shape(N, 8, 3).
        Returns:
            keep_index: torch.Tensor
                The keep index.
        """

        bbx_z_min = torch.min(bbx_3d[:, :, 2], dim=1)[0]
        bbx_z_max = torch.max(bbx_3d[:, :, 2], dim=1)[0]
        keep_index = torch.logical_and(bbx_z_min >= -3, bbx_z_max <= 1)
        return keep_index

    @staticmethod
    def nms_rotated(bbx_3d, scores, threshold):
        """Performs rotated non-maximum suppression, and returns indices of
        kept boxes.

        Args:
            bbx_3d: torch.Tensor
                The bounding boxes, shape(N, 8, 3).
            scores: torch.Tensor
                The predicted scores of bounding boxes, shape(N, ).
            threshold: float
                The threshold to perform nms.
        Returns:
            keep_index: np.ndarray
                The keep index.
        """
        if bbx_3d.shape[0] == 0:
            return np.array([], dtype=np.int32)
        bbx_3d = bbx_3d.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()

        polygons = V2V4RealMetric.bbx_3d_to_z0_2d(bbx_3d)

        # Get indices of boxes sorted by scores (highest first)
        idxs = scores.argsort()[::-1]

        keep = []  # boxes to keep
        while idxs.size > 0:
            # Pick top box and add its index to the list of kept boxes
            i = idxs[0]
            keep.append(i)
            # Compute IoU of the picked box with the rest
            iou = V2V4RealMetric.compute_iou(
                polygons[i], polygons[idxs[1:]]
            )  # idxs[0] is the index of the box with the highest score
            # Identify boxes with IoU over the threshold. This returns indices into
            # idxs[1:], so add 1 to get indices into idxs
            remove_idxs = np.where(iou > threshold)[0] + 1
            # Remove indices of the picked, and overlapped boxes.
            idxs = np.delete(idxs, remove_idxs)
            idxs = np.delete(idxs, 0)

        return np.array(keep, dtype=np.int32)

    @staticmethod
    def bbx_3d_to_z0_2d(bbx_3d):
        """Convert 3d bounding boxes to shapely.geometry.Polygon 2d bounding
        boxes.

        Args:
            bbx_3d: np.ndarray
                The 3d bounding boxes, shape(N, 8, 3)
                in the form of
                (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
                 0       1       2       3       4       5       6       7
        Returns:
            polygons: np.ndarray of shapely.geometry.Polygon
                The 2d bounding boxes, shape(N, )
        """
        z0_indices = [0, 3, 7, 4]
        polygons = [
            Polygon([(bbx_3d_i[j, 0], bbx_3d_i[j, 1]) for j in z0_indices])
            for bbx_3d_i in bbx_3d
        ]
        return np.array(polygons)

    @staticmethod
    def compute_iou(bbx_2d, bbx_2d_list):
        """Compute the iou between a box and a list of boxes efficiently.

        Args:
            bbx_2d: shapely.geometry.Polygon
            bbx_2d_list: list
        Returns:
            iou: np.ndarray
                The iou between the box and the list of boxes, shape(N, )
        """
        # Calculate intersection areas
        iou = [
            bbx_2d.intersection(bbx_2d_i).area / bbx_2d.union(bbx_2d_i).area
            for bbx_2d_i in bbx_2d_list
        ]
        return np.array(iou, dtype=np.float32)
