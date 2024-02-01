from typing import Sequence

from mmengine.dataset import DefaultSampler
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet3d.registry import HOOKS
from mmdet3d.structures import Det3DDataSample
from ..evaluation.metrics import V2V4RealMetric
from .visualizer import V2V4RealVisualizer


@HOOKS.register_module()
class V2V4RealVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize testing process
    prediction results.

    In the testing phase:
    1. If ``show`` is True, it means that only the prediction results are visualized without storing data


    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        vis_task (str): Visualization task. Defaults to 'lidar_det'.
        wait_time (float): The interval of show (s). Defaults to -1, which means wait until the window is closed
        test_out_dir (str, optional): directory where painted images will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the corresponding backend. Defaults to None.
    """

    def __init__(
            self,
            draw: bool = False,
            show: bool = False,
            vis_task: str = "lidar_det",
            wait_time: float = 0.0,
            score_thr: float = 0.2,
            # test_out_dir: Optional[str] = None,
            # backend_args: Optional[dict] = None
    ):
        super().__init__()
        self.wait_time = wait_time
        self.vis_task = vis_task
        self.score_thr = score_thr
        self.show = show
        self.draw = draw
        self.visualizer = V2V4RealVisualizer()
        self._test_index = 0
        assert vis_task == "lidar_det"

    # noinspection PyMethodOverriding
    def after_test_iter(
            self,
            runner: Runner,
            batch_idx: int,
            data_batch: dict,
            outputs: Sequence[Det3DDataSample],
    ) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples that contain annotations and predictions.
        """
        if not self.draw:
            return

        # noinspection PyTypeChecker
        data_sample: Det3DDataSample = None
        for i, data_sample in enumerate(outputs):
            data_input = dict()
            points = data_batch["inputs"]["points"][i].detach().cpu().numpy()
            data_input["points"] = points
            # if self._test_index % 400 == 1:
            # if self._test_index == 401:
            # if self._test_index == 1601:
            # print(self._test_index)
            self.visualizer.add_datasample(
                data_input,
                data_sample,
                draw_gt=True,
                draw_pred=True,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
            )
            # break
            self._test_index += 1
