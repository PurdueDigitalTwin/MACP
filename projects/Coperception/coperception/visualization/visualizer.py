import math
import time
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mmdet.visualization import DetLocalVisualizer
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmengine.visualization.utils import (
    check_type,
    color_val_matplotlib,
    tensor2ndarray,
)
from torch import Tensor

from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import (
    BaseInstance3DBoxes,
    Box3DMode,
    Coord3DMode,
    DepthInstance3DBoxes,
    Det3DDataSample,
    LiDARInstance3DBoxes,
)
from mmdet3d.visualization import to_depth_mode
from .vis_utils import proj_lidar_bbox3d_to_img

try:
    import open3d as o3d
    from open3d import geometry
    from open3d.visualization import Visualizer
except ImportError:
    o3d = geometry = Visualizer = None

VIRIDIS = np.array(cm.get_cmap("plasma").colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


@VISUALIZERS.register_module()
class V2V4RealVisualizer(DetLocalVisualizer):
    def __init__(
        self, name: str = "visualizer", multi_imgs_col: int = 3, fig_show_cfg=None
    ):
        super().__init__(name)
        if fig_show_cfg is None:
            fig_show_cfg = dict(figsize=(18, 12))
        self.multi_imgs_col = multi_imgs_col
        self.fig_show_cfg.update(fig_show_cfg)

    @staticmethod
    def _initialize_o3d_vis(frame_cfg: dict) -> Visualizer:
        """Initialize open3d vis according to frame_cfg.

        Args:
            frame_cfg (dict): The config to create coordinate frame in open3d vis.
        Returns:
            :obj:`o3d.visualization.Visualizer`: Created open3d vis.
        """
        if o3d is None or geometry is None:
            raise ImportError(
                'Please run "pip install open3d" to install open3d first.'
            )
        o3d_vis = o3d.visualization.Visualizer()
        o3d_vis.create_window(width=1500, height=2332)
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(**frame_cfg)
        o3d_vis.add_geometry(mesh_frame)
        return o3d_vis

    @master_only
    def set_points(
        self,
        points: np.ndarray,
        pcd_mode: int = 0,
        vis_mode: str = "replace",
        frame_cfg: dict = None,
        points_color: Tuple[float] = (0.8, 0.8, 0.8),
        points_size: int = 0.2,
    ):
        """
        Set the point cloud to draw
        Args:
            points (np.ndarray): Points to visualize with shape (N, 3+C).
            pcd_mode (int): The point cloud mode (coordinates): 0 represents LiDAR, 1 represents CAMERA,
            2 represents Depth. Defaults to 0.
            vis_mode (str): The visualization mode in Open3D:
                - 'replace': Replace the existing point cloud with input point cloud.
                - 'add': Add input point cloud into existing point cloud.
                Defaults to 'replace'.
            frame_cfg (dict): The coordinate frame config for Open3D visualization initialization.
                Defaults to dict(size=1, origin=[0, 0, 0]).
            points_color (Tuple[float]): The color of points.
                Defaults to (1, 1, 1).
            points_size (int): The size of points to show on visualizer.
                Defaults to 2.
        Returns:

        """
        if frame_cfg is None:
            frame_cfg = dict(size=1, origin=[0, 0, 0])
        assert points is not None
        assert vis_mode in ("replace", "add")
        # assert pcd_mode == 0
        check_type("points", points, np.ndarray)
        if not hasattr(self, "o3d_vis"):
            self.o3d_vis = self._initialize_o3d_vis(frame_cfg)

        # we convert points into depth mode for visualization
        if pcd_mode != Coord3DMode.DEPTH:
            points = Coord3DMode.convert(points, pcd_mode, Coord3DMode.DEPTH)

        if hasattr(self, "pcd") and vis_mode != "add":
            self.o3d_vis.remove_geometry(self.pcd)

        # set points size in Open3D
        render_option = self.o3d_vis.get_render_option()
        if render_option is not None:
            render_option.point_size = points_size
            # render_option.background_color = np.asarray([1., 1., 1.])  # white
            render_option.background_color = np.asarray([0.0, 0.0, 0.0])  # black
            render_option.line_width = 10
            render_option.show_coordinate_frame = False  # FOR DEBUG USE

        vis_pcd = geometry.PointCloud()

        pcd = points.copy()
        pcd, pcd_color = self.color_encoding(pcd, mode="v2v4real")
        # normalize to [0, 1] for Open3D drawing
        # if not ((pcd_intcolor >= 0.0) & (pcd_intcolor <= 1.0)).all():
        #     pcd_intcolor /= 255.0

        vis_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
        vis_pcd.colors = o3d.utility.Vector3dVector(pcd_color)

        self.o3d_vis.add_geometry(vis_pcd)
        self.pcd = vis_pcd
        self.points_colors = pcd_color

    # noinspection PyArgumentList
    @staticmethod
    def _get_yaw_arrow(center, rot_mat):
        z2x_rot_max = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        yaw_arrow_meth = geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.1,
            cone_radius=0.15,
            cylinder_height=0.5,
            cone_height=0.4,
            resolution=5,
        )
        yaw_arrow_meth.rotate(z2x_rot_max)
        yaw_arrow_meth.rotate(rot_mat)
        yaw_arrow_meth.translate(center)
        return yaw_arrow_meth

    def draw_bboxes_3d(
        self,
        bboxes_3d: BaseInstance3DBoxes,
        bbox_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        points_in_box_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        rot_axis: int = 2,
        vis_mode: str = "replace",
        show_direction: bool = True,
    ):
        """
        Draw bbox on visualizer and change the color of points inside bbox3d.
        Args:
            vis_mode (str): The visualization mode in Open3D:
                - 'replace': Replace the existing point cloud with input point cloud.
                - 'add': Add input point cloud into existing point cloud.
                Defaults to 'replace'.
        """
        # Before visualizing the 3D Boxes in point cloud scene, we need to convert the boxes to Depth mode
        check_type("bboxes", bboxes_3d, BaseInstance3DBoxes)

        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            bboxes_3d = bboxes_3d.convert_to(Box3DMode.DEPTH)

        # convert bboxes to numpy dtype
        bboxes_3d = tensor2ndarray(bboxes_3d.tensor)

        in_box_color = np.array(points_in_box_color)
        if hasattr(self, "bboxes_3d") and vis_mode != "add":
            for b in self.bboxes_3d:
                self.o3d_vis.remove_geometry(b)

        if not hasattr(self, "bboxes_3d"):
            self.bboxes_3d = []

        for i in range(len(bboxes_3d)):
            center = bboxes_3d[i, 0:3]
            dim = bboxes_3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = bboxes_3d[i, 6]
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

            # bottom center to gravity center
            center[rot_axis] += dim[rot_axis] / 2
            box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
            # noinspection PyArgumentList
            line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
            line_set.paint_uniform_color(np.array(bbox_color))
            self.o3d_vis.add_geometry(line_set)
            self.bboxes_3d.append(line_set)

            if show_direction:
                yaw_arrow_meth = self._get_yaw_arrow(center, rot_mat)
                yaw_arrow_meth.paint_uniform_color(np.array(bbox_color) / 2)
                self.o3d_vis.add_geometry(yaw_arrow_meth)
                self.bboxes_3d.append(yaw_arrow_meth)

            # # change the color of points which are in the box
            # if self.pcd is not None:
            #     indices = box3d.get_point_indices_within_bounding_box(
            #         self.pcd.points)
            #     self.points_colors[indices] = in_box_color

        # update points colors
        if self.pcd is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(self.points_colors)
            self.o3d_vis.update_geometry(self.pcd)

    @master_only
    def draw_proj_bboxes_3d(
        self,
        bboxes_3d: BaseInstance3DBoxes,
        lidar2img: np.ndarray,
        edge_colors: Union[str, Tuple[int], List[Union[str, Tuple[int]]]] = "royalblue",
        line_styles: Union[str, List[str]] = "-",
        line_widths: Union[int, float, List[Union[int, float]]] = 2,
        alpha: Union[int, float] = 0.4,
        img_size: Tuple[int, int] = None,
    ):
        """
        Draw projected 3D boxes on the image
        Args:
            bboxes_3d (BaseInstance3DBoxes): 3D boxes to be drawn
            lidar2img (np.ndarray): lidar to image transformation matrix
            edge_colors (Union[str, Tuple[int], List[Union[str, Tuple[int]]]]): edge colors of boxes
            alpha (Union[int, float]): transparency of boxes
            img_size (Tuple[int, int]): image size (w, h)
        """
        check_type("bboxes", bboxes_3d, BaseInstance3DBoxes)
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            proj_bbox3d_to_img = proj_lidar_bbox3d_to_img
        else:
            raise NotImplementedError("unsupported box type!")

        edge_colors_norm = color_val_matplotlib(edge_colors)
        corners_2d = proj_bbox3d_to_img(bboxes_3d, lidar2img)
        # corners_2d = np.asarray([[
        #     [100, 200],
        #     [200, 200],
        #     [200, 100],
        #     [100, 100],
        #     [150, 250],
        #     [250, 250],
        #     [250, 350],
        #     [150, 350],
        # ]], dtype=np.float32)
        if img_size is not None:
            # Filter out the bbox where half of the projected bbox is out of the image.
            # This is for the visualization of multi-view images.
            valid_point_idx = (
                (corners_2d[..., 0] >= 0)
                & (corners_2d[..., 0] <= img_size[0])
                & (corners_2d[..., 1] >= 0)
                & (corners_2d[..., 1] <= img_size[1])
            )
            valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 4
            corners_2d = corners_2d[valid_bbox_idx]
            filter_edge_colors = []
            filter_edge_colors_norm = []
            for i, color in enumerate(edge_colors):
                if valid_bbox_idx[i]:
                    filter_edge_colors.append(color)
                    filter_edge_colors_norm.append(edge_colors_norm[i])
            edge_colors = filter_edge_colors
            edge_colors_norm = filter_edge_colors_norm
        lines_verts_idx = [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 5, 1, 2, 6]
        lines_verts = corners_2d[:, lines_verts_idx, :]
        front_polys = corners_2d[:, 4:, :]
        codes = [Path.LINETO] * lines_verts.shape[1]
        codes[0] = Path.MOVETO
        pathpatches = []
        for i in range(len(corners_2d)):
            verts = lines_verts[i]
            pth = Path(verts, codes)
            pathpatches.append(PathPatch(pth))
        p = PatchCollection(
            pathpatches,
            facecolors="none",
            edgecolors=edge_colors_norm,
            linewidths=line_widths,
            linestyles=line_styles,
        )
        self.ax_save.add_collection(p)
        front_polys = [front_poly for front_poly in front_polys]
        return self.draw_polygons(
            front_polys,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=edge_colors,
        )

    @master_only
    def show(
        self,
        drawn_img_3d: Optional[np.ndarray] = None,
        win_name: str = "image",
        wait_time: int = 0,
    ):
        if hasattr(self, "_image"):
            if drawn_img_3d is not None:
                super().show(drawn_img_3d, win_name, 0, " ", backend="cv2")

        if hasattr(self, "o3d_vis"):
            self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()
            # save the image with current timestamp
            image_path = f"{time.time()}.png"
            self.o3d_vis.capture_screen_image(image_path)
            if wait_time > 0:
                time.sleep(wait_time)
            else:
                self.o3d_vis.run()

    def clear(self):
        """Clear open3d vis."""
        if hasattr(self, "o3d_vis"):
            self.o3d_vis.clear_geometries()
            self.o3d_vis.destroy_window()
            self.o3d_vis.close()
            del self.o3d_vis
            del self.pcd
            del self.points_colors
            del self.bboxes_3d

    def draw_instances_3d(
        self,
        data_input: dict,
        instances: InstanceData,
        input_meta: dict = None,
        bbox_color=(0.0, 1.0, 0.0),
        vis_task: str = "lidar_det",
    ):
        if not len(instances) > 0:
            return None
        bboxes_3d = instances.bboxes_3d
        labels_3d = instances.labels_3d
        data_3d = dict()
        if vis_task in ["lidar_det", "multi-modality_det"]:
            assert "points" in data_input
            points = data_input["points"]
            check_type("points", points, (np.ndarray, Tensor))
            points = tensor2ndarray(points)
            if not isinstance(bboxes_3d, DepthInstance3DBoxes):
                points, bboxes_3d_depth = to_depth_mode(points, bboxes_3d)
            else:
                bboxes_3d_depth = bboxes_3d.clone()
            self.set_points(points, pcd_mode=2)
            self.draw_bboxes_3d(bboxes_3d_depth, bbox_color, show_direction=True)
            data_3d["bboxes_3d"] = tensor2ndarray(bboxes_3d_depth.tensor)
            data_3d["points"] = points

        if vis_task in ["mono_det", "multi-modality_det"]:
            assert "img" in data_input
            img = data_input["img"]
            if isinstance(img, list) or (
                isinstance(img, (np.ndarray, Tensor)) and len(img.shape) == 4
            ):
                # show multi-view images
                img_size = img[0].shape[:2] if isinstance(img, list) else img.shape[-2:]
                img_col = self.multi_imgs_col
                img_row = math.ceil(len(img) / img_col)
                composed_img = np.zeros(
                    (img_size[0] * img_row, img_size[1] * img_col, 3), dtype=np.uint8
                )
                for i, single_img in enumerate(img):
                    if isinstance(single_img, Tensor):
                        single_img = single_img.permute(1, 2, 0).numpy()  # H, W, C
                        single_img = single_img[..., [2, 1, 0]]  # bgr to rgb
                    self.set_image(single_img)
                    single_img_meta = dict()
                    for key, meta in input_meta.items():
                        if isinstance(meta, (Sequence, np.ndarray, Tensor)) and len(
                            meta
                        ) == len(img):
                            single_img_meta[key] = meta[i]
                        else:
                            single_img_meta[key] = meta

                    self.draw_proj_bboxes_3d(
                        bboxes_3d,
                        single_img_meta["lidar2img"],
                        edge_colors=[(0, 0, 0)],
                        img_size=single_img.shape[:2][::-1],
                    )
                    composed_img[
                        (i // img_col) * img_size[0] : (i // img_col + 1) * img_size[0],
                        (i % img_col) * img_size[1] : (i % img_col + 1) * img_size[1],
                    ] = self.get_image()
                data_3d["img"] = composed_img
            else:
                raise NotImplementedError
        return data_3d

    @master_only
    def add_datasample(
        self,
        data_input: dict,
        data_sample: Optional[Det3DDataSample] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = True,
        wait_time: int = 0,
        pred_score_thr: float = 0.2,
        step: int = 0,
    ):
        """Draw datasample.

        Args:
            data_input (dict): It should include the point clouds or image to draw.
            data_sample (:obj:`Det3DDataSample`, optional): Prediction Det3DDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT Det3DDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Prediction Det3DDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn point clouds and image.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0, which means wait until the window closed.
            pred_score_thr (float): The threshold to visualize the bboxes and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
            vis_task (str): Visualization task. Defaults to 'lidar_det'.
        """
        if not show:
            return

        points = data_input["points"]
        # sample_idx = data_sample.sample_idx
        gt_instances_3d = data_sample.gt_instances_3d.to("cpu")
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        pred_instances_3d = data_sample.pred_instances_3d
        pred_instances_3d = pred_instances_3d[
            pred_instances_3d.scores_3d > pred_score_thr
        ].to("cpu")
        pred_bboxes_3d = pred_instances_3d.bboxes_3d

        self.set_points(points)
        if draw_gt:
            self.draw_bboxes_3d(
                gt_bboxes_3d,
                bbox_color=(1.0, 0.0, 0.0),  # GT bbxes are red
                show_direction=True,
            )
        if draw_pred:
            self.draw_bboxes_3d(
                pred_bboxes_3d,
                bbox_color=(0.0, 1.0, 0.0),  # Pred bbxes are green
                vis_mode="add",
                show_direction=True,
            )
        self.show(wait_time)
        if wait_time <= 0:
            self.clear()

    @staticmethod
    def format_color(color: str):
        """Format color from hex to numpy float64 array in range [0, 1], shape
        (3,)."""
        color = color.lstrip("#")
        return np.array(tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))) / 255.0

    @staticmethod
    def color_encoding(points, mode="v2v4real", ego_only=False):
        """Encode the single-channel intensity to 3 channels rgb color.

        Parameters
        ----------
        x : np.ndarray
            shape (n,)

        mode : str
            The color rendering mode. intensity, z-value and constant are supported.

        Returns
        -------
        color : np.ndarray
            Encoded Lidar color, shape (n, 3)
        """
        assert mode in ["v2v4real"]
        if mode == "v2v4real":
            palette = [
                "#a7f2a7",
                "#f2a7a7",
                "#f2f2a7",
                "#a7a7f2",
            ]  # green, red, yellow, blue
            cav_ids = np.unique(points[:, -1].astype(np.int32))
            points_list = []
            for idx, label in enumerate(cav_ids):
                points_list.append(points[points[:, -1] == label])
            points_list.sort(key=lambda x: np.abs(np.mean(x[:, :2])))
            cav_ids = [int(p[0, -1]) for p in points_list]
            if not ego_only:
                color = np.zeros((points.shape[0], 3))
                for idx, label in enumerate(cav_ids):
                    color[points[:, -1] == label] = V2V4RealVisualizer.format_color(
                        palette[idx]
                    )
                return points, color
            else:
                points = points_list[0]
                color = np.zeros((points.shape[0], 3))
                color[:] = V2V4RealVisualizer.format_color(palette[0])
                return points, color
