from mmcv.transforms.base import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type


@TRANSFORMS.register_module()
class LoadPointsNumpy(BaseTransform):

    def __init__(
        self,
        coord_type: str = 'LIDAR',
        load_dim: int = 4,
        use_dim=None,
    ):
        super().__init__()
        self.coord_type = coord_type
        if use_dim is None:
            use_dim = [0, 1, 2, 3]
        self.load_dim = load_dim
        self.use_dim = use_dim

    def transform(self, results: dict) -> dict:
        points = results['points']
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        results['points'] = points
        return results
