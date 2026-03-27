import numpy as np
import random
import math
from mmdet3d.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from .transforms_3d import FloorplanPointSample_
from mmdet3d.datasets.transforms import LoadPointsFromFile
from mmdet3d.structures.points import get_points_type


@TRANSFORMS.register_module()
class LoadPointceptPointsFromFile(LoadPointsFromFile):
    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results["lidar_points"]["lidar_path"]
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert (
                len(self.use_dim) >= 4
            ), f"When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}"  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert (
                len(self.use_dim) >= 5
            ), f"When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}"  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points[:, :3] = points[:, :3] / 140
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results


@TRANSFORMS.register_module()
class PointceptPreprocess(FloorplanPointSample_):

    def _points_random_sampling(self, points, num_samples):
        """Points random sampling. Sample points to a certain number.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:
                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        # point_range = range(len(points))
        # choices = np.random.choice(point_range, min(num_samples, len(points)))

        mask = (points.coord[:, 0] <= 1) & (points.coord[:, 1] <= 1)
        # # 首先，创建一个与 mask 等长的索引数组
        # indices = np.arange(len(mask))
        # # 然后，找出 mask 为 True 的索引位置
        # mask_indices = indices[mask]
        # # 最后，从 choice 中筛选出同时出现在 mask_indices 中的元素
        # intersection = choices[np.isin(choices, mask_indices)]

        return points[mask], mask


@TRANSFORMS.register_module()
class PointceptDataAugmentation(BaseTransform):
    def __init__(
        self,
        aug_prob=0.5,
        hflip=True,
        vflip=True,
        rotate_enable=False,
        rotate_angle=(-180, 180),
        rotate2=True,
        scale_enable=True,
        scale_ratio=(0.5, 1.5),
        shift_enable=True,
        shift_scale=(-0.5, 0.5),
        cutmix_enable=False,
        cutmix_queueK=32,
        cutmix_relative_shift=(-0.5, 0.5),
    ):
        self.aug_prob = aug_prob
        self.hflip = hflip
        self.vflip = vflip
        self.rotate_enable = rotate_enable
        self.rotate_angle = rotate_angle
        self.rotate2 = rotate2
        self.scale_enable = scale_enable
        self.scale_ratio = scale_ratio
        self.shift_enable = shift_enable
        self.shift_scale = shift_scale
        self.cutmix_enable = cutmix_enable
        self.cutmix_queueK = cutmix_queueK
        self.cutmix_relative_shift = cutmix_relative_shift

        self.instance_queues = []

    def normalize_coords(self, coords):
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        normalized_coords = (coords - min_coords) / (max_coords - min_coords)
        return normalized_coords

    def rotate_xy(self, points, width=1, height=1, angle=None):
        if angle is None:
            angle = random.uniform(*self.rotate_angle)
        center_x, center_y = width / 2, height / 2

        shifted_points = points - np.array([center_x, center_y])
        pi_angle = angle * math.pi / 180

        rotation_matrix = np.array(
            [
                [np.cos(pi_angle), -np.sin(pi_angle)],
                [np.sin(pi_angle), np.cos(pi_angle)],
            ]
        )
        rotated_points = np.dot(shifted_points, rotation_matrix)
        rotated_points = rotated_points + np.array([center_x, center_y])

        return rotated_points

    def random_rotate(self, points, width=1, height=1):
        angle = random.uniform(-180, 180)
        centroid = [width / 2, height / 2]

        points_centered = points - centroid
        angle_radians = np.deg2rad(angle)
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), -np.sin(angle_radians)],
                [np.sin(angle_radians), np.cos(angle_radians)],
            ]
        )
        points_rotated = np.dot(points_centered, rotation_matrix.T) + centroid

        return points_rotated

    def random_shift(self, points, scale_min, scale_max):
        scale = np.random.uniform(scale_min, scale_max, size=3)
        scale[2] = 0
        shifted_points = points + scale
        return shifted_points

    def random_scale(self, points, ratio_min, ratio_max):
        scale = np.random.uniform(ratio_min, ratio_max, 1)
        scaled_points = points * scale
        return scaled_points
        # scaled_features = features.copy()
        # scaled_features[:, 1] = scaled_features[:, 1] * scale
        # return scaled_points, scaled_features

    def horizontal_flip(self, points, width=1):
        points[:, 0] = width - points[:, 0]
        return points

    def vertical_flip(self, points, height=1):
        points[:, 1] = height - points[:, 1]
        return points

    def cutmix(self, coord, label):
        unique_labels = np.unique(label)

        for sem in unique_labels:
            if sem >= 30:
                continue
            valid = label == sem
            if len(self.instance_queues) <= self.cutmix_queueK:
                self.instance_queues.insert(
                    0, {"coord": coord[valid], "label": label[valid]}
                )
            else:
                self.instance_queues.pop()

        if not self.instance_queues:
            return coord, label

        mix_coord, mix_label = [], []
        mix_coord.append(coord)
        mix_label.append(label)

        rand_pos = np.random.uniform(*self.cutmix_relative_shift, 3)
        rand_pos[2] = 0
        for instance in self.instance_queues:
            mix_coord.append(instance["coord"] + rand_pos)
            mix_label.append(instance["label"])

        if not mix_coord or not mix_label:
            return coord, label

        coord = np.concatenate(mix_coord, axis=0)
        label = np.concatenate(mix_label, axis=0)

        return coord, label

    def transform(self, input_dict):
        coord = np.array(input_dict["points"].coord)

        coord[:, :2] = self.normalize_coords(coord[:, :2])
        # Horizontal Flip
        # * random
        if self.hflip and np.random.rand() < self.aug_prob:
            coord[:, :2] = self.horizontal_flip(coord[:, :2])

        # Vertical Flip
        if self.vflip and np.random.rand() < self.aug_prob:
            coord[:, :2] = self.vertical_flip(coord[:, :2])

        # Rotate
        if self.rotate_enable and np.random.rand() < self.aug_prob:
            coord[:, :2] = self.rotate_xy(coord[:, :2])

        # Extra Rotate
        if self.rotate2 and np.random.rand() < self.aug_prob:
            coord[:, :2] = self.random_rotate(coord[:, :2])

        # Random Shift
        if self.shift_enable and np.random.rand() < self.aug_prob:
            coord = self.random_shift(coord, *self.shift_scale)

        # Random Scale
        if self.scale_enable and np.random.rand() < self.aug_prob:
            # coord, feat = self.random_scale(coord, *self.scale_ratio, feat)
            coord = self.random_scale(coord, *self.scale_ratio)

        # # Cutmix
        # if self.cutmix_enable and np.random.rand() < self.aug_prob:
        #     coord, label = self.cutmix(coord, label)

        # # Shuffle
        # shuf_idx = np.arange(coord.shape[0])
        # np.random.shuffle(shuf_idx)
        # coord = coord[shuf_idx]
        # # feat = feat[shuf_idx] if feat is not None else None
        # label = label[shuf_idx] if label is not None else None

        # Coordinate Normalization
        coord -= np.mean(coord, 0)

        # Update the data dictionary
        input_dict["points"].coord = coord

        return input_dict


@TRANSFORMS.register_module()
class SympointTest(object):
    def __init__(self, data_norm=True, x_scale=1, y_scale=1):
        self.data_norm = data_norm
        self.x_scale = x_scale
        self.y_scale = y_scale

    def normalize_coords(self, coords):
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        normalized_coords = (coords - min_coords) / (max_coords - min_coords)
        normalized_coords[:, 0] = normalized_coords[:, 0] * self.x_scale
        normalized_coords[:, 1] = normalized_coords[:, 1] * self.y_scale
        return normalized_coords

    def __call__(self, input_dict):
        if self.data_norm:
            coords = np.array(input_dict["points"].coord)
            coords[:, :2] = self.normalize_coords(coords[:, :2])
            input_dict["points"].coord = coords
        return input_dict
