from os import path as osp
import numpy as np
import random
from typing import Callable, List, Optional, Union
from oneformer3d import ScanNetSegDataset_
from mmdet3d.datasets.seg3d_dataset import Seg3DDataset
from mmdet3d.datasets.s3dis_dataset import S3DISDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class S3DFloorplanSegDataset_(S3DISDataset):
    METAINFO = {
        "classes": (
            "single door",
            "double door",
            "sliding door",
            "folding door",
            "revolving door",
            "rolling door",
            "window",
            "bay window",
            "blind window",
            "opening symbol",
            "sofa",
            "bed",
            "chair",
            "table",
            "TV cabinet",
            "Wardrobe",
            "cabinet",
            "gas stove",
            "sink",
            "refrigerator",
            "airconditioner",
            "bath",
            "bath tub",
            "washing machine",
            "squat toilet",
            "urinal",
            "toilet",
            "stairs",
            "elevator",
            "escalator",
            "row chairs",
            "parking spot",
            "wall",
            "curtain wall",
            "railing",
            "bg",
        ),
        "palette": [random.sample(range(0, 255), 3) for i in range(36)],
        "seg_valid_class_ids": tuple(range(36)),
        "seg_all_class_ids": tuple(range(36)),  # possibly with 'stair' class
    }

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info["pts_instance_mask_path"] = osp.join(
            self.data_prefix.get("pts_instance_mask", ""),
            info["pts_instance_mask_path"],
        )
        info["pts_semantic_mask_path"] = osp.join(
            self.data_prefix.get("pts_semantic_mask", ""),
            info["pts_semantic_mask_path"],
        )
        if self.data_prefix.get("pts_reverse_map", None):
            info["pts_reverse_map_path"] = osp.join(
                self.data_prefix.get("pts_reverse_map", None),
                info["pts_reverse_map_path"],
            )

        info = super(S3DISDataset, self).parse_data_info(info)
        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info["seg_label_mapping"] = self.seg_label_mapping
        return info


@DATASETS.register_module()
class S3DFloorplanSegDataset_thing(S3DFloorplanSegDataset_):
    METAINFO = {
        "classes": (
            "single door",
            "double door",
            "sliding door",
            "folding door",
            "revolving door",
            "rolling door",
            "window",
            "bay window",
            "blind window",
            "opening symbol",
            "sofa",
            "bed",
            "chair",
            "table",
            "TV cabinet",
            "Wardrobe",
            "cabinet",
            "gas stove",
            "sink",
            "refrigerator",
            "airconditioner",
            "bath",
            "bath tub",
            "washing machine",
            "squat toilet",
            "urinal",
            "toilet",
            "stairs",
            "elevator",
            "escalator",
        ),
        "palette": [random.sample(range(0, 255), 3) for i in range(30)],
        "seg_valid_class_ids": tuple(range(30)),
        "seg_all_class_ids": tuple(range(30)),  # possibly with 'stair' class
    }


@DATASETS.register_module()
class FloorscanSegDataset(ScanNetSegDataset_):

    METAINFO = {
        "classes": (
            "single door",
            "double door",
            "sliding door",
            "folding door",
            "revolving door",
            "rolling door",
            "window",
            "bay window",
            "blind window",
            "opening symbol",
            "sofa",
            "bed",
            "chair",
            "table",
            "TV cabinet",
            "Wardrobe",
            "cabinet",
            "gas stove",
            "sink",
            "refrigerator",
            "airconditioner",
            "bath",
            "bath tub",
            "washing machine",
            "squat toilet",
            "urinal",
            "toilet",
            "stairs",
            "elevator",
            "escalator",
            "row chairs",
            "parking spot",
            "wall",
            "curtain wall",
            "railing",
            "bg",
        ),
        "palette": [random.sample(range(0, 255), 3) for i in range(36)],
        "seg_valid_class_ids": tuple(range(36)),
        "seg_all_class_ids": tuple(range(36)),
    }
