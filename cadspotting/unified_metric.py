import torch
import numpy as np

from mmengine.logging import MMLogger

from mmdet3d.evaluation import InstanceSegMetric
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from .instance_seg_eval import instance_seg_eval
from .semantic_seg_eval import semantic_seg_eval, semantic_F1_eval
from .sympoint_metric import (
    sympoint_instance_inference_eval,
    sympoint_semantic_F1_eval,
    sympoint_semantic_eval,
    sympoint_instance_eval,
    sympoint_instance_eval_delete,
    sympoint_instance_eval_delete_1,
    sympoint_instance_eval_new,
)
import os
from tqdm import tqdm


@METRICS.register_module()
class UnifiedSegMetric(SegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(
        self,
        thing_class_inds,
        stuff_class_inds,
        min_num_points,
        id_offset,
        sem_mapping,
        inst_mapping,
        metric_meta,
        logger_keys=[("miou",), ("all_ap", "all_ap_50%", "all_ap_25%"), ("pq",)],
        **kwargs,
    ):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.logger_keys = logger_keys
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]
        label2cat = self.metric_meta["label2cat"]
        ignore_index = self.metric_meta["ignore_index"]
        classes = self.metric_meta["classes"]
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)

        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        for eval_ann, single_pred_results in results:

            if (
                self.metric_meta["dataset_name"] == "S3DIS"
                or self.metric_meta["dataset_name"] == "Floorplan"
            ):
                pan_gt = {}
                pan_gt["pts_semantic_mask"] = eval_ann["pts_semantic_mask"]
                pan_gt["pts_instance_mask"] = eval_ann["pts_instance_mask"].copy()

                for stuff_cls in self.stuff_class_inds:
                    pan_gt["pts_instance_mask"][
                        pan_gt["pts_semantic_mask"] == stuff_cls
                    ] = (np.max(pan_gt["pts_instance_mask"]) + 1)

                pan_gt["pts_instance_mask"] = np.unique(
                    pan_gt["pts_instance_mask"], return_inverse=True
                )[1]
                gt_masks_pan.append(pan_gt)
            else:
                gt_masks_pan.append(eval_ann)

            pred_masks_pan.append(
                {
                    "pts_instance_mask": single_pred_results["pts_instance_mask"][1],
                    "pts_semantic_mask": single_pred_results["pts_semantic_mask"][1],
                }
            )

            gt_semantic_masks_sem_task.append(eval_ann["pts_semantic_mask"])
            pred_semantic_masks_sem_task.append(
                single_pred_results["pts_semantic_mask"][0]
            )

            if (
                self.metric_meta["dataset_name"] == "S3DIS"
                or self.metric_meta["dataset_name"] == "Floorplan"
            ):
                gt_semantic_masks_inst_task.append(eval_ann["pts_semantic_mask"])
                gt_instance_masks_inst_task.append(eval_ann["pts_instance_mask"])
            else:
                sem_mask, inst_mask = self.map_inst_markup(
                    eval_ann["pts_semantic_mask"].copy(),
                    eval_ann["pts_instance_mask"].copy(),
                    self.valid_class_ids[num_stuff_cls:],
                    num_stuff_cls,
                )
                gt_semantic_masks_inst_task.append(sem_mask)
                gt_instance_masks_inst_task.append(inst_mask)

            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results["pts_instance_mask"][0])
            )
            pred_instance_labels.append(
                torch.tensor(single_pred_results["instance_labels"])
            )
            pred_instance_scores.append(
                torch.tensor(single_pred_results["instance_scores"])
            )

        ret_pan = panoptic_seg_eval(
            gt_masks_pan,
            pred_masks_pan,
            classes,
            thing_classes,
            stuff_classes,
            self.min_num_points,
            self.id_offset,
            label2cat,
            ignore_index,
            logger,
        )

        ret_sem = seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index[0],
            logger=logger,
        )

        if (
            self.metric_meta["dataset_name"] == "S3DIS"
            or self.metric_meta["dataset_name"] == "Floorplan"
        ):
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids,
                class_labels=classes[:-1],
                logger=logger,
            )
        else:
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger,
            )

        metrics = dict()
        for ret, keys in zip((ret_sem, ret_inst, ret_pan), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics

    def map_inst_markup(
        self, pts_semantic_mask, pts_instance_mask, valid_class_ids, num_stuff_cls
    ):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes

        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]

        return pts_semantic_mask, pts_instance_mask


@METRICS.register_module()
class UnifiedSegMetric_Entity_Oneformer(UnifiedSegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(
        self,
        npz_path,
        ignore_bg=False,
        logger_keys=[
            ("miou",),
            ("all_ap", "all_ap_50%", "all_ap_75%"),
            ("pq", "rq_mean", "sq_mean"),
            ("F1", "wF1"),
        ],
        **kwargs,
    ):
        self.npz_path = npz_path
        self.ignore_bg = ignore_bg
        super().__init__(logger_keys=logger_keys, **kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]
        label2cat = self.metric_meta["label2cat"]
        ignore_index = self.metric_meta["ignore_index"]
        if self.ignore_bg:
            ignore_index = ignore_index + [35]
        classes = self.metric_meta["classes"]
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        element_weights = []

        for eval_ann, single_pred_results in results:
            lidar_paths = single_pred_results["lidar_paths"]
            assert len(lidar_paths) == 1
            npz_filename = lidar_paths[0][-13:].split(".")[0]
            # 从gt文件中读取gt_elementIds, gt_elementlengths
            val_or_test = "val"
            if "val" not in lidar_paths[0]:
                val_or_test = "test"
            file_path = f"{self.npz_path}/{npz_filename}.npz"
            file_path = f"{self.npz_path}/{npz_filename}.npz"
            if not os.path.exists(file_path):
                logger.warning("File not exist" + lidar_paths[0])
                continue
            # assert os.path.exists(file_path) == True
            with np.load(file_path, mmap_mode="r") as data:
                logger.info("load file " + file_path)
                gt_elementIds, gt_elementlengths = (
                    data["elementIds"],
                    data["elementlengths"],
                )
            assert len(eval_ann["pts_semantic_mask"]) == len(gt_elementIds)

            # back to entity lengths
            element_lengths = back_element_lengths(
                np.array(gt_elementIds), np.array(gt_elementlengths)
            )
            element_weights.append(np.round(np.log(1 + element_lengths), 3))

            if (
                self.metric_meta["dataset_name"] == "S3DIS"
                or self.metric_meta["dataset_name"] == "Floorplan"
            ):
                pan_gt = {}
                pan_gt["pts_semantic_mask"] = eval_ann["pts_semantic_mask"].copy()
                pan_gt["pts_instance_mask"] = eval_ann["pts_instance_mask"].copy()
                for stuff_cls in self.stuff_class_inds:
                    pan_gt["pts_instance_mask"][
                        pan_gt["pts_semantic_mask"] == stuff_cls
                    ] = (np.max(pan_gt["pts_instance_mask"]) + 1)

                pan_gt["pts_instance_mask"] = np.unique(
                    pan_gt["pts_instance_mask"], return_inverse=True
                )[1]

                # get entity GT panoptic
                pan_gt["pts_semantic_mask"], pan_gt["pts_instance_mask"] = (
                    back_to_entity_panoptic(
                        pan_gt["pts_semantic_mask"].copy(),
                        pan_gt["pts_instance_mask"].copy(),
                        gt_elementIds,
                    )
                )
                gt_masks_pan.append(pan_gt)
            else:
                gt_masks_pan.append(eval_ann)
            # get entity pred panoptic
            (
                single_pred_results["pts_semantic_mask"][1],
                single_pred_results["pts_instance_mask"][1],
            ) = back_to_entity_panoptic(
                single_pred_results["pts_semantic_mask"][1].copy(),
                single_pred_results["pts_instance_mask"][1].copy(),
                gt_elementIds,
            )
            pred_masks_pan.append(
                {
                    "pts_instance_mask": single_pred_results["pts_instance_mask"][1],
                    "pts_semantic_mask": single_pred_results["pts_semantic_mask"][1],
                }
            )

            # get entity GT semantic
            gt_semantic_mask = back_to_entity_semantic(
                eval_ann["pts_semantic_mask"].copy(), gt_elementIds
            )
            gt_semantic_masks_sem_task.append(gt_semantic_mask)
            # get entity pred semantic
            pred_semantic_mask = back_to_entity_semantic(
                single_pred_results["pts_semantic_mask"][0].copy(), gt_elementIds
            )

            pred_semantic_masks_sem_task.append(pred_semantic_mask)

            # get entity GT instance
            if self.metric_meta["dataset_name"] == "S3DIS":
                gt_semantic_masks_inst_task.append(eval_ann["pts_semantic_mask"])
                gt_instance_masks_inst_task.append(eval_ann["pts_instance_mask"])
            elif self.metric_meta["dataset_name"] == "Floorplan":
                sem_mask = eval_ann["pts_semantic_mask"].copy()
                inst_mask = eval_ann["pts_instance_mask"].copy()
                # back to entity
                sem_mask, inst_mask = back_to_entity_panoptic(
                    sem_mask.copy(),
                    inst_mask.copy(),
                    gt_elementIds,
                )
                # print(sem_mask)
                # print(inst_mask)
                inst_mask[sem_mask >= 30] = -1
                sem_mask[inst_mask == -1] = -1
                gt_semantic_masks_inst_task.append(sem_mask)
                gt_instance_masks_inst_task.append(inst_mask)
            else:
                sem_mask, inst_mask = self.map_inst_markup(
                    eval_ann["pts_semantic_mask"].copy(),
                    eval_ann["pts_instance_mask"].copy(),
                    self.valid_class_ids[num_stuff_cls:],
                    num_stuff_cls,
                )
                gt_semantic_masks_inst_task.append(sem_mask)
                gt_instance_masks_inst_task.append(inst_mask)

            # get entity pred instance
            pred_instance_mask, instance_labels, instance_scores = (
                back_to_entity_instance(
                    single_pred_results["pts_instance_mask"][0].copy(),
                    single_pred_results["instance_labels"].copy(),
                    single_pred_results["instance_scores"].copy(),
                    gt_elementIds,
                )
            )
            pred_instance_masks_inst_task.append(torch.tensor(pred_instance_mask))
            pred_instance_labels.append(
                # TODO 预测有错误？instance_labels中出现了stuff_id
                # 原因：predict时, n_instance_classes == num_classes
                torch.tensor(instance_labels)
            )
            pred_instance_scores.append(torch.tensor(instance_scores))

        ret_pan = panoptic_seg_eval(
            gt_masks_pan,
            pred_masks_pan,
            classes,
            thing_classes,
            stuff_classes,
            self.min_num_points,
            self.id_offset,
            label2cat,
            ignore_index,
            logger,
        )

        ret_sem = semantic_seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index,
            logger=logger,
        )

        ret_sem_F1 = semantic_F1_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        if self.metric_meta["dataset_name"] == "S3DIS":
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids,
                class_labels=classes[:-1],
                logger=logger,
            )
        elif self.metric_meta["dataset_name"] == "Floorplan":
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[: -num_stuff_cls + 1],
                class_labels=classes[:-num_stuff_cls],
                options=dict(min_region_sizes=np.array([0]), ap75=True),
                logger=logger,
            )
        else:
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger,
            )

        metrics = dict()
        for ret, keys in zip(
            (ret_sem, ret_inst, ret_pan, ret_sem_F1), self.logger_keys
        ):
            for key in keys:
                metrics[key] = ret[key]
        return metrics


@METRICS.register_module()
class UnifiedSegMetric_Entity_Sympoint(UnifiedSegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(
        self,
        npz_path,
        ignore_bg=False,
        logger_keys=[
            ("F1", "wF1"),
            ("PQ", "SQ", "RQ", "mAP", "AP50", "AP75"),
            ("mIoU", "fwIoU", "pACC"),
        ],
        **kwargs,
    ):
        self.npz_path = npz_path
        self.ignore_bg = ignore_bg
        super().__init__(logger_keys=logger_keys, **kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        assert self.metric_meta["dataset_name"] == "Floorplan"
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]
        label2cat = self.metric_meta["label2cat"]
        ignore_index = self.metric_meta["ignore_index"]
        if self.ignore_bg:
            ignore_index = ignore_index + [35]
        classes = self.metric_meta["classes"]
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        gt_semantic_masks_inst_task_sym = []
        gt_instance_masks_inst_task_sym = []

        gt_semantic_bbox_labels_inst_task_sym = []
        gt_instance_bboxs_inst_task_sym = []

        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []
        pred_instance_masks_inst_task_sym = []
        pred_instance_labels_sym = []
        pred_instance_scores_sym = []

        pred_instance_bboxs_inst_task_sym = []

        # pred_instances_list = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        element_weights = []

        for eval_ann, single_pred_results in results:
            lidar_paths = single_pred_results["lidar_paths"]
            assert len(lidar_paths) == 1
            npz_filename = lidar_paths[0][-13:].split(".")[0]
            # 从gt文件中读取gt_elementIds, gt_elementlengths
            file_path = f"{self.npz_path}/{npz_filename}.npz"
            if not os.path.exists(file_path):
                logger.warning("File not exist " + file_path)
                continue
            # assert os.path.exists(file_path) == True
            with np.load(file_path, mmap_mode="r") as data:
                logger.info("load file " + file_path)
                gt_coords, gt_elementIds, gt_elementlengths = (
                    data["coords"],
                    data["elementIds"],
                    data["elementlengths"],
                )
            assert len(eval_ann["pts_semantic_mask"]) == len(gt_elementIds)

            # back to entity lengths
            element_lengths = back_element_lengths(
                np.array(gt_elementIds), np.array(gt_elementlengths)
            )
            ele_weight = np.round(np.log(1 + element_lengths), 3)
            element_weights.append(ele_weight)

            # oneformer pan gt
            pan_gt = {}
            pan_gt["pts_semantic_mask"] = eval_ann["pts_semantic_mask"].copy()
            pan_gt["pts_instance_mask"] = eval_ann["pts_instance_mask"].copy()
            for stuff_cls in self.stuff_class_inds:
                pan_gt["pts_instance_mask"][
                    pan_gt["pts_semantic_mask"] == stuff_cls
                ] = (np.max(pan_gt["pts_instance_mask"]) + 1)
            pan_gt["pts_instance_mask"] = np.unique(
                pan_gt["pts_instance_mask"], return_inverse=True
            )[1]

            # oneformer get entity GT panoptic
            pan_gt["pts_semantic_mask"], pan_gt["pts_instance_mask"] = (
                back_to_entity_panoptic(
                    pan_gt["pts_semantic_mask"].copy(),
                    pan_gt["pts_instance_mask"].copy(),
                    gt_elementIds,
                )
            )
            # oneformer gt panoptic
            gt_masks_pan.append(pan_gt)

            # oneformer get entity pred panoptic
            (
                single_pred_results["pts_semantic_mask"][1],
                single_pred_results["pts_instance_mask"][1],
            ) = back_to_entity_panoptic(
                single_pred_results["pts_semantic_mask"][1].copy(),
                single_pred_results["pts_instance_mask"][1].copy(),
                gt_elementIds,
            )
            pred_masks_pan.append(
                {
                    "pts_instance_mask": single_pred_results["pts_instance_mask"][1],
                    "pts_semantic_mask": single_pred_results["pts_semantic_mask"][1],
                }
            )

            # oneformer get entity GT semantic
            gt_semantic_mask = back_to_entity_semantic(
                eval_ann["pts_semantic_mask"].copy(), gt_elementIds
            )
            gt_semantic_masks_sem_task.append(gt_semantic_mask)
            # oneformer get entity pred semantic
            pred_semantic_mask = back_to_entity_semantic(
                single_pred_results["pts_semantic_mask"][0].copy(), gt_elementIds
            )
            pred_semantic_masks_sem_task.append(pred_semantic_mask)

            # sympoint get entity GT instance
            sem_mask = eval_ann["pts_semantic_mask"].copy()
            inst_mask = eval_ann["pts_instance_mask"].copy()
            # sympoint back to entity
            sem_mask, inst_mask = back_to_entity_panoptic(
                sem_mask.copy(),
                inst_mask.copy(),
                gt_elementIds,
            )
            gt_bbox_labels, gt_bboxs = back_to_ins_bbox_gt(
                sem_mask.copy(),
                inst_mask.copy(),
                gt_elementIds,
                gt_coords,
            )
            gt_semantic_masks_inst_task_sym.append(sem_mask.copy())
            gt_instance_masks_inst_task_sym.append(inst_mask.copy())
            gt_semantic_bbox_labels_inst_task_sym.append(gt_bbox_labels.copy())
            gt_instance_bboxs_inst_task_sym.append(gt_bboxs.copy())

            # oneformer instance gt
            inst_mask[sem_mask >= 30] = -1
            sem_mask[inst_mask == -1] = -1
            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)

            # onefomer get entity pred instance
            pred_instance_mask, instance_labels, instance_scores = (
                back_to_entity_instance(
                    single_pred_results["pts_instance_mask"][0].copy(),
                    single_pred_results["instance_labels"].copy(),
                    single_pred_results["instance_scores"].copy(),
                    gt_elementIds,
                )
            )
            pred_instance_masks_inst_task.append(torch.tensor(pred_instance_mask))
            pred_instance_labels.append(
                # TODO 预测有错误？instance_labels中出现了stuff_id
                # 原因：predict时, n_instance_classes == num_classes
                torch.tensor(instance_labels)
            )
            pred_instance_scores.append(torch.tensor(instance_scores))

            # sympoint get entity pred for instance
            pred_instance_mask_sym = back_to_entity_instance_sym(
                single_pred_results["pts_instance_mask"][0].copy(),
                gt_elementIds,
            )
            pred_instance_bboxs_sym = back_to_instance_bbox_pred_sym(
                pred_instance_mask_sym,
                gt_elementIds,
                gt_coords,
            )
            pred_instance_masks_inst_task_sym.append(
                torch.tensor(pred_instance_mask_sym)
            )
            pred_instance_bboxs_inst_task_sym.append(pred_instance_bboxs_sym)
            pred_instance_labels_sym.append(
                torch.tensor(single_pred_results["instance_labels"])
            )
            pred_instance_scores_sym.append(
                torch.tensor(single_pred_results["instance_scores"])
            )

            # if 'sym_inst_res' in single_pred_results.keys():
            #     instances_sympoint_inf = single_pred_results['sym_inst_res']
            #     pred_instances_list.append(back_to_entity_instance_symoint_inf(instances_sympoint_inf, gt_elementIds))

        ret_pan = panoptic_seg_eval(
            gt_masks_pan,
            pred_masks_pan,
            classes,
            thing_classes,
            stuff_classes,
            self.min_num_points,
            self.id_offset,
            label2cat,
            ignore_index,
            logger,
        )

        ret_sem = semantic_seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index,
            logger=logger,
        )

        # :-1 for unlabeled
        ret_inst = instance_seg_eval(
            gt_semantic_masks_inst_task,
            gt_instance_masks_inst_task,
            pred_instance_masks_inst_task,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids[: -num_stuff_cls + 1],
            class_labels=classes[:-num_stuff_cls],
            options=dict(min_region_sizes=np.array([0]), ap75=True),
            logger=logger,
        )

        logger.warning(
            "================================>Sympoint Metrics<================================"
        )

        sym_ret_sem = sympoint_semantic_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            classes,
            ignore_index,
            logger=logger,
        )

        # sym_ret_sem_F1
        ret_sem_F1 = sympoint_semantic_F1_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        sym_ret_ins = sympoint_instance_eval(
            gt_semantic_masks_inst_task_sym,
            gt_instance_masks_inst_task_sym,
            pred_instance_masks_inst_task_sym,
            gt_semantic_bbox_labels_inst_task_sym,
            gt_instance_bboxs_inst_task_sym,
            pred_instance_bboxs_inst_task_sym,
            pred_instance_labels_sym,
            pred_instance_scores_sym,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        # if len(pred_instances_list) > 0:
        #     sym_ret_ins_inf = sympoint_instance_inference_eval(
        #         gt_semantic_masks_inst_task_sym,
        #         gt_instance_masks_inst_task_sym,
        #         pred_instances_list,
        #         element_weights,
        #         classes,
        #         ignore_index,
        #         logger=logger,
        #     )

        metrics = dict()
        for ret, keys in zip((ret_sem_F1, sym_ret_ins, sym_ret_sem), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics


@METRICS.register_module()
class InstanceSegMetric_(InstanceSegMetric):
    """The only difference with InstanceSegMetric is that following ScanNet
    evaluator we accept instance prediction as a boolean tensor of shape
    (n_points, n_instances) instead of integer tensor of shape (n_points, ).

    For this purpose we only replace instance_seg_eval call.
    """

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.classes = self.dataset_meta["classes"]
        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]

        gt_semantic_masks = []
        gt_instance_masks = []
        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []

        for eval_ann, single_pred_results in results:
            gt_semantic_masks.append(eval_ann["pts_semantic_mask"])
            gt_instance_masks.append(eval_ann["pts_instance_mask"])
            pred_instance_masks.append(single_pred_results["pts_instance_mask"])
            pred_instance_labels.append(single_pred_results["instance_labels"])
            pred_instance_scores.append(single_pred_results["instance_scores"])

        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            logger=logger,
        )

        return ret_dict


def get_most_common_element(ids):
    """
    找到一个数组中出现次数最多的元素
    """
    unique_elements, counts = np.unique(ids, return_counts=True)
    # 找到出现次数最多的元素的索引
    most_common_index = np.argmax(counts)
    # 找到出现次数最多的元素
    most_common_element = unique_elements[most_common_index]
    return most_common_element


def back_to_entity_panoptic(semanticIds, instanceIds, elementIds):
    """
    将点的预测映射回图元
    args:
        semanticIds: list
        instanceIds: list
        elementIds: list
    """
    # 图元总数
    entities = set(elementIds)
    entity_semantic_ids = []
    entity_instance_ids = []

    for i in list(entities):
        # 该图元中出现次数最多的semid和insid==该图元的semid和insid
        semid = get_most_common_element(semanticIds[elementIds == i])
        insid = get_most_common_element(instanceIds[elementIds == i])
        entity_semantic_ids.append(semid)
        entity_instance_ids.append(insid)

    return np.array(entity_semantic_ids, dtype=np.int32), np.array(
        entity_instance_ids, dtype=np.int32
    )


def back_to_entity_semantic(semanticIds, elementIds):
    # 图元总数
    entities = set(elementIds)
    entity_semantic_ids = []

    for i in list(entities):
        # 该图元中出现次数最多的semid和insid==该图元的semid和insid
        semid = get_most_common_element(semanticIds[elementIds == i])
        entity_semantic_ids.append(semid)

    return np.array(entity_semantic_ids, dtype=np.int32)


def back_to_entity_instance(
    pred_instance_ids, instance_labels, instance_scores, elementIds
):
    entity_instance_ids = []
    all_entity_ids = set(elementIds)

    for raw, label in zip(pred_instance_ids, instance_labels):
        if label >= 30:
            continue
        # elementIds[raw]
        entity_ids = list(set(elementIds[raw]))
        res = np.zeros(len(all_entity_ids))
        res[entity_ids] = 1
        entity_instance_ids.append(res == 1)

    mask = instance_labels < 30
    instance_labels = instance_labels[mask]
    instance_scores = instance_scores[mask]

    return (
        np.array(entity_instance_ids, dtype=np.bool_),
        instance_labels,
        instance_scores,
    )


def back_to_entity_instance_sym(pred_instance_ids, elementIds):
    """
    所有类都保留
    """
    entity_instance_ids = []
    all_entity_ids = set(elementIds)

    for raw in pred_instance_ids:
        # elementIds[raw]
        entity_ids = list(set(elementIds[raw]))
        res = np.zeros(len(all_entity_ids))
        res[entity_ids] = 1
        entity_instance_ids.append(res == 1)

    return np.array(entity_instance_ids, dtype=np.bool_)


def back_to_instance_bbox_pred_sym1(instance_masks, element_bbox):
    bboxs = []
    for mask in instance_masks:
        bbox = (
            element_bbox[mask][:, 0].min().item(),
            element_bbox[mask][:, 1].min().item(),
            element_bbox[mask][:, 2].max().item(),
            element_bbox[mask][:, 3].max().item(),
        )
        bboxs.append(bbox)
    return bboxs


def back_to_instance_bbox_pred_sym(instance_masks, elementIds, coords):
    """
    将每一个instance的mask转换为bbox
    """
    entities = set(elementIds.tolist())
    entities = np.array(list(entities))
    bboxs = []

    for mask in instance_masks:
        ins_coords = []
        for eid in entities[mask].tolist():
            ins_coords += coords[elementIds == eid].tolist()
        bbox = compute_bbox_from_coords(ins_coords)
        bboxs.append(bbox)

    return bboxs


def back_to_entity_instance_symoint_inf(instances, elementIds):
    """
    使用sympoint inference method
    """
    all_entity_ids = set(elementIds)

    for item in instances:
        entity_ids = list(set(elementIds[item["masks"].copy()]))
        res = np.zeros(len(all_entity_ids))
        res[entity_ids] = 1
        item["masks"] = res == 1

    return instances


def compute_bbox_from_coords(coords):
    """
    计算一个坐标列表的边界框。

    参数:
    coords (list of tuples): 坐标列表，每个坐标是一个 (x, y) 元组。

    返回:
    tuple: 边界框的左上角和右下角坐标，格式为 ((min_x, min_y), (max_x, max_y))。
    """
    if not coords:
        raise ValueError("坐标列表不能为空")

    # 初始化最小和最大值
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    # 遍历坐标列表，更新最小和最大值
    for x, y in coords:
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    return min_x, min_y, max_x, max_y


def back_to_ins_bbox_gt(entity_semids, entity_insids, elementIds, coords):
    """
    return {
        "labels": [1, 2, 3, 4]
        "bboxs": [
            [x1, y1, x2, y2], [x1, y1, x2, y2], ..., [x1, y1, x2, y2]
        ]
    }
    """
    # 图元总数
    entities = set(elementIds.tolist())
    assert len(entities) == len(entity_semids) == len(entity_insids)
    labels = []
    bboxs = []

    entity_semids = np.array(entity_semids)
    entity_insids_set = list(set(entity_insids))
    entity_insids = np.array(entity_insids)
    entities = np.array(list(entities))
    # 获取实例的label和bbox
    for id in entity_insids_set:
        label = np.unique(entity_semids[entity_insids == id])
        assert len(label) == 1
        label = label.tolist()[0]
        ins_coords = []
        for eid in entities[entity_insids == id].tolist():
            ins_coords += coords[elementIds == eid].tolist()
        labels.append(label)
        bbox = compute_bbox_from_coords(ins_coords)
        bboxs.append(bbox)

    return labels, bboxs


def back_element_lengths(elementIds, elementLengths):
    all_entity_ids = set(elementIds)
    res = []

    for i in all_entity_ids:
        id_to_len = set(elementLengths[elementIds == i])
        assert len(id_to_len) == 1
        res.append(id_to_len.pop())

    return np.array(res, dtype=np.float32)


from PIL import Image, ImageDraw


def draw_bboxes(
    image_path,
    bboxes,
    output_path,
    colors=None,
    widths=None,
    labels=None,
    target_size=(140, 140),
):
    """
    在图像上绘制多个边界框并保存图像。

    参数:
    image_path (str): 图像文件路径。
    bboxes (list of tuples): 边界框列表，每个边界框格式为 (x1, y1, x2, y2)。
    output_path (str): 保存图像的路径。
    colors (list of str): 边界框的颜色列表，默认为 None（使用红色）。
    widths (list of int): 边界框的线宽列表，默认为 None（使用 2）。
    labels (list of str): 边界框的标签列表，默认为 None。
    """
    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 设置默认颜色和线宽
    if colors is None:
        colors = ["red"] * len(bboxes)
    if widths is None:
        widths = [2] * len(bboxes)
    if labels is None:
        labels = [None] * len(bboxes)

    # 绘制每个边界框
    for bbox, color, width, label in zip(bboxes, colors, widths, labels):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = x1 * 5, y1 * 5, x2 * 5, y2 * 5
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # 如果提供了标签，绘制标签
        if label:
            text_color = "white" if color == "red" else "black"
            draw.text((x1, y1 - 10), str(label), fill=text_color)

    # 保存图像
    image.save(output_path)
    print(f"图像已保存到 {output_path}")


@METRICS.register_module()
class UnifiedSegMetric_PointVisual(UnifiedSegMetric_Entity_Oneformer):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    SVG_CATEGORIES = [
        {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "single door"},
        {"color": [157, 34, 101], "isthing": 1, "id": 2, "name": "double door"},
        {"color": [232, 116, 91], "isthing": 1, "id": 3, "name": "sliding door"},
        {"color": [101, 54, 72], "isthing": 1, "id": 4, "name": "folding door"},
        {"color": [172, 107, 133], "isthing": 1, "id": 5, "name": "revolving door"},
        {"color": [142, 76, 101], "isthing": 1, "id": 6, "name": "rolling door"},
        {"color": [96, 78, 245], "isthing": 1, "id": 7, "name": "window"},
        {"color": [26, 2, 219], "isthing": 1, "id": 8, "name": "bay window"},
        {"color": [63, 140, 221], "isthing": 1, "id": 9, "name": "blind window"},
        {"color": [233, 59, 217], "isthing": 1, "id": 10, "name": "opening symbol"},
        {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "sofa"},
        {"color": [94, 150, 113], "isthing": 1, "id": 12, "name": "bed"},
        {"color": [66, 107, 81], "isthing": 1, "id": 13, "name": "chair"},
        {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "table"},
        {"color": [94, 150, 83], "isthing": 1, "id": 15, "name": "TV cabinet"},
        {"color": [66, 107, 59], "isthing": 1, "id": 16, "name": "Wardrobe"},
        {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "cabinet"},
        {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "gas stove"},
        {"color": [113, 151, 82], "isthing": 1, "id": 19, "name": "sink"},
        {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "refrigerator"},
        {"color": [81, 107, 58], "isthing": 1, "id": 21, "name": "airconditioner"},
        {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "bath"},
        {"color": [141, 152, 83], "isthing": 1, "id": 23, "name": "bath tub"},
        {"color": [80, 72, 147], "isthing": 1, "id": 24, "name": "washing machine"},
        {"color": [100, 108, 59], "isthing": 1, "id": 25, "name": "squat toilet"},
        {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "urinal"},
        {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "toilet"},
        {"color": [247, 206, 75], "isthing": 1, "id": 28, "name": "stairs"},
        {"color": [237, 112, 45], "isthing": 1, "id": 29, "name": "elevator"},
        {"color": [233, 59, 46], "isthing": 1, "id": 30, "name": "escalator"},
        {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "row chairs"},
        {"color": [102, 67, 62], "isthing": 0, "id": 32, "name": "parking spot"},
        {"color": [167, 92, 32], "isthing": 0, "id": 33, "name": "wall"},
        {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "curtain wall"},
        {"color": [64, 52, 105], "isthing": 0, "id": 35, "name": "railing"},
        {"color": [0, 0, 0], "isthing": 0, "id": 36, "name": "bg"},
    ]

    def __init__(
        self,
        saved_dir=None,
        **kwargs,
    ):
        self.saved_dir = saved_dir
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]
        label2cat = self.metric_meta["label2cat"]
        ignore_index = self.metric_meta["ignore_index"]
        if self.ignore_bg:
            ignore_index = ignore_index + [35]
        classes = self.metric_meta["classes"]
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        element_weights = []

        for eval_ann, single_pred_results in results:
            lidar_paths = single_pred_results["lidar_paths"]
            assert len(lidar_paths) == 1
            npz_filename = lidar_paths[0][-13:].split(".")[0]
            # 从gt文件中读取gt_elementIds, gt_elementlengths
            file_path = f"{self.npz_path}/{npz_filename}.npz"
            if not os.path.exists(file_path):
                logger.warning("File not exist " + file_path)
                continue
            # assert os.path.exists(file_path) == True
            with np.load(file_path, mmap_mode="r") as data:
                logger.info("load file " + file_path)
                coords, gt_elementIds, gt_elementlengths = (
                    data["coords"],
                    data["elementIds"],
                    data["elementlengths"],
                )
            assert len(eval_ann["pts_semantic_mask"]) == len(gt_elementIds)

            # back to entity lengths
            element_lengths = back_element_lengths(
                np.array(gt_elementIds), np.array(gt_elementlengths)
            )
            element_weights.append(np.round(np.log(1 + element_lengths), 3))

            if (
                self.metric_meta["dataset_name"] == "S3DIS"
                or self.metric_meta["dataset_name"] == "Floorplan"
            ):
                pan_gt = {}
                pan_gt["pts_semantic_mask"] = eval_ann["pts_semantic_mask"].copy()
                pan_gt["pts_instance_mask"] = eval_ann["pts_instance_mask"].copy()
                for stuff_cls in self.stuff_class_inds:
                    pan_gt["pts_instance_mask"][
                        pan_gt["pts_semantic_mask"] == stuff_cls
                    ] = (np.max(pan_gt["pts_instance_mask"]) + 1)

                pan_gt["pts_instance_mask"] = np.unique(
                    pan_gt["pts_instance_mask"], return_inverse=True
                )[1]

                # get entity GT panoptic
                pan_gt["pts_semantic_mask"], pan_gt["pts_instance_mask"] = (
                    back_to_entity_panoptic(
                        pan_gt["pts_semantic_mask"].copy(),
                        pan_gt["pts_instance_mask"].copy(),
                        gt_elementIds,
                    )
                )
                gt_masks_pan.append(pan_gt)
            else:
                gt_masks_pan.append(eval_ann)
            # get entity pred panoptic
            (
                single_pred_results["pts_semantic_mask"][1],
                single_pred_results["pts_instance_mask"][1],
            ) = back_to_entity_panoptic(
                single_pred_results["pts_semantic_mask"][1].copy(),
                single_pred_results["pts_instance_mask"][1].copy(),
                gt_elementIds,
            )
            pred_masks_pan.append(
                {
                    "pts_instance_mask": single_pred_results["pts_instance_mask"][1],
                    "pts_semantic_mask": single_pred_results["pts_semantic_mask"][1],
                }
            )

            # get entity GT semantic
            gt_semantic_mask = back_to_entity_semantic(
                eval_ann["pts_semantic_mask"].copy(), gt_elementIds
            )
            gt_semantic_masks_sem_task.append(gt_semantic_mask)
            # get entity pred semantic
            pred_semantic_mask = back_to_entity_semantic(
                single_pred_results["pts_semantic_mask"][0].copy(), gt_elementIds
            )
            pred_semantic_masks_sem_task.append(pred_semantic_mask)

            # get entity GT instance
            if self.metric_meta["dataset_name"] == "S3DIS":
                gt_semantic_masks_inst_task.append(eval_ann["pts_semantic_mask"])
                gt_instance_masks_inst_task.append(eval_ann["pts_instance_mask"])
            elif self.metric_meta["dataset_name"] == "Floorplan":
                sem_mask = eval_ann["pts_semantic_mask"].copy()
                inst_mask = eval_ann["pts_instance_mask"].copy()
                # back to entity
                sem_mask, inst_mask = back_to_entity_panoptic(
                    sem_mask.copy(),
                    inst_mask.copy(),
                    gt_elementIds,
                )
                # print(sem_mask)
                # print(inst_mask)
                inst_mask[sem_mask >= 30] = -1
                sem_mask[inst_mask == -1] = -1
                gt_semantic_masks_inst_task.append(sem_mask)
                gt_instance_masks_inst_task.append(inst_mask)
            else:
                sem_mask, inst_mask = self.map_inst_markup(
                    eval_ann["pts_semantic_mask"].copy(),
                    eval_ann["pts_instance_mask"].copy(),
                    self.valid_class_ids[num_stuff_cls:],
                    num_stuff_cls,
                )
                gt_semantic_masks_inst_task.append(sem_mask)
                gt_instance_masks_inst_task.append(inst_mask)

            # get entity pred instance
            pred_instance_mask, instance_labels, instance_scores = (
                back_to_entity_instance(
                    single_pred_results["pts_instance_mask"][0].copy(),
                    single_pred_results["instance_labels"].copy(),
                    single_pred_results["instance_scores"].copy(),
                    gt_elementIds,
                )
            )
            pred_instance_masks_inst_task.append(torch.tensor(pred_instance_mask))
            pred_instance_labels.append(
                # TODO 预测有错误？instance_labels中出现了stuff_id
                # 原因：predict时, n_instance_classes == num_classes
                torch.tensor(instance_labels)
            )
            pred_instance_scores.append(torch.tensor(instance_scores))
            output_dir = os.path.join(self.saved_dir, "cloudpoint_output")
            os.makedirs(output_dir, exist_ok=True)
            pts_instance_mask = single_pred_results["pts_instance_mask"][0].copy()
            pts_semantic_mask = single_pred_results["pts_semantic_mask"][0].copy()
            point_colors_mask = (
                np.array([self.SVG_CATEGORIES[id]["color"] for id in pts_semantic_mask])
                / 255.0
            )
            save_filename = os.path.join(output_dir, f"{npz_filename}.png")
            bboxes = []
            labels = []
            for i in range(len(pts_instance_mask)):
                temp_idx = np.where(pts_instance_mask[i] == True)[0]
                temp_semantic_id = np.argmax(np.bincount(pts_semantic_mask[temp_idx]))
                if temp_semantic_id in [30, 31, 32, 33, 34, 35]:
                    continue
                temp_coords = coords[temp_idx]
                temp_bbox_x = temp_coords[:, 0].min()
                temp_bbox_y = temp_coords[:, 1].min()
                width = temp_coords[:, 0].max() - temp_bbox_x
                height = temp_coords[:, 1].max() - temp_bbox_y
                bboxes.append([temp_bbox_x, temp_bbox_y, width, height])
                labels.append(f"ins_{i}")
            self.save_plt_png(coords, point_colors_mask, bboxes, labels, save_filename)

        ret_pan = panoptic_seg_eval(
            gt_masks_pan,
            pred_masks_pan,
            classes,
            thing_classes,
            stuff_classes,
            self.min_num_points,
            self.id_offset,
            label2cat,
            ignore_index,
            logger,
        )

        ret_sem = semantic_seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index,
            logger=logger,
        )

        ret_sem_F1 = semantic_F1_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        if self.metric_meta["dataset_name"] == "S3DIS":
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids,
                class_labels=classes[:-1],
                logger=logger,
            )
        elif self.metric_meta["dataset_name"] == "Floorplan":
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[: -num_stuff_cls + 1],
                class_labels=classes[:-num_stuff_cls],
                options=dict(min_region_sizes=np.array([0]), ap75=True),
                logger=logger,
            )
        else:
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger,
            )

        metrics = dict()
        for ret, keys in zip(
            (ret_sem, ret_inst, ret_pan, ret_sem_F1), self.logger_keys
        ):
            for key in keys:
                metrics[key] = ret[key]
        return metrics

    def add_bbox_and_text(self, ax, bbox, label=None):
        """
        ax : Matplotlib.Axes
        bbox : tuple or list (x, y, width, height)。
        label : str, optional
        """
        x, y, width, height = bbox
        rect = self.mpatches.Rectangle(
            (x, y),
            width,
            height,
            edgecolor="red",
            facecolor="grey",
            linewidth=2,
            alpha=0.3,
        )
        ax.add_patch(rect)
        if label is not None:
            text_x = x + width / 2
            text_y = y + height
            ax.text(
                text_x,
                text_y,
                label,
                ha="center",
                va="bottom",
                color="black",
                fontsize=10,
            )

    def save_plt_png(self, coords, colors, bboxes, labels, save_path):

        fig, ax = self.plt.subplots(figsize=(8, 8))
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=0.1)

        for bbox, label in zip(bboxes, labels):
            self.add_bbox_and_text(ax, bbox, label)

        ax.set_xlim([0, 140])
        ax.set_ylim([0, 140])
        self.plt.savefig(save_path)


@METRICS.register_module()
class UnifiedSegMetric_Save(UnifiedSegMetric_Entity_Sympoint):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    import pickle

    SVG_CATEGORIES = [
        {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "single door"},
        {"color": [157, 34, 101], "isthing": 1, "id": 2, "name": "double door"},
        {"color": [232, 116, 91], "isthing": 1, "id": 3, "name": "sliding door"},
        {"color": [101, 54, 72], "isthing": 1, "id": 4, "name": "folding door"},
        {"color": [172, 107, 133], "isthing": 1, "id": 5, "name": "revolving door"},
        {"color": [142, 76, 101], "isthing": 1, "id": 6, "name": "rolling door"},
        {"color": [96, 78, 245], "isthing": 1, "id": 7, "name": "window"},
        {"color": [26, 2, 219], "isthing": 1, "id": 8, "name": "bay window"},
        {"color": [63, 140, 221], "isthing": 1, "id": 9, "name": "blind window"},
        {"color": [233, 59, 217], "isthing": 1, "id": 10, "name": "opening symbol"},
        {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "sofa"},
        {"color": [94, 150, 113], "isthing": 1, "id": 12, "name": "bed"},
        {"color": [66, 107, 81], "isthing": 1, "id": 13, "name": "chair"},
        {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "table"},
        {"color": [94, 150, 83], "isthing": 1, "id": 15, "name": "TV cabinet"},
        {"color": [66, 107, 59], "isthing": 1, "id": 16, "name": "Wardrobe"},
        {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "cabinet"},
        {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "gas stove"},
        {"color": [113, 151, 82], "isthing": 1, "id": 19, "name": "sink"},
        {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "refrigerator"},
        {"color": [81, 107, 58], "isthing": 1, "id": 21, "name": "airconditioner"},
        {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "bath"},
        {"color": [141, 152, 83], "isthing": 1, "id": 23, "name": "bath tub"},
        {"color": [80, 72, 147], "isthing": 1, "id": 24, "name": "washing machine"},
        {"color": [100, 108, 59], "isthing": 1, "id": 25, "name": "squat toilet"},
        {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "urinal"},
        {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "toilet"},
        {"color": [247, 206, 75], "isthing": 1, "id": 28, "name": "stairs"},
        {"color": [237, 112, 45], "isthing": 1, "id": 29, "name": "elevator"},
        {"color": [233, 59, 46], "isthing": 1, "id": 30, "name": "escalator"},
        {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "row chairs"},
        {"color": [102, 67, 62], "isthing": 0, "id": 32, "name": "parking spot"},
        {"color": [167, 92, 32], "isthing": 0, "id": 33, "name": "wall"},
        {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "curtain wall"},
        {"color": [64, 52, 105], "isthing": 0, "id": 35, "name": "railing"},
        {"color": [0, 0, 0], "isthing": 0, "id": 36, "name": "bg"},
    ]

    def __init__(
        self,
        saved_dir=None,
        ifeval=True,
        ifsave=True,
        **kwargs,
    ):
        # * 先指定好保存路径
        self.ifsave = ifsave
        self.saved_dir = saved_dir
        if ifsave:
            assert self.saved_dir
            os.makedirs(saved_dir, exist_ok=True)

        self.ifeval = ifeval
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        assert self.metric_meta["dataset_name"] == "Floorplan"
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]
        label2cat = self.metric_meta["label2cat"]
        ignore_index = self.metric_meta["ignore_index"]
        if self.ignore_bg:
            ignore_index = ignore_index + [35]
        classes = self.metric_meta["classes"]
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        gt_semantic_masks_inst_task_sym = []
        gt_instance_masks_inst_task_sym = []

        gt_semantic_bbox_labels_inst_task_sym = []
        gt_instance_bboxs_inst_task_sym = []
        pred_instance_masks_inst_task_sym = []
        pred_instance_labels_sym = []
        pred_instance_scores_sym = []

        pred_instance_bboxs_inst_task_sym = []

        # pred_instances_list = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []

        element_weights = []

        for eval_ann, single_pred_results in results:
            lidar_paths = single_pred_results["lidar_paths"]
            assert len(lidar_paths) == 1
            # npz_filename = lidar_paths[0][-13:].split(".")[0] # TODO
            if "train" in lidar_paths[0]:
                npz_filename = (
                    lidar_paths[0]
                    .split("/")[-1]
                    .split(".")[0]
                    .replace("Area_1_train", "")
                )
            elif "test" in lidar_paths[0]:
                npz_filename = (
                    lidar_paths[0]
                    .split("/")[-1]
                    .split(".")[0]
                    .replace("Area_2_test", "")
                )
            elif "val" in lidar_paths[0]:
                npz_filename = (
                    lidar_paths[0]
                    .split("/")[-1]
                    .split(".")[0]
                    .replace("Area_3_val", "")
                )
            # 从gt文件中读取gt_elementIds, gt_elementlengths
            file_path = f"{self.npz_path}/{npz_filename}.npz"
            if not os.path.exists(file_path):
                logger.warning("File not exist " + file_path)
                continue
            # assert os.path.exists(file_path) == True
            with np.load(file_path, mmap_mode="r") as data:
                logger.info("load file " + file_path)
                gt_coords, gt_elementIds, gt_elementlengths = (
                    data["coords"],
                    data["elementIds"],
                    data["elementlengths"],
                )
            assert len(eval_ann["pts_semantic_mask"]) == len(gt_elementIds)

            # back to entity lengths
            element_lengths = back_element_lengths(
                np.array(gt_elementIds), np.array(gt_elementlengths)
            )
            ele_weight = np.round(np.log(1 + element_lengths), 3)
            element_weights.append(ele_weight)

            if self.ifeval:
                # oneformer pan gt
                # pan_gt = {}
                # pan_gt["pts_semantic_mask"] = eval_ann["pts_semantic_mask"].copy()
                # pan_gt["pts_instance_mask"] = eval_ann["pts_instance_mask"].copy()
                # for stuff_cls in self.stuff_class_inds:
                #     pan_gt["pts_instance_mask"][
                #         pan_gt["pts_semantic_mask"] == stuff_cls
                #     ] = (np.max(pan_gt["pts_instance_mask"]) + 1)
                # pan_gt["pts_instance_mask"] = np.unique(
                #     pan_gt["pts_instance_mask"], return_inverse=True
                # )[1]

                # # oneformer get entity GT panoptic
                # pan_gt["pts_semantic_mask"], pan_gt["pts_instance_mask"] = (
                #     back_to_entity_panoptic(
                #         pan_gt["pts_semantic_mask"].copy(),
                #         pan_gt["pts_instance_mask"].copy(),
                #         gt_elementIds,
                #     )
                # )
                # # oneformer gt panoptic
                # gt_masks_pan.append(pan_gt)

                # sympoint get entity GT semantic
                gt_semantic_mask = back_to_entity_semantic(
                    eval_ann["pts_semantic_mask"].copy(), gt_elementIds
                )

                gt_semantic_masks_sem_task.append(gt_semantic_mask)
                # sympoint get entity GT instance
                sem_mask = eval_ann["pts_semantic_mask"].copy()
                inst_mask = eval_ann["pts_instance_mask"].copy()
                # sympoint back to entity
                sem_mask, inst_mask = back_to_entity_panoptic(
                    sem_mask.copy(),
                    inst_mask.copy(),
                    gt_elementIds,
                )
                gt_bbox_labels, gt_bboxs = back_to_ins_bbox_gt(
                    sem_mask.copy(),
                    inst_mask.copy(),
                    gt_elementIds,
                    gt_coords,
                )
                gt_semantic_masks_inst_task_sym.append(sem_mask.copy())
                gt_instance_masks_inst_task_sym.append(inst_mask.copy())
                gt_semantic_bbox_labels_inst_task_sym.append(gt_bbox_labels.copy())
                gt_instance_bboxs_inst_task_sym.append(gt_bboxs.copy())

                # # oneformer instance gt
                # inst_mask[sem_mask >= 30] = -1
                # sem_mask[inst_mask == -1] = -1
                # gt_semantic_masks_inst_task.append(sem_mask)
                # gt_instance_masks_inst_task.append(inst_mask)
            # # oneformer get entity pred panoptic
            # (
            #     single_pred_results["pts_semantic_mask"][1],
            #     single_pred_results["pts_instance_mask"][1],
            # ) = back_to_entity_panoptic(
            #     single_pred_results["pts_semantic_mask"][1].copy(),
            #     single_pred_results["pts_instance_mask"][1].copy(),
            #     gt_elementIds,
            # )
            # pred_masks_pan.append(
            #     {
            #         "pts_instance_mask": single_pred_results["pts_instance_mask"][1],
            #         "pts_semantic_mask": single_pred_results["pts_semantic_mask"][1],
            #     }
            # )
            # oneformer& sympoint get entity pred semantic
            pred_semantic_mask = back_to_entity_semantic(
                single_pred_results["pts_semantic_mask"][0].copy(), gt_elementIds
            )
            if self.ifsave:
                self.save_sem_pred(self.saved_dir, npz_filename, pred_semantic_mask)    
            pred_semantic_masks_sem_task.append(pred_semantic_mask)

            # # onefomer get entity pred instance
            # pred_instance_mask, instance_labels, instance_scores = (
            #     back_to_entity_instance(
            #         single_pred_results["pts_instance_mask"][0].copy(),
            #         single_pred_results["instance_labels"].copy(),
            #         single_pred_results["instance_scores"].copy(),
            #         gt_elementIds,
            #     )
            # )
            # pred_instance_masks_inst_task.append(torch.tensor(pred_instance_mask))
            # pred_instance_labels.append(
            #     # TODO 预测有错误？instance_labels中出现了stuff_id
            #     # 原因：predict时, n_instance_classes == num_classes
            #     torch.tensor(instance_labels)
            # )
            # pred_instance_scores.append(torch.tensor(instance_scores))

            # sympoint get entity pred for instance
            pred_instance_mask_sym = back_to_entity_instance_sym(
                single_pred_results["pts_instance_mask"][0].copy(),
                gt_elementIds,
            )
            pred_instance_bboxs_sym = back_to_instance_bbox_pred_sym(
                pred_instance_mask_sym,
                gt_elementIds,
                gt_coords,
            )
            pred_instance_masks_inst_task_sym.append(
                torch.tensor(pred_instance_mask_sym)
            )
            pred_instance_bboxs_inst_task_sym.append(pred_instance_bboxs_sym)
            pred_instance_labels_sym.append(
                torch.tensor(single_pred_results["instance_labels"])
            )
            pred_instance_scores_sym.append(
                torch.tensor(single_pred_results["instance_scores"])
            )
            bboxes = []
            labels = []
            scores = []
            pts_instance_mask = single_pred_results["pts_instance_mask"][0].copy()
            pts_semantic_mask = single_pred_results["pts_semantic_mask"][0].copy()
            pts_score = single_pred_results["instance_scores"].copy()
            if self.ifsave:
                for i in range(len(pts_instance_mask)):
                    temp_idx = np.where(pts_instance_mask[i] == True)[0]
                    temp_semantic_id = np.argmax(np.bincount(pts_semantic_mask[temp_idx]))
                    if temp_semantic_id in [30, 31, 32, 33, 34, 35]:
                        continue
                    temp_coords = gt_coords[temp_idx]
                    temp_bbox_x = temp_coords[:, 0].min()
                    temp_bbox_y = temp_coords[:, 1].min()
                    width = temp_coords[:, 0].max() - temp_bbox_x
                    height = temp_coords[:, 1].max() - temp_bbox_y
                    bboxes.append([temp_bbox_x, temp_bbox_y, width, height])
                    labels.append(temp_semantic_id)
                    scores.append(pts_score[i])
                save_data = {"bboxes": bboxes, "labels": labels, "scores": scores}
                save_filename = os.path.join(self.saved_dir, f"{npz_filename}.pkl")
                with open(save_filename, "wb") as f:
                    self.pickle.dump(save_data, f)

        if self.ifeval:
            logger.warning(
                "================================>Sympoint Metrics<================================"
            )

            sym_ret_sem = sympoint_semantic_eval(
                gt_semantic_masks_sem_task,
                pred_semantic_masks_sem_task,
                classes,
                ignore_index,
                logger=logger,
            )

            # sym_ret_sem_F1
            ret_sem_F1 = sympoint_semantic_F1_eval(
                gt_semantic_masks_sem_task,
                pred_semantic_masks_sem_task,
                element_weights,
                classes,
                ignore_index,
                logger=logger,
            )

            sym_ret_ins = sympoint_instance_eval(
                gt_semantic_masks_inst_task_sym,
                gt_instance_masks_inst_task_sym,
                pred_instance_masks_inst_task_sym,
                gt_semantic_bbox_labels_inst_task_sym,
                gt_instance_bboxs_inst_task_sym,
                pred_instance_bboxs_inst_task_sym,
                pred_instance_labels_sym,
                pred_instance_scores_sym,
                element_weights,
                classes,
                ignore_index,
                logger=logger,
            )

            # if len(pred_instances_list) > 0:
            #     sym_ret_ins_inf = sympoint_instance_inference_eval(
            #         gt_semantic_masks_inst_task_sym,
            #         gt_instance_masks_inst_task_sym,
            #         pred_instances_list,
            #         element_weights,
            #         classes,
            #         ignore_index,
            #         logger=logger,
            #     )

            metrics = dict()
            for ret, keys in zip(
                (ret_sem_F1, sym_ret_ins, sym_ret_sem), self.logger_keys
            ):
                for key in keys:
                    metrics[key] = ret[key]
            return metrics
        return dict()

    def save_sem_pred(self, save_dir, file_name, pred_semantic_mask):
        filename = os.path.join(save_dir, file_name)
        np.save(filename, pred_semantic_mask)
        np.save(filename, pred_semantic_mask)


@METRICS.register_module()
class UnifiedSegMetric_Entity_Sympoint_delete(UnifiedSegMetric_Entity_Sympoint):
    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        assert self.metric_meta["dataset_name"] == "Floorplan"
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]
        label2cat = self.metric_meta["label2cat"]
        ignore_index = self.metric_meta["ignore_index"]
        if self.ignore_bg:
            ignore_index = ignore_index + [35]
        classes = self.metric_meta["classes"]
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        gt_semantic_masks_inst_task_sym = []
        gt_instance_masks_inst_task_sym = []

        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []
        pred_instance_masks_inst_task_sym = []
        pred_instance_labels_sym = []
        pred_instance_scores_sym = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        element_weights = []

        for eval_ann, single_pred_results in results:
            lidar_paths = single_pred_results["lidar_paths"]
            assert len(lidar_paths) == 1
            npz_filename = lidar_paths[0][-13:].split(".")[0]
            # 从gt文件中读取gt_elementIds, gt_elementlengths
            file_path = f"{self.npz_path}/{npz_filename}.npz"
            if not os.path.exists(file_path):
                logger.warning("File not exist " + file_path)
                continue
            # assert os.path.exists(file_path) == True
            with np.load(file_path, mmap_mode="r") as data:
                logger.info("load file " + file_path)
                gt_elementIds, gt_elementlengths = (
                    data["elementIds"],
                    data["elementlengths"],
                )
            assert len(eval_ann["pts_semantic_mask"]) == len(gt_elementIds)

            # back to entity lengths
            element_lengths = back_element_lengths(
                np.array(gt_elementIds), np.array(gt_elementlengths)
            )
            ele_weight = np.round(np.log(1 + element_lengths), 3)
            element_weights.append(ele_weight)

            # oneformer pan gt
            pan_gt = {}
            pan_gt["pts_semantic_mask"] = eval_ann["pts_semantic_mask"].copy()
            pan_gt["pts_instance_mask"] = eval_ann["pts_instance_mask"].copy()
            for stuff_cls in self.stuff_class_inds:
                pan_gt["pts_instance_mask"][
                    pan_gt["pts_semantic_mask"] == stuff_cls
                ] = (np.max(pan_gt["pts_instance_mask"]) + 1)

            pan_gt["pts_instance_mask"] = np.unique(
                pan_gt["pts_instance_mask"], return_inverse=True
            )[1]

            # get entity GT panoptic
            pan_gt["pts_semantic_mask"], pan_gt["pts_instance_mask"] = (
                back_to_entity_panoptic(
                    pan_gt["pts_semantic_mask"].copy(),
                    pan_gt["pts_instance_mask"].copy(),
                    gt_elementIds,
                )
            )
            gt_masks_pan.append(pan_gt)

            # get entity pred panoptic
            (
                single_pred_results["pts_semantic_mask"][1],
                single_pred_results["pts_instance_mask"][1],
            ) = back_to_entity_panoptic(
                single_pred_results["pts_semantic_mask"][1].copy(),
                single_pred_results["pts_instance_mask"][1].copy(),
                gt_elementIds,
            )
            pred_masks_pan.append(
                {
                    "pts_instance_mask": single_pred_results["pts_instance_mask"][1],
                    "pts_semantic_mask": single_pred_results["pts_semantic_mask"][1],
                }
            )

            # get entity GT semantic
            gt_semantic_mask = back_to_entity_semantic(
                eval_ann["pts_semantic_mask"].copy(), gt_elementIds
            )
            gt_semantic_masks_sem_task.append(gt_semantic_mask)
            # get entity pred semantic
            pred_semantic_mask = back_to_entity_semantic(
                single_pred_results["pts_semantic_mask"][0].copy(), gt_elementIds
            )
            pred_semantic_masks_sem_task.append(pred_semantic_mask)

            # get entity GT instance
            sem_mask = eval_ann["pts_semantic_mask"].copy()
            inst_mask = eval_ann["pts_instance_mask"].copy()
            # back to entity
            sem_mask, inst_mask = back_to_entity_panoptic(
                sem_mask.copy(),
                inst_mask.copy(),
                gt_elementIds,
            )

            gt_semantic_masks_inst_task_sym.append(sem_mask.copy())
            gt_instance_masks_inst_task_sym.append(inst_mask.copy())
            # print(sem_mask)
            # print(inst_mask)
            inst_mask[sem_mask >= 30] = -1
            sem_mask[inst_mask == -1] = -1
            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)

            # get entity pred instance
            pred_instance_mask, instance_labels, instance_scores = (
                back_to_entity_instance(
                    single_pred_results["pts_instance_mask"][0].copy(),
                    single_pred_results["instance_labels"].copy(),
                    single_pred_results["instance_scores"].copy(),
                    gt_elementIds,
                )
            )
            pred_instance_masks_inst_task.append(torch.tensor(pred_instance_mask))
            pred_instance_labels.append(
                # TODO 预测有错误？instance_labels中出现了stuff_id
                # 原因：predict时, n_instance_classes == num_classes
                torch.tensor(instance_labels)
            )
            pred_instance_scores.append(torch.tensor(instance_scores))

            # get entity pred for sympoint instance
            pred_instance_mask_sym = back_to_entity_instance_sym(
                single_pred_results["pts_instance_mask"][0].copy(),
                gt_elementIds,
            )
            pred_instance_masks_inst_task_sym.append(
                torch.tensor(pred_instance_mask_sym)
            )
            pred_instance_labels_sym.append(
                torch.tensor(single_pred_results["instance_labels"])
            )
            pred_instance_scores_sym.append(
                torch.tensor(single_pred_results["instance_scores"])
            )

        ret_pan = panoptic_seg_eval(
            gt_masks_pan,
            pred_masks_pan,
            classes,
            thing_classes,
            stuff_classes,
            self.min_num_points,
            self.id_offset,
            label2cat,
            ignore_index,
            logger,
        )

        ret_sem = semantic_seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index,
            logger=logger,
        )

        # :-1 for unlabeled
        ret_inst = instance_seg_eval(
            gt_semantic_masks_inst_task,
            gt_instance_masks_inst_task,
            pred_instance_masks_inst_task,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids[: -num_stuff_cls + 1],
            class_labels=classes[:-num_stuff_cls],
            options=dict(min_region_sizes=np.array([0]), ap75=True),
            logger=logger,
        )

        logger.warning(
            "================================>Sympoint Metrics<================================"
        )

        sym_ret_sem = sympoint_semantic_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            classes,
            ignore_index,
            logger=logger,
        )

        # sym_ret_sem_F1
        ret_sem_F1 = sympoint_semantic_F1_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        sym_ret_ins = sympoint_instance_eval_delete(
            gt_semantic_masks_inst_task_sym,
            gt_instance_masks_inst_task_sym,
            pred_instance_masks_inst_task_sym,
            pred_instance_labels_sym,
            pred_instance_scores_sym,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        metrics = dict()
        for ret, keys in zip((ret_sem_F1, sym_ret_ins, sym_ret_sem), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics


@METRICS.register_module()
class UnifiedSegMetric_Entity_Sympoint_delete_1(UnifiedSegMetric_Entity_Sympoint):
    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        assert self.metric_meta["dataset_name"] == "Floorplan"
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]
        label2cat = self.metric_meta["label2cat"]
        ignore_index = self.metric_meta["ignore_index"]
        if self.ignore_bg:
            ignore_index = ignore_index + [35]
        classes = self.metric_meta["classes"]
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        gt_semantic_masks_inst_task_sym = []
        gt_instance_masks_inst_task_sym = []

        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []
        pred_instance_masks_inst_task_sym = []
        pred_instance_labels_sym = []
        pred_instance_scores_sym = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        element_weights = []

        for eval_ann, single_pred_results in results:
            lidar_paths = single_pred_results["lidar_paths"]
            assert len(lidar_paths) == 1
            npz_filename = lidar_paths[0][-13:].split(".")[0]
            # 从gt文件中读取gt_elementIds, gt_elementlengths
            file_path = f"{self.npz_path}/{npz_filename}.npz"
            if not os.path.exists(file_path):
                logger.warning("File not exist " + file_path)
                continue
            # assert os.path.exists(file_path) == True
            with np.load(file_path, mmap_mode="r") as data:
                logger.info("load file " + file_path)
                gt_elementIds, gt_elementlengths = (
                    data["elementIds"],
                    data["elementlengths"],
                )
            assert len(eval_ann["pts_semantic_mask"]) == len(gt_elementIds)

            # back to entity lengths
            element_lengths = back_element_lengths(
                np.array(gt_elementIds), np.array(gt_elementlengths)
            )
            ele_weight = np.round(np.log(1 + element_lengths), 3)
            element_weights.append(ele_weight)

            # oneformer pan gt
            pan_gt = {}
            pan_gt["pts_semantic_mask"] = eval_ann["pts_semantic_mask"].copy()
            pan_gt["pts_instance_mask"] = eval_ann["pts_instance_mask"].copy()
            for stuff_cls in self.stuff_class_inds:
                pan_gt["pts_instance_mask"][
                    pan_gt["pts_semantic_mask"] == stuff_cls
                ] = (np.max(pan_gt["pts_instance_mask"]) + 1)

            pan_gt["pts_instance_mask"] = np.unique(
                pan_gt["pts_instance_mask"], return_inverse=True
            )[1]

            # get entity GT panoptic
            pan_gt["pts_semantic_mask"], pan_gt["pts_instance_mask"] = (
                back_to_entity_panoptic(
                    pan_gt["pts_semantic_mask"].copy(),
                    pan_gt["pts_instance_mask"].copy(),
                    gt_elementIds,
                )
            )
            gt_masks_pan.append(pan_gt)

            # get entity pred panoptic
            (
                single_pred_results["pts_semantic_mask"][1],
                single_pred_results["pts_instance_mask"][1],
            ) = back_to_entity_panoptic(
                single_pred_results["pts_semantic_mask"][1].copy(),
                single_pred_results["pts_instance_mask"][1].copy(),
                gt_elementIds,
            )
            pred_masks_pan.append(
                {
                    "pts_instance_mask": single_pred_results["pts_instance_mask"][1],
                    "pts_semantic_mask": single_pred_results["pts_semantic_mask"][1],
                }
            )

            # get entity GT semantic
            gt_semantic_mask = back_to_entity_semantic(
                eval_ann["pts_semantic_mask"].copy(), gt_elementIds
            )
            gt_semantic_masks_sem_task.append(gt_semantic_mask)
            # get entity pred semantic
            pred_semantic_mask = back_to_entity_semantic(
                single_pred_results["pts_semantic_mask"][0].copy(), gt_elementIds
            )

            pred_semantic_masks_sem_task.append(pred_semantic_mask)

            # get entity GT instance
            sem_mask = eval_ann["pts_semantic_mask"].copy()
            inst_mask = eval_ann["pts_instance_mask"].copy()
            # back to entity
            sem_mask, inst_mask = back_to_entity_panoptic(
                sem_mask.copy(),
                inst_mask.copy(),
                gt_elementIds,
            )

            gt_semantic_masks_inst_task_sym.append(sem_mask.copy())
            gt_instance_masks_inst_task_sym.append(inst_mask.copy())
            # print(sem_mask)
            # print(inst_mask)
            inst_mask[sem_mask >= 30] = -1
            sem_mask[inst_mask == -1] = -1
            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)

            # get entity pred instance
            pred_instance_mask, instance_labels, instance_scores = (
                back_to_entity_instance(
                    single_pred_results["pts_instance_mask"][0].copy(),
                    single_pred_results["instance_labels"].copy(),
                    single_pred_results["instance_scores"].copy(),
                    gt_elementIds,
                )
            )
            pred_instance_masks_inst_task.append(torch.tensor(pred_instance_mask))
            pred_instance_labels.append(
                # TODO 预测有错误？instance_labels中出现了stuff_id
                # 原因：predict时, n_instance_classes == num_classes
                torch.tensor(instance_labels)
            )
            pred_instance_scores.append(torch.tensor(instance_scores))

            # get entity pred for sympoint instance
            pred_instance_mask_sym = back_to_entity_instance_sym(
                single_pred_results["pts_instance_mask"][0].copy(),
                gt_elementIds,
            )
            pred_instance_masks_inst_task_sym.append(
                torch.tensor(pred_instance_mask_sym)
            )
            pred_instance_labels_sym.append(
                torch.tensor(single_pred_results["instance_labels"])
            )
            pred_instance_scores_sym.append(
                torch.tensor(single_pred_results["instance_scores"])
            )

        ret_pan = panoptic_seg_eval(
            gt_masks_pan,
            pred_masks_pan,
            classes,
            thing_classes,
            stuff_classes,
            self.min_num_points,
            self.id_offset,
            label2cat,
            ignore_index,
            logger,
        )

        ret_sem = semantic_seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index,
            logger=logger,
        )

        # :-1 for unlabeled
        ret_inst = instance_seg_eval(
            gt_semantic_masks_inst_task,
            gt_instance_masks_inst_task,
            pred_instance_masks_inst_task,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids[: -num_stuff_cls + 1],
            class_labels=classes[:-num_stuff_cls],
            options=dict(min_region_sizes=np.array([0]), ap75=True),
            logger=logger,
        )

        logger.warning(
            "================================>Sympoint Metrics<================================"
        )

        sym_ret_sem = sympoint_semantic_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            classes,
            ignore_index,
            logger=logger,
        )

        # sym_ret_sem_F1
        ret_sem_F1 = sympoint_semantic_F1_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        sym_ret_ins = sympoint_instance_eval_delete_1(
            gt_semantic_masks_inst_task_sym,
            gt_instance_masks_inst_task_sym,
            pred_instance_masks_inst_task_sym,
            pred_instance_labels_sym,
            pred_instance_scores_sym,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        metrics = dict()
        for ret, keys in zip((ret_sem_F1, sym_ret_ins, sym_ret_sem), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics

    def save_sem_pred(self, save_dir, file_name, pred_semantic_mask):
        filename = os.path.join(save_dir, file_name)
        np.save(filename, pred_semantic_mask)


@METRICS.register_module()
class UnifiedSegMetric_Entity_Sympoint_new(UnifiedSegMetric_Entity_Sympoint):
    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        assert self.metric_meta["dataset_name"] == "Floorplan"
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]
        label2cat = self.metric_meta["label2cat"]
        ignore_index = self.metric_meta["ignore_index"]
        if self.ignore_bg:
            ignore_index = ignore_index + [35]
        classes = self.metric_meta["classes"]
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        gt_semantic_masks_inst_task_sym = []
        gt_instance_masks_inst_task_sym = []

        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []
        pred_instance_masks_inst_task_sym = []
        pred_instance_labels_sym = []
        pred_instance_scores_sym = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        element_weights = []

        for eval_ann, single_pred_results in results:
            lidar_paths = single_pred_results["lidar_paths"]
            assert len(lidar_paths) == 1
            npz_filename = lidar_paths[0][-13:].split(".")[0]
            # 从gt文件中读取gt_elementIds, gt_elementlengths
            file_path = f"{self.npz_path}/{npz_filename}.npz"
            if not os.path.exists(file_path):
                logger.warning("File not exist " + file_path)
                continue
            # assert os.path.exists(file_path) == True
            with np.load(file_path, mmap_mode="r") as data:
                logger.info("load file " + file_path)
                gt_elementIds, gt_elementlengths = (
                    data["elementIds"],
                    data["elementlengths"],
                )
            assert len(eval_ann["pts_semantic_mask"]) == len(gt_elementIds)

            # back to entity lengths
            element_lengths = back_element_lengths(
                np.array(gt_elementIds), np.array(gt_elementlengths)
            )
            ele_weight = np.round(np.log(1 + element_lengths), 3)
            element_weights.append(ele_weight)

            # oneformer pan gt
            pan_gt = {}
            pan_gt["pts_semantic_mask"] = eval_ann["pts_semantic_mask"].copy()
            pan_gt["pts_instance_mask"] = eval_ann["pts_instance_mask"].copy()
            for stuff_cls in self.stuff_class_inds:
                pan_gt["pts_instance_mask"][
                    pan_gt["pts_semantic_mask"] == stuff_cls
                ] = (np.max(pan_gt["pts_instance_mask"]) + 1)

            pan_gt["pts_instance_mask"] = np.unique(
                pan_gt["pts_instance_mask"], return_inverse=True
            )[1]

            # get entity GT panoptic
            pan_gt["pts_semantic_mask"], pan_gt["pts_instance_mask"] = (
                back_to_entity_panoptic(
                    pan_gt["pts_semantic_mask"].copy(),
                    pan_gt["pts_instance_mask"].copy(),
                    gt_elementIds,
                )
            )
            gt_masks_pan.append(pan_gt)

            # get entity pred panoptic
            (
                single_pred_results["pts_semantic_mask"][1],
                single_pred_results["pts_instance_mask"][1],
            ) = back_to_entity_panoptic(
                single_pred_results["pts_semantic_mask"][1].copy(),
                single_pred_results["pts_instance_mask"][1].copy(),
                gt_elementIds,
            )
            pred_masks_pan.append(
                {
                    "pts_instance_mask": single_pred_results["pts_instance_mask"][1],
                    "pts_semantic_mask": single_pred_results["pts_semantic_mask"][1],
                }
            )

            # get entity GT semantic
            gt_semantic_mask = back_to_entity_semantic(
                eval_ann["pts_semantic_mask"].copy(), gt_elementIds
            )
            gt_semantic_masks_sem_task.append(gt_semantic_mask)
            # get entity pred semantic
            pred_semantic_mask = back_to_entity_semantic(
                single_pred_results["pts_semantic_mask"][0].copy(), gt_elementIds
            )
            pred_semantic_masks_sem_task.append(pred_semantic_mask)

            # get entity GT instance
            sem_mask = eval_ann["pts_semantic_mask"].copy()
            inst_mask = eval_ann["pts_instance_mask"].copy()
            # back to entity
            sem_mask, inst_mask = back_to_entity_panoptic(
                sem_mask.copy(),
                inst_mask.copy(),
                gt_elementIds,
            )

            gt_semantic_masks_inst_task_sym.append(sem_mask.copy())
            gt_instance_masks_inst_task_sym.append(inst_mask.copy())
            # print(sem_mask)
            # print(inst_mask)
            inst_mask[sem_mask >= 30] = -1
            sem_mask[inst_mask == -1] = -1
            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)

            # get entity pred instance
            pred_instance_mask, instance_labels, instance_scores = (
                back_to_entity_instance(
                    single_pred_results["pts_instance_mask"][0].copy(),
                    single_pred_results["instance_labels"].copy(),
                    single_pred_results["instance_scores"].copy(),
                    gt_elementIds,
                )
            )
            pred_instance_masks_inst_task.append(torch.tensor(pred_instance_mask))
            pred_instance_labels.append(
                # TODO 预测有错误？instance_labels中出现了stuff_id
                # 原因：predict时, n_instance_classes == num_classes
                torch.tensor(instance_labels)
            )
            pred_instance_scores.append(torch.tensor(instance_scores))

            # get entity pred for sympoint instance
            pred_instance_mask_sym = back_to_entity_instance_sym(
                single_pred_results["pts_instance_mask"][0].copy(),
                gt_elementIds,
            )
            pred_instance_masks_inst_task_sym.append(
                torch.tensor(pred_instance_mask_sym)
            )
            pred_instance_labels_sym.append(
                torch.tensor(single_pred_results["instance_labels"])
            )
            pred_instance_scores_sym.append(
                torch.tensor(single_pred_results["instance_scores"])
            )

        ret_pan = panoptic_seg_eval(
            gt_masks_pan,
            pred_masks_pan,
            classes,
            thing_classes,
            stuff_classes,
            self.min_num_points,
            self.id_offset,
            label2cat,
            ignore_index,
            logger,
        )

        ret_sem = semantic_seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index,
            logger=logger,
        )

        # :-1 for unlabeled
        ret_inst = instance_seg_eval(
            gt_semantic_masks_inst_task,
            gt_instance_masks_inst_task,
            pred_instance_masks_inst_task,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids[: -num_stuff_cls + 1],
            class_labels=classes[:-num_stuff_cls],
            options=dict(min_region_sizes=np.array([0]), ap75=True),
            logger=logger,
        )

        logger.warning(
            "================================>Sympoint Metrics<================================"
        )

        sym_ret_sem = sympoint_semantic_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            classes,
            ignore_index,
            logger=logger,
        )

        # sym_ret_sem_F1
        ret_sem_F1 = sympoint_semantic_F1_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        sym_ret_ins = sympoint_instance_eval_new(
            gt_semantic_masks_inst_task_sym,
            gt_instance_masks_inst_task_sym,
            pred_instance_masks_inst_task_sym,
            pred_instance_labels_sym,
            pred_instance_scores_sym,
            element_weights,
            classes,
            ignore_index,
            logger=logger,
        )

        metrics = dict()
        for ret, keys in zip((ret_sem_F1, sym_ret_ins, sym_ret_sem), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics

    def save_sem_pred(self, save_dir, file_name, pred_semantic_mask):
        filename = os.path.join(save_dir, file_name)
        np.save(filename, pred_semantic_mask)


@METRICS.register_module()
class UnifiedSegMetric_Save_Primitive(UnifiedSegMetric_Save):
    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        assert self.metric_meta["dataset_name"] == "Floorplan"
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta["seg_valid_class_ids"]
        ignore_index = self.metric_meta["ignore_index"]
        if self.ignore_bg:
            ignore_index = ignore_index + [35]
        classes = self.metric_meta["classes"]

        gt_semantic_masks_inst_task_sym = []
        gt_instance_masks_inst_task_sym = []

        gt_semantic_bbox_labels_inst_task_sym = []
        gt_instance_bboxs_inst_task_sym = []
        pred_instance_masks_inst_task_sym = []
        pred_instance_labels_sym = []
        pred_instance_scores_sym = []

        pred_instance_bboxs_inst_task_sym = []
        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []
        element_weights = []

        for eval_ann, single_pred_results in tqdm(results, desc="Processing"):
            lidar_paths = single_pred_results["lidar_paths"]
            assert len(lidar_paths) == 1
            if "testls" in lidar_paths[0]:
                npz_filename = (
                    lidar_paths[0]
                    .split("/")[-1]
                    .split(".")[0]
                    .replace("Area_2_testls", "")
                )
            elif "train" in lidar_paths[0]:
                npz_filename = (
                    lidar_paths[0]
                    .split("/")[-1]
                    .split(".")[0]
                    .replace("Area_1_train", "")
                )
            elif "test" in lidar_paths[0]:
                npz_filename = (
                    lidar_paths[0]
                    .split("/")[-1]
                    .split(".")[0]
                    .replace("Area_2_test", "")
                )
            elif "val" in lidar_paths[0]:
                npz_filename = (
                    lidar_paths[0]
                    .split("/")[-1]
                    .split(".")[0]
                    .replace("Area_3_val", "")
                )
            # 从gt文件中读取gt_elementIds, gt_elementlengths
            file_path = f"{self.npz_path}/{npz_filename}.npz"
            if not os.path.exists(file_path):
                logger.warning("File not exist " + file_path)
                continue
            # assert os.path.exists(file_path) == True
            with np.load(file_path, mmap_mode="r") as data:
                logger.info("load file " + file_path)
                gt_coords, gt_elementIds, gt_elementlengths, element_bboxes = (
                    data["coords"],
                    data["elementIds"],
                    data["elementlengths"],
                    data["bboxes"],
                )
            # assert len(eval_ann["pts_semantic_mask"]) == len(gt_elementIds)

            if self.ifeval:
                # back to entity lengths
                element_lengths = back_element_lengths(
                    np.array(gt_elementIds), np.array(gt_elementlengths)
                )
                ele_weight = np.round(np.log(1 + element_lengths), 3)
                element_weights.append(ele_weight)

                # sympoint get entity GT semantic
                gt_semantic_mask = back_to_entity_semantic(
                    eval_ann["pts_semantic_mask"].copy(), gt_elementIds
                )

                gt_semantic_masks_sem_task.append(gt_semantic_mask)
                # sympoint get entity GT instance
                sem_mask = eval_ann["pts_semantic_mask"].copy()
                inst_mask = eval_ann["pts_instance_mask"].copy()
                # sympoint back to entity
                sem_mask, inst_mask = back_to_entity_panoptic(
                    sem_mask.copy(),
                    inst_mask.copy(),
                    gt_elementIds,
                )
                gt_bbox_labels, gt_bboxs = back_to_ins_bbox_gt(
                    sem_mask.copy(),
                    inst_mask.copy(),
                    gt_elementIds,
                    gt_coords,
                )
                gt_semantic_masks_inst_task_sym.append(sem_mask.copy())
                gt_instance_masks_inst_task_sym.append(inst_mask.copy())
                gt_semantic_bbox_labels_inst_task_sym.append(gt_bbox_labels.copy())
                gt_instance_bboxs_inst_task_sym.append(gt_bboxs.copy())

            # oneformer& sympoint get entity pred semantic

            # pred_semantic_mask = back_to_entity_semantic(
            #     single_pred_results["pts_semantic_mask"][0].copy(), gt_elementIds
            # )
            pred_semantic_mask = single_pred_results["pts_semantic_mask"][0].copy()
            if self.ifsave:
                self.save_sem_pred(self.saved_dir, npz_filename, pred_semantic_mask)
            pred_semantic_masks_sem_task.append(pred_semantic_mask)

            # sympoint get entity pred for instance
            # pred_instance_mask_sym = back_to_entity_instance_sym(
            #     single_pred_results["pts_instance_mask"][0].copy(),
            #     gt_elementIds,
            # )
            pred_instance_mask_sym = single_pred_results["pts_instance_mask"][0].copy()
            pred_instance_bboxs_sym = back_to_instance_bbox_pred_sym1(
                pred_instance_mask_sym, element_bboxes
            )
            if self.ifeval:
                pred_instance_masks_inst_task_sym.append(
                    torch.tensor(pred_instance_mask_sym)
                )
                pred_instance_bboxs_inst_task_sym.append(pred_instance_bboxs_sym)
                pred_instance_labels_sym.append(
                    torch.tensor(single_pred_results["instance_labels"])
                )
                pred_instance_scores_sym.append(
                    torch.tensor(single_pred_results["instance_scores"])
                )
            if self.ifsave:
                bboxes = []
                labels = []
                scores = []
                for index, bbox in enumerate(pred_instance_bboxs_sym):
                    temp_bbox_x = bbox[0]
                    temp_bbox_y = bbox[1]
                    width = bbox[2] - temp_bbox_x
                    height = bbox[3] - temp_bbox_y
                    temp_semantic_id = single_pred_results["instance_labels"][index]
                    bboxes.append([temp_bbox_x, temp_bbox_y, width, height])
                    labels.append(temp_semantic_id)
                    scores.append(single_pred_results["instance_scores"][index])
                # pts_instance_mask = single_pred_results["pts_instance_mask"][0].copy()
                # pts_semantic_mask = single_pred_results["pts_semantic_mask"][0].copy()
                # pts_score = single_pred_results["instance_scores"].copy()
                # for i in range(len(pts_instance_mask)):
                #     temp_idx = np.where(pts_instance_mask[i] == True)[0]
                #     temp_semantic_id = np.argmax(np.bincount(pts_semantic_mask[temp_idx]))
                #     if temp_semantic_id in [30,31,32,33,34,35]:
                #         continue
                #     temp_coords = gt_coords[temp_idx]
                #     temp_bbox_x = temp_coords[:,0].min()
                #     temp_bbox_y = temp_coords[:,1].min()
                #     width = temp_coords[:,0].max() - temp_bbox_x
                #     height = temp_coords[:,1].max() - temp_bbox_y
                #     bboxes.append([temp_bbox_x, temp_bbox_y, width, height])
                #     labels.append(temp_semantic_id)
                #     scores.append(pts_score[i])
                save_data = {"bboxes": bboxes, "labels": labels, "scores": scores, "insts": pred_instance_mask_sym}
                save_filename = os.path.join(self.saved_dir, f"{npz_filename}.pkl")
                with open(save_filename, "wb") as f:
                    self.pickle.dump(save_data, f)
                print(save_filename+'is saved')

        if self.ifeval:
            logger.warning(
                "================================>Sympoint Metrics<================================"
            )

            sym_ret_sem = sympoint_semantic_eval(
                gt_semantic_masks_sem_task,
                pred_semantic_masks_sem_task,
                classes,
                ignore_index,
                logger=logger,
            )

            # sym_ret_sem_F1
            ret_sem_F1 = sympoint_semantic_F1_eval(
                gt_semantic_masks_sem_task,
                pred_semantic_masks_sem_task,
                element_weights,
                classes,
                ignore_index,
                logger=logger,
            )

            sym_ret_ins = sympoint_instance_eval(
                gt_semantic_masks_inst_task_sym,
                gt_instance_masks_inst_task_sym,
                pred_instance_masks_inst_task_sym,
                gt_semantic_bbox_labels_inst_task_sym,
                gt_instance_bboxs_inst_task_sym,
                pred_instance_bboxs_inst_task_sym,
                pred_instance_labels_sym,
                pred_instance_scores_sym,
                element_weights,
                classes,
                ignore_index,
                logger=logger,
            )
            metrics = dict()
            for ret, keys in zip(
                (ret_sem_F1, sym_ret_ins, sym_ret_sem), self.logger_keys
            ):
                for key in keys:
                    metrics[key] = ret[key]
            return metrics
        return dict()

