import torch
import random
import torch.nn as nn
import torch_scatter
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean

try:
    import MinkowskiEngine as ME
except ImportError:
    class _MissingMinkowskiEngine:
        def __getattr__(self, name):
            raise ImportError(
                "MinkowskiEngine is required for Minkowski-based backbones/configs. "
                "Install it or use the PTv3/spconv configs."
            )

    ME = _MissingMinkowskiEngine()

from mmdet3d.registry import MODELS

# from mmengine.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from .mask_matrix_nms import mask_matrix_nms, mask_matrix_nms_cpu
from .pointcept.utils.structure import Point


class ScanNetOneFormer3DMixin:
    """Class contains common methods for ScanNet and ScanNet200."""

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(out, superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.predict_by_feat_semantic(out, superpoints)
        pan_res = self.predict_by_feat_panoptic(out, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(), pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy(),
            )
        ]

    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out["cls_preds"][0]
        pred_masks = out["masks"][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out["scores"][0] is not None:
            scores *= out["scores"][0]
        labels = (
            torch.arange(self.num_classes, device=scores.device).unsqueeze(0).repeat(len(cls_preds), 1).flatten(0, 1)
        )
        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode="floor")
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get("obj_normalization", None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get("nms", None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def predict_by_feat_semantic(self, out, superpoints, classes=None):
        """Predict semantic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `sem_preds` of shape (n_queries, n_semantic_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            classes (List[int] or None): semantic (stuff) class ids.

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, n_semantic_classe + 1),
        """
        if classes is None:
            classes = list(range(out["sem_preds"][0].shape[1] - 1))
        return out["sem_preds"][0][:, classes].argmax(dim=1)[superpoints]

    def predict_by_feat_panoptic(self, out, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        sem_map = self.predict_by_feat_semantic(out, superpoints, self.test_cfg.stuff_classes)
        mask_pred, labels, scores = self.predict_by_feat_instance(out, superpoints, self.test_cfg.pan_score_thr)
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes,
            mask_pred.shape[0] + n_stuff_classes,
            device=mask_pred.device,
        ).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0

        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask
        return sem_map, inst_map

    def _select_queries(self, x, gt_instances):
        """Select queries for train pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, n_channels).
            gt_instances (List[InstanceData_]): of len batch_size.
                Ground truth which can contain `labels` of shape (n_gts_i,),
                `sp_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                List[InstanceData_]: of len batch_size, each updated
                    with `query_masks` of shape (n_gts_i, n_queries_i).
        """
        queries = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
            else:
                queries.append(x[i])
                gt_instances[i].query_masks = gt_instances[i].sp_masks
        return queries, gt_instances


@MODELS.register_module()
class FloorplanOneFormer3D(Base3DDetector):
    r"""OneFormer3D for S3DIS dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        in_channels,
        num_channels,
        voxel_size,
        num_classes,
        min_spatial_shape,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super(Base3DDetector, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.pt = MODELS.build(backbone)
        self.pool = MODELS.build(pooling) if pooling else None
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
        }
        res = self.pt(data_dict)

        return res

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        if self.pool:
            reverse_mapping_list = [
                batch_data_samples[i].gt_pts_seg.pts_reverse_map for i in range(0, len(batch_data_samples))
            ]
        else:
            reverse_mapping_list = [
                torch.arange(
                    0,
                    len(batch_data_samples[i].gt_pts_seg.pts_reverse_map),
                    device="cuda",
                )
                for i in range(0, len(batch_data_samples))
            ]
        features_list = [batch_inputs_dict["points"][i] for i in range(0, len(batch_inputs_dict["points"]))]
        coords_list = [
            batch_inputs_dict["points"][i][:, :3].contiguous() for i in range(0, len(batch_inputs_dict["points"]))
        ]
        x = self.extract_feat(features_list, coords_list)

        out = []
        start = 0
        for i in range(0, len(reverse_mapping_list)):
            if self.pool:
                out.append(
                    self.pool(
                        x[start : start + len(reverse_mapping_list[i])],
                        reverse_mapping_list[i],
                    )
                )
            else:
                out.append(x[start : start + len(reverse_mapping_list[i])])
            start += len(reverse_mapping_list[i])

        x = self.decoder(out)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            # voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            # voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            segment_superpoints = reverse_mapping_list[i]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask

            assert segment_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = self.get_gt_semantic_masks(
                sem_mask, segment_superpoints, self.num_classes
            )
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = self.get_gt_inst_masks(inst_mask, segment_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        # coordinates, features, inverse_mapping, spatial_shape = self.collate(
        #     batch_inputs_dict["points"]
        # )
        # x = spconv.SparseConvTensor(
        #     features, coordinates, spatial_shape, len(batch_data_samples)
        # )

        # x = self.extract_feat(x)

        # x = self.decoder(x)

        if self.pool:
            reverse_mapping_list = [
                batch_data_samples[i].gt_pts_seg.pts_reverse_map for i in range(0, len(batch_data_samples))
            ]
        else:
            reverse_mapping_list = [
                torch.arange(
                    0,
                    len(batch_data_samples[i].gt_pts_seg.pts_reverse_map),
                    device="cuda",
                )
                for i in range(0, len(batch_data_samples))
            ]
        features_list = [batch_inputs_dict["points"][i] for i in range(0, len(batch_inputs_dict["points"]))]
        coords_list = [
            batch_inputs_dict["points"][i][:, :3].contiguous() for i in range(0, len(batch_inputs_dict["points"]))
        ]
        x = self.extract_feat(features_list, coords_list)

        out = []
        start = 0
        for i in range(0, len(reverse_mapping_list)):
            if self.pool:
                out.append(
                    self.pool(
                        x[start : start + len(reverse_mapping_list[i])],
                        reverse_mapping_list[i],
                    )
                )
            else:
                out.append(x[start : start + len(reverse_mapping_list[i])])
            start += len(reverse_mapping_list[i])

        x = self.decoder(out)

        results_list = self.predict_by_feat(x, torch.cat(reverse_mapping_list, dim=0))

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out["cls_preds"][0]
        pred_masks = out["masks"][0]
        pred_scores = out["scores"][0]

        inst_res = self.pred_inst(
            pred_masks[: -self.test_cfg.num_sem_cls, :],
            pred_scores[: -self.test_cfg.num_sem_cls, :],
            pred_labels[: -self.test_cfg.num_sem_cls, :],
            superpoints,
            self.test_cfg.inst_score_thr,
        )
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls :, :], superpoints)
        pan_res = self.pred_pan(pred_masks, pred_scores, pred_labels, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(), pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy(),
            )
        ]

    def pred_inst(self, pred_masks, pred_scores, pred_labels, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = (
            torch.arange(self.num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(self.decoder.num_queries - self.test_cfg.num_sem_cls, 1)
            .flatten(0, 1)
        )

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode="floor")
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get("obj_normalization", None):
            mask_pred_thr = mask_pred_sigmoid > self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get("nms", None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]  # TODO reverse 2 dense points
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]  # TODO reverse 2 dense points
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, pred_labels, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints
        )  # TODO reverse 2 dense points
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst(
            pred_masks[:-n_cls, :],
            pred_scores[:-n_cls, :],
            pred_labels[:-n_cls, :],
            superpoints,
            thr,
        )  # TODO reverse 2 dense points

        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)

        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0

        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):
        """Create ground truth semantic masks.

        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.

        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_classes).
        """

        mask = torch.nn.functional.one_hot(mask_src, num_classes=num_classes + 1)

        mask = mask.T
        sp_masks = scatter_mean(mask.float(), sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5
        sp_masks[-1, sp_masks.sum(axis=0) == 0] = True
        assert sp_masks.sum(axis=0).max().item() == 1

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.

        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).

        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks


@MODELS.register_module()
class FloorpicOneFormer3D(Base3DDetector):
    r"""OneFormer3D for S3DIS dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        voxel_size,
        num_classes,
        min_spatial_shape,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super(Base3DDetector, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.pt = MODELS.build(backbone)
        self.pool = MODELS.build(pooling) if pooling else None
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, coords, feats, offsets):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """

        offset_tensor = torch.cumsum(torch.tensor(offsets), dim=0).to("cuda")
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": coords,
            "feat": feats,
            "offset": offset_tensor,
        }
        res = self.pt(data_dict)
        out = []
        start = 0
        for i in offsets:
            out.append(res[start : start + i])
            start += i

        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for p in points
                ]
            )
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (el_p - el_p.min(0)[0]),
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for el_p, p in zip(elastic_points, points)
                ]
            )

        spatial_shape = torch.clip(coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict["points"], batch_inputs_dict.get("elastic_coords", None)
        )
        # x = spconv.SparseConvTensor(
        #     features, coordinates, spatial_shape, len(batch_data_samples)
        # )

        x = self.extract_feat(
            coordinates[:, 1:].contiguous(),
            features,
            [coordinates[coordinates[:, 0] == i].shape[0] for i in range(len(batch_data_samples))],
        )

        x = self.decoder(x)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask

            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = self.get_gt_semantic_masks(
                sem_mask, voxel_superpoints, self.num_classes
            )
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = self.get_gt_inst_masks(inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        # x = spconv.SparseConvTensor(
        #     features, coordinates, spatial_shape, len(batch_data_samples)
        # )

        # x = self.extract_feat(x)

        # x = self.decoder(x)

        # reverse_mapping_list = [
        #     batch_data_samples[i].gt_pts_seg.pts_reverse_map
        #     for i in range(0, len(batch_data_samples))
        # ]
        # features_list = [
        #     batch_inputs_dict["points"][i]
        #     for i in range(0, len(batch_inputs_dict["points"]))
        # ]
        # coords_list = [
        #     batch_inputs_dict["points"][i][:, :3].contiguous()
        #     for i in range(0, len(batch_inputs_dict["points"]))
        # ]
        # x = self.extract_feat(features_list, coords_list)

        # out = []
        # start = 0
        # for i in range(0, len(reverse_mapping_list)):
        #     out.append(
        #         self.pool(
        #             x[start : start + len(reverse_mapping_list[i])],
        #             reverse_mapping_list[i],
        #         )
        #     )
        #     start += len(reverse_mapping_list[i])

        coordinates, features, inverse_mapping, spatial_shape = self.collate(batch_inputs_dict["points"])
        x = self.extract_feat(
            coordinates[:, 1:].contiguous(),
            features,
            [coordinates[coordinates[:, 0] == i].shape[0] for i in range(len(batch_data_samples))],
        )

        x = self.decoder(x)

        results_list = self.predict_by_feat(x, inverse_mapping)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out["cls_preds"][0]
        pred_masks = out["masks"][0]
        pred_scores = out["scores"][0]

        inst_res = self.pred_inst(
            pred_masks[: -self.test_cfg.num_sem_cls, :],
            pred_scores[: -self.test_cfg.num_sem_cls, :],
            pred_labels[: -self.test_cfg.num_sem_cls, :],
            superpoints,
            self.test_cfg.inst_score_thr,
        )
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls :, :], superpoints)
        pan_res = self.pred_pan(pred_masks, pred_scores, pred_labels, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(), pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy(),
            )
        ]

    def pred_inst(self, pred_masks, pred_scores, pred_labels, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = (
            torch.arange(self.num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(self.decoder.num_queries - self.test_cfg.num_sem_cls, 1)
            .flatten(0, 1)
        )

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode="floor")
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get("obj_normalization", None):
            mask_pred_thr = mask_pred_sigmoid > self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get("nms", None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, pred_labels, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst(
            pred_masks[:-n_cls, :],
            pred_scores[:-n_cls, :],
            pred_labels[:-n_cls, :],
            superpoints,
            thr,
        )

        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)

        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0

        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):
        """Create ground truth semantic masks.

        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.

        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_classes).
        """

        mask = torch.nn.functional.one_hot(mask_src, num_classes=num_classes + 1)

        mask = mask.T
        sp_masks = scatter_mean(mask.float(), sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5
        sp_masks[-1, sp_masks.sum(axis=0) == 0] = True
        assert sp_masks.sum(axis=0).max().item() == 1

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.

        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).

        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks


@MODELS.register_module()
class FloorpicOneFormer3D_PTv3(FloorpicOneFormer3D):
    def __init__(
        self,
        voxel_size,
        grid_size,
        num_classes,
        min_spatial_shape,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            voxel_size,
            num_classes,
            min_spatial_shape,
            backbone,
            decoder,
            pooling,
            criterion,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.grid_size = grid_size

    def extract_feat(self, coords, feats, offsets):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset_tensor = torch.cumsum(torch.tensor(offsets), dim=0).to("cuda")
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": coords,
            "feat": feats,
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        out = []
        start = 0
        for i in offsets:
            out.append(feat[start : start + i])
            start += i
        return out


@MODELS.register_module()
class FloorplanOneFormer3D_Metric(FloorplanOneFormer3D):
    r"""Inherit from FloorplanOneFormer3D. Add lidar_paths to the output of predict_by_feat."""

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        if self.pool:
            reverse_mapping_list = [
                batch_data_samples[i].gt_pts_seg.pts_reverse_map for i in range(0, len(batch_data_samples))
            ]
        else:
            reverse_mapping_list = [
                torch.arange(
                    0,
                    len(batch_data_samples[i].gt_pts_seg.pts_reverse_map),
                    device="cuda",
                )
                for i in range(0, len(batch_data_samples))
            ]
        features_list = [batch_inputs_dict["points"][i] for i in range(0, len(batch_inputs_dict["points"]))]
        coords_list = [
            batch_inputs_dict["points"][i][:, :3].contiguous() for i in range(0, len(batch_inputs_dict["points"]))
        ]
        x = self.extract_feat(features_list, coords_list)

        out = []
        start = 0
        for i in range(0, len(reverse_mapping_list)):
            if self.pool:
                out.append(
                    self.pool(
                        x[start : start + len(reverse_mapping_list[i])],
                        reverse_mapping_list[i],
                    )
                )
            else:
                out.append(x[start : start + len(reverse_mapping_list[i])])

        x = self.decoder(out)

        # add lidar_paths to self.predict_by_feat
        lidar_paths = [sample.lidar_path for sample in batch_data_samples]
        results_list = self.predict_by_feat(x, torch.cat(reverse_mapping_list, dim=0), lidar_paths)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints, lidar_paths):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out["cls_preds"][0]
        pred_masks = out["masks"][0]
        pred_scores = out["scores"][0]

        inst_res = self.pred_inst(
            pred_masks[: -self.test_cfg.num_sem_cls, :],
            pred_scores[: -self.test_cfg.num_sem_cls, :],
            pred_labels[: -self.test_cfg.num_sem_cls, :],
            # superpoints,
            self.test_cfg.inst_score_thr,
        )
        sem_res = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls :, :],
            # superpoints,
        )
        if not inst_res:
            return None
        pts_semantic_mask = [sem_res.cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy(),
                lidar_paths=lidar_paths,  # add lidar_paths to the output
            )
        ]

    def pred_inst(self, pred_masks, pred_scores, pred_labels, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = (
            torch.arange(self.num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(self.decoder.num_queries - self.test_cfg.num_sem_cls, 1)
            .flatten(0, 1)
        )

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode="floor")
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get("obj_normalization", None):
            mask_pred_thr = mask_pred_sigmoid > self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get("nms", None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        # mask_pred = mask_pred[:, superpoints] # TODO reverse 2 dense points
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def pred_sem(self, pred_masks):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        # mask_pred = mask_pred[:, superpoints] # TODO reverse 2 dense points
        seg_map = mask_pred.argmax(0)
        return seg_map


@MODELS.register_module()
class FloorplanOneFormer3D_PTv3(FloorplanOneFormer3D):
    def __init__(
        self,
        in_channels,
        num_channels,
        voxel_size,
        grid_size,
        num_classes,
        min_spatial_shape,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            in_channels,
            num_channels,
            voxel_size,
            num_classes,
            min_spatial_shape,
            backbone,
            decoder,
            pooling,
            criterion,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.grid_size = grid_size

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat


# @MODELS.register_module()
# class FloorplanOneFormer3D_PTv3_PE(FloorplanOneFormer3D):
#     def __init__(
#         self,
#         num_pe: int,
#         in_channels,
#         num_channels,
#         voxel_size,
#         grid_size,
#         num_classes,
#         min_spatial_shape,
#         backbone=None,
#         decoder=None,
#         pooling=None,
#         criterion=None,
#         train_cfg=None,
#         test_cfg=None,
#         data_preprocessor=None,
#         init_cfg=None,
#     ):
#         super().__init__(
#             in_channels,
#             num_channels,
#             voxel_size,
#             num_classes,
#             min_spatial_shape,
#             backbone,
#             decoder,
#             pooling,
#             criterion,
#             train_cfg,
#             test_cfg,
#             data_preprocessor,
#             init_cfg,
#         )
#         self.grid_size = grid_size
#         self.num_pe = num_pe

#     def extract_feat(self, feat, coord):
#         """Extract features from sparse tensor.

#         Args:
#             feat (List[Tensor]): of len batch_size
#             coord (List[Tensor]): of len batch_size.

#         Returns:
#             List[Tensor]: of len batch_size,
#                 each of shape (n_points_i, n_channels).
#         """
#         # with torch.no_grad():
#         feats_cat = torch.cat(feat, dim=0)
#         coord_cat = torch.cat(coord, dim=0)
#         offset = [coord[i].shape[0] for i in range(0, len(coord))]
#         offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
#         if random.random() < 0.8:
#             offset_tensor = torch.cat(
#                 [offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0
#             )
#         feats_pe_l = list()
#         for i in range(self.num_pe):
#             # FIXME: should we assume coords are normalized?
#             # FIXME: why is 2^k pi x performing less than 2 k pi ?
#             twokpi = torch.pi * 2 * (i + 1)
#             feats_pe_l.append(torch.sin(feats_cat[:, :2] * twokpi))
#             feats_pe_l.append(torch.cos(feats_cat[:, :2] * twokpi))
#         feats_pe_l.append(feats_cat[:, 2:])

#         feats_pe = torch.cat(feats_pe_l, dim=1)

#         data_dict = {
#             "coord": coord_cat,
#             "feat": feats_pe,
#             "offset": offset_tensor,
#             "grid_size": self.grid_size,
#         }
#         point = Point(data_dict)
#         point = self.pt(point)
#         if isinstance(point, Point):
#             feat = point.feat
#         else:
#             feat = point
#         return feat


@MODELS.register_module()
class FloorplanOneFormer3D_PTv3_PE(FloorplanOneFormer3D):
    def __init__(
        self,
        num_pe: int,
        in_channels,
        num_channels,
        voxel_size,
        grid_size,
        num_classes,
        min_spatial_shape,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            in_channels,
            num_channels,
            voxel_size,
            num_classes,
            min_spatial_shape,
            backbone,
            decoder,
            pooling,
            criterion,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.grid_size = grid_size
        self.num_pe = num_pe

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        # with torch.no_grad():
        feats_cat = torch.cat(feat, dim=0)
        coord_cat = torch.cat(coord, dim=0)
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        feats_pe_l = list()
        for i in range(self.num_pe):
            # FIXME: should we assume coords are normalized?
            # FIXME: why is 2^k pi x performing less than 2 k pi ?
            twokpi = torch.pi * 2 * (i + 1)
            feats_pe_l.append(torch.sin(feats_cat[:, :2] * twokpi))
            feats_pe_l.append(torch.cos(feats_cat[:, :2] * twokpi))
        feats_pe_l.append(feats_cat[:, 2:])

        feats_pe = torch.cat(feats_pe_l, dim=1)

        data_dict = {
            "coord": coord_cat,
            "feat": feats_pe,
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat


@MODELS.register_module()
class FloorplanOneFormer3D_PTv3_Metric(FloorplanOneFormer3D_Metric):
    def __init__(
        self,
        in_channels,
        num_channels,
        voxel_size,
        grid_size,
        num_classes,
        min_spatial_shape,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            in_channels,
            num_channels,
            voxel_size,
            num_classes,
            min_spatial_shape,
            backbone,
            decoder,
            pooling,
            criterion,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.grid_size = grid_size

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat


@MODELS.register_module()
class ScanNetOneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    r"""OneFormer3D for ScanNet dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        in_channels,
        num_channels,
        voxel_size,
        num_classes,
        min_spatial_shape,
        query_thr,
        backbone=None,
        decoder=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super(Base3DDetector, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        )

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i] : batch_offsets[i + 1]])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for p in points
                ]
            )
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (el_p - el_p.min(0)[0]),
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for el_p, p in zip(elastic_points, points)
                ]
            )

        spatial_shape = torch.clip(coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg

            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict["points"], batch_inputs_dict.get("elastic_coords", None)
        )

        x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(x, sp_pts_masks, inverse_mapping, batch_offsets)
        queries, sp_gt_instances = self._select_queries(x, sp_gt_instances)
        x = self.decoder(x, queries)
        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(batch_inputs_dict["points"])

        x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(x, sp_pts_masks, inverse_mapping, batch_offsets)
        x = self.decoder(x, x)

        results_list = self.predict_by_feat(x, sp_pts_masks)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples


@MODELS.register_module()
class FloorscanOneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    r"""OneFormer3D for ScanNet dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        voxel_size,
        num_classes,
        min_spatial_shape,
        query_thr,
        backbone=None,
        decoder=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super(Base3DDetector, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.pt = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
        }
        res = self.pt(data_dict)

        return res

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg

            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        features_list = [batch_inputs_dict["points"][i] for i in range(0, len(batch_inputs_dict["points"]))]
        coords_list = [
            batch_inputs_dict["points"][i][:, :3].contiguous() for i in range(0, len(batch_inputs_dict["points"]))
        ]

        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(features_list, coords_list)

        x = scatter_mean(x, sp_pts_masks, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i] : batch_offsets[i + 1]])

        queries, sp_gt_instances = self._select_queries(out, sp_gt_instances)
        x = self.decoder(out, queries)
        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        features_list = [batch_inputs_dict["points"][i] for i in range(0, len(batch_inputs_dict["points"]))]
        coords_list = [
            batch_inputs_dict["points"][i][:, :3].contiguous() for i in range(0, len(batch_inputs_dict["points"]))
        ]

        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(features_list, coords_list)

        x = scatter_mean(x, sp_pts_masks, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i] : batch_offsets[i + 1]])

        x = self.decoder(out, out)

        results_list = self.predict_by_feat(x, sp_pts_masks)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples


@MODELS.register_module()
class FloorscanOneFormer3D_PTv3(FloorscanOneFormer3D):
    def __init__(
        self,
        voxel_size,
        grid_size,
        num_classes,
        min_spatial_shape,
        query_thr,
        backbone=None,
        decoder=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            voxel_size,
            num_classes,
            min_spatial_shape,
            query_thr,
            backbone,
            decoder,
            criterion,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.grid_size = grid_size

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat


@MODELS.register_module()
class ScanNet200OneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.

    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        voxel_size,
        num_classes,
        query_thr,
        backbone=None,
        neck=None,
        decoder=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super(Base3DDetector, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict["points"])):
            if "elastic_coords" in batch_inputs_dict:
                coordinates.append(batch_inputs_dict["elastic_coords"][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict["points"][i][:, :3])
            features.append(batch_inputs_dict["points"][i][:, 3:])

        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device,
        )
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse())
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field).features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        x = scatter_mean(x, torch.cat(sp_pts_masks), dim=0)  # todo: do we need dim?

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[: i + 1])
            features.append(x[begin:end])
        return features

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        queries, gt_instances = self._select_queries(x, gt_instances)
        x = self.decoder(x, queries)
        return self.criterion(x, gt_instances)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        assert len(batch_data_samples) == 1
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        x = self.decoder(x, x)
        pred_pts_seg = self.predict_by_feat(x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples


import traceback


# @MODELS.register_module(name=traceback.print_stack())
@MODELS.register_module()
class S3DISOneFormer3D(Base3DDetector):
    r"""OneFormer3D for S3DIS dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        in_channels,
        num_channels,
        voxel_size,
        num_classes,
        min_spatial_shape,
        backbone=None,
        decoder=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super(Base3DDetector, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        )

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for p in points
                ]
            )
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (el_p - el_p.min(0)[0]),
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for el_p, p in zip(elastic_points, points)
                ]
            )

        spatial_shape = torch.clip(coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict["points"], batch_inputs_dict.get("elastic_coords", None)
        )
        x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        x = self.decoder(x)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = self.get_gt_semantic_masks(
                sem_mask, voxel_superpoints, self.num_classes
            )
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = self.get_gt_inst_masks(inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        coordinates, features, inverse_mapping, spatial_shape = self.collate(batch_inputs_dict["points"])
        x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        x = self.decoder(x)

        results_list = self.predict_by_feat(x, inverse_mapping)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out["cls_preds"][0]
        pred_masks = out["masks"][0]
        pred_scores = out["scores"][0]

        inst_res = self.pred_inst(
            pred_masks[: -self.test_cfg.num_sem_cls, :],
            pred_scores[: -self.test_cfg.num_sem_cls, :],
            pred_labels[: -self.test_cfg.num_sem_cls, :],
            superpoints,
            self.test_cfg.inst_score_thr,
        )
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls :, :], superpoints)
        pan_res = self.pred_pan(pred_masks, pred_scores, pred_labels, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(), pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy(),
            )
        ]

    def pred_inst(self, pred_masks, pred_scores, pred_labels, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = (
            torch.arange(self.num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(self.decoder.num_queries - self.test_cfg.num_sem_cls, 1)
            .flatten(0, 1)
        )

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode="floor")
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get("obj_normalization", None):
            mask_pred_thr = mask_pred_sigmoid > self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get("nms", None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, pred_labels, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst(
            pred_masks[:-n_cls, :],
            pred_scores[:-n_cls, :],
            pred_labels[:-n_cls, :],
            superpoints,
            thr,
        )

        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)

        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0

        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):
        """Create ground truth semantic masks.

        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.

        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_classes).
        """

        mask = torch.nn.functional.one_hot(mask_src, num_classes=num_classes + 1)

        mask = mask.T
        sp_masks = scatter_mean(mask.float(), sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5
        sp_masks[-1, sp_masks.sum(axis=0) == 0] = True
        assert sp_masks.sum(axis=0).max().item() == 1

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.

        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).

        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks


@MODELS.register_module()
class S3DISOneFormer3D_Metric(S3DISOneFormer3D):
    """
    Inherit from S3DISOneFormer3D
    """

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(batch_inputs_dict["points"])
        x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        x = self.decoder(x)

        lidar_paths = [sample.lidar_path for sample in batch_data_samples]
        results_list = self.predict_by_feat(x, inverse_mapping, lidar_paths)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints, lidar_paths):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out["cls_preds"][0]
        pred_masks = out["masks"][0]
        pred_scores = out["scores"][0]

        inst_res = self.pred_inst(
            pred_masks[: -self.test_cfg.num_sem_cls, :],
            pred_scores[: -self.test_cfg.num_sem_cls, :],
            pred_labels[: -self.test_cfg.num_sem_cls, :],
            superpoints,
            self.test_cfg.inst_score_thr,
        )
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls :, :], superpoints)
        pan_res = self.pred_pan(pred_masks, pred_scores, pred_labels, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(), pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy(),
                lidar_paths=lidar_paths,
            )
        ]


@MODELS.register_module()
class InstanceOnlyOneFormer3D(Base3DDetector):
    r"""InstanceOnlyOneFormer3D for training on different datasets jointly.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        prefix_2dataset (string): Prefix for the second dataset.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        in_channels,
        num_channels,
        voxel_size,
        num_classes_1dataset,
        num_classes_2dataset,
        prefix_1dataset,
        prefix_2dataset,
        min_spatial_shape,
        backbone=None,
        decoder=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super(InstanceOnlyOneFormer3D, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes_1dataset = num_classes_1dataset
        self.num_classes_2dataset = num_classes_2dataset

        self.prefix_1dataset = prefix_1dataset
        self.prefix_2dataset = prefix_2dataset

        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        )

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for p in points
                ]
            )
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (el_p - el_p.min(0)[0]),
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for el_p, p in zip(elastic_points, points)
                ]
            )

        spatial_shape = torch.clip(coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict["points"], batch_inputs_dict.get("elastic_coords", None)
        )
        x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_masks = S3DISOneFormer3D.get_gt_inst_masks(
                inst_mask, voxel_superpoints
            )
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(batch_inputs_dict["points"])
        x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        results_list = self.predict_by_feat(x, inverse_mapping, scene_names)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints, scene_names):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            scene_names (List[string]): of len 1, which contain scene name.

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`,
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out["cls_preds"]
        pred_masks = out["masks"]
        pred_scores = out["scores"]
        scene_name = scene_names[0]

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]

        if self.prefix_1dataset in scene_name:
            labels = (
                torch.arange(self.num_classes_1dataset, device=scores.device)
                .unsqueeze(0)
                .repeat(self.decoder.num_queries_1dataset, 1)
                .flatten(0, 1)
            )
        elif self.prefix_2dataset in scene_name:
            labels = (
                torch.arange(self.num_classes_2dataset, device=scores.device)
                .unsqueeze(0)
                .repeat(self.decoder.num_queries_2dataset, 1)
                .flatten(0, 1)
            )
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        if self.prefix_1dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_1dataset, rounding_mode="floor")
        elif self.prefix_2dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_2dataset, rounding_mode="floor")
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')

        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get("obj_normalization", None):
            mask_pred_thr = mask_pred_sigmoid > self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get("nms", None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return [
            PointData(
                pts_instance_mask=mask_pred,
                instance_labels=labels,
                instance_scores=scores,
            )
        ]


@MODELS.register_module()
class Floors3dInstanceOnlyOneFormer3D(Base3DDetector):
    r"""InstanceOnlyOneFormer3D for training on different datasets jointly.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes_dataset (int): Number of classes in the first dataset.
        prefix_dataset (string): Prefix for the first dataset.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        num_classes_dataset,
        prefix_dataset,
        min_spatial_shape,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super(Floors3dInstanceOnlyOneFormer3D, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes_dataset = num_classes_dataset

        self.prefix_dataset = prefix_dataset

        self.pt = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.pool = MODELS.build(pooling) if pooling else None
        self.criterion = MODELS.build(criterion)
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
        }
        res = self.pt(data_dict)

        return res

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        reverse_mapping_list = [
            batch_data_samples[i].gt_pts_seg.pts_reverse_map for i in range(0, len(batch_data_samples))
        ]
        features_list = [batch_inputs_dict["points"][i] for i in range(0, len(batch_inputs_dict["points"]))]
        coords_list = [
            batch_inputs_dict["points"][i][:, :3].contiguous() for i in range(0, len(batch_inputs_dict["points"]))
        ]
        x = self.extract_feat(features_list, coords_list)

        out = []
        start = 0
        for i in range(0, len(reverse_mapping_list)):
            if self.pool:
                out.append(
                    self.pool(
                        x[start : start + len(reverse_mapping_list[i])],
                        reverse_mapping_list[i],
                    )
                )
            else:
                out.append(x[start : start + len(reverse_mapping_list[i])])
            start += len(reverse_mapping_list[i])

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(out, scene_names)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            segment_superpoints = reverse_mapping_list[i]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            assert segment_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_masks = FloorplanOneFormer3D.get_gt_inst_masks(
                inst_mask, segment_superpoints
            )
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        reverse_mapping_list = [
            batch_data_samples[i].gt_pts_seg.pts_reverse_map for i in range(0, len(batch_data_samples))
        ]
        features_list = [batch_inputs_dict["points"][i] for i in range(0, len(batch_inputs_dict["points"]))]
        coords_list = [
            batch_inputs_dict["points"][i][:, :3].contiguous() for i in range(0, len(batch_inputs_dict["points"]))
        ]
        x = self.extract_feat(features_list, coords_list)

        out = []
        start = 0
        for i in range(0, len(reverse_mapping_list)):
            out.append(
                self.pool(
                    x[start : start + len(reverse_mapping_list[i])],
                    reverse_mapping_list[i],
                )
            )
            start += len(reverse_mapping_list[i])

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(out, scene_names)

        results_list = self.predict_by_feat(x, torch.cat(reverse_mapping_list, dim=0), scene_names)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints, scene_names):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            scene_names (List[string]): of len 1, which contain scene name.

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`,
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out["cls_preds"]
        pred_masks = out["masks"]
        pred_scores = out["scores"]
        scene_name = scene_names[0]

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]

        if self.prefix_dataset in scene_name:
            labels = (
                torch.arange(self.num_classes_dataset, device=scores.device)
                .unsqueeze(0)
                .repeat(self.decoder.num_queries_dataset, 1)
                .flatten(0, 1)
            )
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        if self.prefix_dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_dataset, rounding_mode="floor")
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')

        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get("obj_normalization", None):
            mask_pred_thr = mask_pred_sigmoid > self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get("nms", None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return [
            PointData(
                pts_instance_mask=mask_pred,
                instance_labels=labels,
                instance_scores=scores,
            )
        ]


@MODELS.register_module()
class Floors3dInstanceOnlyOneFormer3D_PTv3(Floors3dInstanceOnlyOneFormer3D):
    def __init__(
        self,
        num_classes_dataset,
        prefix_dataset,
        min_spatial_shape,
        grid_size,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            num_classes_dataset,
            prefix_dataset,
            min_spatial_shape,
            backbone,
            decoder,
            pooling,
            criterion,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.grid_size = grid_size

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat


@MODELS.register_module()
class FloorplanOneFormer3D_PTv3_ColorPooling(FloorplanOneFormer3D_PTv3):
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        reverse_mapping_list = [
            batch_data_samples[i].gt_pts_seg.pts_reverse_map for i in range(0, len(batch_data_samples))
        ]
        features_list = [batch_inputs_dict["points"][i] for i in range(0, len(batch_inputs_dict["points"]))]
        coords_list = [
            batch_inputs_dict["points"][i][:, :3].contiguous() for i in range(0, len(batch_inputs_dict["points"]))
        ]
        x = self.extract_feat(features_list, coords_list)

        out = []
        start = 0
        for i in range(0, len(reverse_mapping_list)):
            if self.pool:
                out.append(
                    self.pool(
                        x[start : start + len(reverse_mapping_list[i])],
                        reverse_mapping_list[i],
                        batch_inputs_dict["points"][i][:, 3:],
                    )
                )
            else:
                out.append(x[start : start + len(reverse_mapping_list[i])])
            start += len(reverse_mapping_list[i])

        x = self.decoder(out)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            # voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            # voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            segment_superpoints = reverse_mapping_list[i]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask

            assert segment_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = self.get_gt_semantic_masks(
                sem_mask, segment_superpoints, self.num_classes
            )
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = self.get_gt_inst_masks(inst_mask, segment_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        # coordinates, features, inverse_mapping, spatial_shape = self.collate(
        #     batch_inputs_dict["points"]
        # )
        # x = spconv.SparseConvTensor(
        #     features, coordinates, spatial_shape, len(batch_data_samples)
        # )

        # x = self.extract_feat(x)

        # x = self.decoder(x)

        reverse_mapping_list = [
            batch_data_samples[i].gt_pts_seg.pts_reverse_map for i in range(0, len(batch_data_samples))
        ]
        features_list = [batch_inputs_dict["points"][i] for i in range(0, len(batch_inputs_dict["points"]))]
        coords_list = [
            batch_inputs_dict["points"][i][:, :3].contiguous() for i in range(0, len(batch_inputs_dict["points"]))
        ]
        x = self.extract_feat(features_list, coords_list)

        out = []
        start = 0
        for i in range(0, len(reverse_mapping_list)):
            out.append(
                self.pool(
                    x[start : start + len(reverse_mapping_list[i])],
                    reverse_mapping_list[i],
                    batch_inputs_dict["points"][i][:, 3:],
                )
            )
            start += len(reverse_mapping_list[i])

        x = self.decoder(out)

        results_list = self.predict_by_feat(x, torch.cat(reverse_mapping_list, dim=0))

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples


@MODELS.register_module()
class FloorplanOneFormer3D_Metric_large(FloorplanOneFormer3D):
    r"""Inherit from FloorplanOneFormer3D. Add lidar_paths to the output of predict_by_feat."""

    def __init__(
        self,
        in_channels,
        num_channels,
        voxel_size,
        num_classes,
        min_spatial_shape,
        sliding_step,
        window_size,
        sparse_nms,
        nms_device,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            in_channels,
            num_channels,
            voxel_size,
            num_classes,
            min_spatial_shape,
            backbone,
            decoder,
            pooling,
            criterion,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.sliding_step = sliding_step
        self.window_size = window_size
        self.nms_device = nms_device
        self.sparse_nms = sparse_nms

    def pool_(self, x, reverse_map):
        out = []
        if self.pool:
            out.append(
                self.pool(
                    x,
                    reverse_map,
                )
            )
        else:
            out.append(x)
        return out

    def pred_inst_value(self, pred_masks, pred_scores, pred_labels):
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = (
            torch.arange(self.num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(self.decoder.num_queries - self.test_cfg.num_sem_cls, 1)
            .flatten(0, 1)
        )

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode="floor")
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get("obj_normalization", None):
            mask_pred_thr = mask_pred_sigmoid > self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores
        return scores, labels, mask_pred_sigmoid

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        assert len(batch_inputs_dict["points"]) == 1
        EPS = 1e-13
        x_list = []
        width = int(batch_inputs_dict["points"][0][:, 0].max()) + self.window_size
        height = int(batch_inputs_dict["points"][0][:, 1].max()) + self.window_size

        for y in range(self.window_size, height, self.sliding_step):
            for x in range(self.window_size, width, self.sliding_step):
                mask = torch.where(
                    (batch_inputs_dict["points"][0][:, 0] <= x + EPS)
                    & (batch_inputs_dict["points"][0][:, 0] >= x - self.window_size - EPS)
                    & (batch_inputs_dict["points"][0][:, 1] <= y + EPS)
                    & (batch_inputs_dict["points"][0][:, 1] >= y - self.window_size - EPS)
                )
                if len(mask[0]):
                    reverse_map = (batch_data_samples[0].gt_pts_seg.pts_reverse_map[mask] + 2).to("cuda")
                    for idx, inst in enumerate(torch.unique(reverse_map)):
                        mask_reverse = reverse_map == inst
                        reverse_map[mask_reverse] = idx

                    features = (
                        batch_inputs_dict["points"][0][mask]
                        - torch.tensor(
                            [x - self.window_size, y - self.window_size, 0, 0, 0, 0],
                            device="cuda",
                        )
                    ).to("cuda")
                    coords = (
                        (
                            batch_inputs_dict["points"][0][:, :3][mask]
                            - torch.tensor(
                                [x - self.window_size, y - self.window_size, 0],
                                device="cuda",
                            )
                        )
                        .contiguous()
                        .to("cuda")
                    )

                    x = self.extract_feat([features], coord=[coords])
                    out = self.pool_(x, reverse_map)
                    x = self.decoder(out)
                    x["range"] = [x, y]
                    x["inverse"] = reverse_map
                    x["indices"] = mask
                    x_list.append(x)

        lidar_paths = [sample.lidar_path for sample in batch_data_samples]
        results_list = self.predict_by_feat(x_list, batch_data_samples[0].gt_pts_seg.pts_reverse_map, lidar_paths)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, x_list, superpoints, lidar_paths):
        entity_num = len(torch.unique(superpoints))
        large_sem_entity = torch.zeros(entity_num, self.test_cfg.num_sem_cls, device="cuda", dtype=torch.int32)
        large_label = torch.zeros(self.test_cfg.topk_insts * len(x_list), device="cuda", dtype=torch.int32)
        large_score = torch.zeros(self.test_cfg.topk_insts * len(x_list), device="cuda")
        large_mask = torch.zeros(self.test_cfg.topk_insts * len(x_list), entity_num, device="cuda")

        index = 0
        for x_item in x_list:
            # superpoints = x_item["inverse"]
            pred_labels = x_item["cls_preds"][0]
            pred_masks = x_item["masks"][0]
            pred_scores = x_item["scores"][0]
            sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls :, :], x_item["inverse"])
            one_hot = torch.nn.functional.one_hot(sem_res, num_classes=self.test_cfg.num_sem_cls)
            for eId in torch.unique(superpoints[x_item["indices"]]):
                mask_temp = superpoints[x_item["indices"]] == eId
                result = one_hot[mask_temp].sum(dim=0)
                large_sem_entity[eId] += result

            x_score, x_label, x_mask = self.pred_inst_value(
                pred_masks[: -self.test_cfg.num_sem_cls, :],
                pred_scores[: -self.test_cfg.num_sem_cls, :],
                pred_labels[: -self.test_cfg.num_sem_cls, :],
            )
            # temp_instmask_entity = torch.zeros(query_num,entity_num,device='cuda')
            for i in range(self.test_cfg.topk_insts):
                large_mask[
                    index,
                    torch.unique(
                        superpoints[x_item["indices"]],
                        sorted=False,
                    ),
                ] = x_mask[i]
                large_label[index] = x_label[i]
                large_score[index] = x_score[i]
                index += 1

        assert self.test_cfg.get("nms", None)
        kernel = self.test_cfg.matrix_nms_kernel
        if self.sparse_nms:
            import oneformer3d.mask_matrix_nms_cupy as nms
            import scipy.sparse
            import cupy

            if self.nms_device == "cpu":
                import numpy

                nms.numpy = numpy
                nms.sparse = scipy.sparse
                scores, labels, mask_pred_sigmoid, _ = nms.mask_matrix_nms(
                    large_mask.cpu(),
                    large_label.cpu(),
                    large_score.cpu(),
                    kernel=kernel,
                )
                scores = scores.cuda()
                labels = labels.cuda()
                mask_pred_sigmoid = mask_pred_sigmoid.cuda()
            elif self.nms_device == "gpu":
                nms.numpy = cupy
                nms.sparse = cupy.sparse
                # may use another device
                cupy.cuda.runtime.setDevice(0)
                cupy.cuda.runtime.getDevice()
                scores, labels, mask_pred_sigmoid, _ = nms.mask_matrix_nms(
                    large_mask, large_label, large_score, kernel=kernel
                )
        else:
            if self.nms_device == "cpu":
                scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms_cpu(
                    large_mask, large_label, large_score, kernel=kernel
                )
            elif self.nms_device == "gpu":
                scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                    large_mask, large_label, large_score, kernel=kernel
                )
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        # score_thr
        score_mask = scores > self.test_cfg.inst_score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask][:, superpoints]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        inst_res = mask_pred, labels, scores
        sem_res = torch.argmax(large_sem_entity, dim=1).to(torch.int32)
        sem_res = torch.unsqueeze(sem_res, 0)[:, superpoints]

        pts_semantic_mask = [sem_res.cpu().numpy()[0], None]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(), None]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy(),
                lidar_paths=lidar_paths,  # add lidar_paths to the output
            )
        ]


@MODELS.register_module()
class FloorplanOneFormer3D_Metric_entity_large(FloorplanOneFormer3D_Metric_large):
    def predict_by_feat(self, x_list, superpoints, lidar_paths):
        entity_num = len(torch.unique(superpoints))
        large_sem_entity = torch.zeros(entity_num, self.test_cfg.num_sem_cls, device="cuda", dtype=torch.int32)
        large_label = torch.zeros(self.test_cfg.topk_insts * len(x_list), device="cuda", dtype=torch.int32)
        large_score = torch.zeros(self.test_cfg.topk_insts * len(x_list), device="cuda")
        large_mask = torch.zeros(self.test_cfg.topk_insts * len(x_list), entity_num, device="cuda")

        index = 0
        for x_item in x_list:
            # superpoints = x_item["inverse"]
            pred_labels = x_item["cls_preds"][0]
            pred_masks = x_item["masks"][0]
            pred_scores = x_item["scores"][0]
            sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls :, :], x_item["inverse"])
            one_hot = torch.nn.functional.one_hot(sem_res, num_classes=self.test_cfg.num_sem_cls)
            for eId in torch.unique(superpoints[x_item["indices"]]):
                mask_temp = superpoints[x_item["indices"]] == eId
                result = one_hot[mask_temp].sum(dim=0)
                large_sem_entity[eId] += result

            x_score, x_label, x_mask = self.pred_inst_value(
                pred_masks[: -self.test_cfg.num_sem_cls, :],
                pred_scores[: -self.test_cfg.num_sem_cls, :],
                pred_labels[: -self.test_cfg.num_sem_cls, :],
            )
            # temp_instmask_entity = torch.zeros(query_num,entity_num,device='cuda')
            for i in range(self.test_cfg.topk_insts):
                large_mask[
                    index,
                    torch.unique(
                        superpoints[x_item["indices"]],
                        sorted=False,
                    ),
                ] = x_mask[i]
                large_label[index] = x_label[i]
                large_score[index] = x_score[i]
                index += 1

        sem_res = torch.argmax(large_sem_entity, dim=1).to(torch.int32)
        sem_res = torch.unsqueeze(sem_res, 0)
        del large_sem_entity
        torch.cuda.empty_cache()

        assert self.test_cfg.get("nms", None)
        kernel = self.test_cfg.matrix_nms_kernel
        if self.sparse_nms:
            import oneformer3d.mask_matrix_nms_cupy as nms
            import scipy.sparse
            import cupy

            if self.nms_device == "cpu":
                import numpy

                nms.numpy = numpy
                nms.sparse = scipy.sparse
                scores, labels, mask_pred_sigmoid, _ = nms.mask_matrix_nms(
                    large_mask.cpu(),
                    large_label.cpu(),
                    large_score.cpu(),
                    kernel=kernel,
                )
                scores = scores.cuda()
                labels = labels.cuda()
                mask_pred_sigmoid = mask_pred_sigmoid.cuda()
            elif self.nms_device == "gpu":
                nms.numpy = cupy
                nms.sparse = cupy.sparse
                # may use another device
                cupy.cuda.runtime.setDevice(0)
                cupy.cuda.runtime.getDevice()
                scores, labels, mask_pred_sigmoid, _ = nms.mask_matrix_nms(
                    large_mask, large_label, large_score, kernel=kernel
                )
        else:
            if self.nms_device == "cpu":
                scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms_cpu(
                    large_mask, large_label, large_score, kernel=kernel
                )
            elif self.nms_device == "gpu":
                scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                    large_mask, large_label, large_score, kernel=kernel
                )

        del large_mask
        del large_label
        del large_score
        torch.cuda.empty_cache()

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        del mask_pred_sigmoid
        torch.cuda.empty_cache()
        # score_thr
        score_mask = scores > self.test_cfg.inst_score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        inst_res = mask_pred, labels, scores

        pts_semantic_mask = [sem_res.cpu().numpy()[0], None]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(), None]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy(),
                lidar_paths=lidar_paths,  # add lidar_paths to the output
            )
        ]


@MODELS.register_module()
class FloorplanOneFormer3D_PTv3_Metric_large(FloorplanOneFormer3D_Metric_large):
    def __init__(
        self,
        in_channels,
        num_channels,
        voxel_size,
        grid_size,
        num_classes,
        min_spatial_shape,
        sliding_step,
        window_size,
        sparse_nms,
        nms_device,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            in_channels,
            num_channels,
            voxel_size,
            num_classes,
            min_spatial_shape,
            sliding_step,
            window_size,
            sparse_nms,
            nms_device,
            backbone,
            decoder,
            pooling,
            criterion,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.grid_size = grid_size

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat


@MODELS.register_module()
class FloorplanOneFormer3D_PTv3_Metric_entity_large(FloorplanOneFormer3D_Metric_entity_large):
    def __init__(
        self,
        in_channels,
        num_channels,
        voxel_size,
        grid_size,
        num_classes,
        min_spatial_shape,
        sliding_step,
        window_size,
        sparse_nms,
        nms_device,
        backbone=None,
        decoder=None,
        pooling=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            in_channels,
            num_channels,
            voxel_size,
            num_classes,
            min_spatial_shape,
            sliding_step,
            window_size,
            sparse_nms,
            nms_device,
            backbone,
            decoder,
            pooling,
            criterion,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.grid_size = grid_size

    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat


@MODELS.register_module()
class FloorplanOneFormer3D_PTv3_Onlycoord(FloorplanOneFormer3D_PTv3):
    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        feat = coord.copy()
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat


@MODELS.register_module()
class FloorplanOneFormer3D_PTv3_Onlycoord_Metric(FloorplanOneFormer3D_PTv3_Metric):
    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        feat = coord.copy()
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat


@MODELS.register_module()
class FloorplanOneFormer3D_PTv3_Onlycoord_Metric_entity_large(FloorplanOneFormer3D_PTv3_Metric_entity_large):
    def extract_feat(self, feat, coord):
        """Extract features from sparse tensor.

        Args:
            feat (List[Tensor]): of len batch_size
            coord (List[Tensor]): of len batch_size.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        offset = [coord[i].shape[0] for i in range(0, len(coord))]
        feat = coord.copy()
        offset_tensor = torch.cumsum(torch.tensor(offset), dim=0).to(feat[0].device)
        if random.random() < 0.8:
            offset_tensor = torch.cat([offset_tensor[1:-1:2], offset_tensor[-1].unsqueeze(0)], dim=0)
        data_dict = {
            "coord": torch.cat(coord, dim=0),
            "feat": torch.cat(feat, dim=0),
            "offset": offset_tensor,
            "grid_size": self.grid_size,
        }
        point = Point(data_dict)
        point = self.pt(point)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        return feat
