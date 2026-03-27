# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable


def fast_hist(preds, labels, num_classes):
    """Compute the confusion matrix for every batch.

    Args:
        preds (np.ndarray):  Prediction labels of points with shape of
        (num_points, ).
        labels (np.ndarray): Ground truth labels of points with shape of
        (num_points, ).
        num_classes (int): number of classes

    Returns:
        np.ndarray: Calculated confusion matrix.
    """

    k = (labels >= 0) & (labels < num_classes)
    bin_count = np.bincount(
        num_classes * labels[k].astype(int) + preds[k], minlength=num_classes**2
    )
    return bin_count[: num_classes**2].reshape(num_classes, num_classes)


def per_class_iou(hist):
    """Compute the per class iou.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        np.ndarray: Calculated per class iou
    """

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def get_acc(hist):
    """Compute the overall accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated overall acc
    """

    return np.diag(hist).sum() / hist.sum()


def get_acc_cls(hist):
    """Compute the class average accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated class average acc
    """

    return np.nanmean(np.diag(hist) / hist.sum(axis=1))


def semantic_seg_eval(gt_labels, seg_preds, label2cat, ignore_index, logger=None):
    """Semantic Segmentation  Evaluation.

    Evaluate the result of the Semantic Segmentation.

    Args:
        gt_labels (list[torch.Tensor]): Ground truth labels.
        seg_preds  (list[torch.Tensor]): Predictions.
        label2cat (dict): Map from label to category name.
        ignore_index (list[int]): A list of index that will be ignored in evaluation.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(seg_preds) == len(gt_labels)
    num_classes = len(label2cat)

    hist_list = []
    for i in range(len(gt_labels)):
        gt_seg = gt_labels[i].astype(np.int64)
        pred_seg = seg_preds[i].astype(np.int64)

        # filter out ignored points
        for index in ignore_index:
            pred_seg[gt_seg == index] = -1
            gt_seg[gt_seg == index] = -1

        # calculate one instance result
        hist_list.append(fast_hist(pred_seg, gt_seg, num_classes))

    iou = per_class_iou(sum(hist_list))
    # if ignore_index is in iou, replace it with nan
    for index in ignore_index:
        if index < len(iou):
            iou[index] = np.nan
    miou = np.nanmean(iou)
    acc = get_acc(sum(hist_list))
    acc_cls = get_acc_cls(sum(hist_list))

    header = ["classes"]
    for i in range(len(label2cat)):
        header.append(label2cat[i])
    header.extend(["miou", "acc", "acc_cls"])

    ret_dict = dict()
    table_columns = [["results"]]
    for i in range(len(label2cat)):
        ret_dict[label2cat[i]] = float(iou[i])
        table_columns.append([f"{iou[i]:.4f}"])
    ret_dict["miou"] = float(miou)
    ret_dict["acc"] = float(acc)
    ret_dict["acc_cls"] = float(acc_cls)

    table_columns.append([f"{miou:.4f}"])
    table_columns.append([f"{acc:.4f}"])
    table_columns.append([f"{acc_cls:.4f}"])

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log("\n" + table.table, logger=logger)

    return ret_dict


class PointWiseEval(object):
    def __init__(self, classes, num_classes=35, ignore_labels=[35, 36]) -> None:
        self.ignore_labels = ignore_labels
        self._num_classes = num_classes
        self._conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.float32
        )
        self._w_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.float32
        )
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self.classes = classes

    def update(self, pred_sem, gt_sem, weight_sem):

        pos_inds = (1 - np.isin(pred_sem, self.ignore_labels)) > 0
        pred = pred_sem[pos_inds]
        gt = gt_sem[pos_inds]
        weight = weight_sem[pos_inds]
        self._w_conf_matrix += np.bincount(
            (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
            minlength=self._w_conf_matrix.size,
            weights=weight,
        ).reshape(self._w_conf_matrix.shape)

        self._conf_matrix += np.bincount(
            (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
            minlength=self._conf_matrix.size,
        ).reshape(self._conf_matrix.shape)

    def get_eval(self, logger):
        # mIoU

        acc = np.zeros(self._num_classes, dtype=np.float64)
        iou = np.zeros(self._num_classes, dtype=np.float64)

        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        fp = self._conf_matrix.sum(axis=0)[:-1] - tp
        fn = self._conf_matrix.sum(axis=1)[:-1] - tp
        precision = tp / (tp + fp + 1e-4)
        recall = tp / (tp + fn + 1e-4)
        f1 = (2 * precision * recall) / (precision + recall + 1e-4)

        wtp = self._w_conf_matrix.diagonal()[:-1].astype(np.float64)
        wfp = self._w_conf_matrix.sum(axis=0)[:-1] - wtp
        wfn = self._w_conf_matrix.sum(axis=1)[:-1] - wtp
        wprecision = wtp / (wtp + wfp + 1e-4)
        wrecall = wtp / (wtp + wfn + 1e-4)
        wf1 = (2 * wprecision * wrecall) / (wprecision + wrecall + 1e-4)

        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        class_weights = pos_gt / (np.sum(pos_gt) + 1e-8)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / (pos_gt[acc_valid] + 1e-8)
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / (union[iou_valid] + 1e-8)
        macc = np.sum(acc[acc_valid]) / (np.sum(acc_valid) + 1e-8)
        miou = np.sum(iou[iou_valid]) / (np.sum(iou_valid) + 1e-8)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / (np.sum(pos_gt) + 1e-8)
        miou, fiou, pACC = 100 * miou, 100 * fiou, 100 * pacc

        micro_tp = np.sum(tp)
        micro_fp = np.sum(fp)
        micro_fn = np.sum(fn)
        micro_precision = micro_tp / (micro_tp + micro_fp + 1e-4)
        micro_recall = micro_tp / (micro_tp + micro_fn + 1e-4)
        micro_f1 = (2 * micro_precision * micro_recall) / (
            micro_precision + micro_recall + 1e-4
        )

        w_tp = np.sum(wtp)
        w_fp = np.sum(wfp)
        w_fn = np.sum(wfn)
        w_precision = w_tp / (w_tp + w_fp + 1e-4)
        w_recall = w_tp / (w_tp + w_fn + 1e-4)
        w_f1 = (2 * w_precision * w_recall) / (w_precision + w_recall + 1e-4)

        for i in range(self._num_classes):
            logger.info(
                # "Class_{idx} - {name} Result: iou/accuracy/f1/wf1 {iou:.4f}/{accuracy:.4f}/{f1:.4f}/{wf1:.4f}".format(
                "Class_{idx} - {name} Result: f1/wf1/{f1:.4f}/{wf1:.4f}".format(
                    idx=i,
                    name=self.classes[i],
                    # iou=iou[i] * 100,
                    # accuracy=acc[i] * 100,
                    f1=f1[i] * 100,
                    wf1=wf1[i] * 100,
                )
            )
        # logger.info(
        #     "mIoU / fwIoU / pACC : {:.3f} / {:.3f} / {:.3f}".format(miou, fiou, pACC)
        # )
        logger.info("F1/wF1 : {:.3f} / {:.3f} ".format(micro_f1 * 100, w_f1 * 100))
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

        return {"F1": micro_f1, "wF1": w_f1}


def semantic_F1_eval(gt_labels, seg_preds, ele_weights, classes, ignore_index, logger=None):
    pointWiseEval = PointWiseEval(classes, ignore_labels=ignore_index)
    for gt, prd, weights in zip(gt_labels, seg_preds, ele_weights):
        assert gt.shape == prd.shape == weights.shape
        pointWiseEval.update(gt, prd, weights)
    res = pointWiseEval.get_eval(logger)
    return res