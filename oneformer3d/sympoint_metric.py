import numpy as np
import torch.nn.functional as F

# import time


class PointWiseEvalF1(object):
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
                "Class_{idx} - {name} Result: f1/wf1/{f1:.4f}/{wf1:.4f}".format(
                    idx=i,
                    name=self.classes[i],
                    f1=f1[i] * 100,
                    wf1=wf1[i] * 100,
                )
            )
        logger.info("F1/wF1 : {:.3f} / {:.3f} ".format(micro_f1 * 100, w_f1 * 100))
        # logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

        return {"F1": micro_f1 * 100, "wF1": w_f1 * 100}


class PointWiseEval(object):
    def __init__(self, classes, num_classes=35, ignore_labels=[35, 36]) -> None:
        self.ignore_labels = ignore_labels
        self._num_classes = num_classes
        self._conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.float32
        )
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._class_names = classes

    def update(self, pred_sem, gt_sem):
        pos_inds = (1 - np.isin(pred_sem, self.ignore_labels)) > 0
        pred = pred_sem[pos_inds]
        gt = gt_sem[pos_inds]

        self._conf_matrix += np.bincount(
            (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
            minlength=self._conf_matrix.size,
        ).reshape(self._conf_matrix.shape)

    def get_eval(self, logger):
        # mIoU
        acc = np.full(self._num_classes, np.nan, dtype=np.float64)
        iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
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
        for i in range(self._num_classes):
            logger.info(
                "Class_{}  IoU: {:.3f}".format(self._class_names[i], iou[i] * 100)
            )

        logger.info(
            "mIoU / fwIoU / pACC : {:.3f} / {:.3f} / {:.3f}".format(miou, fiou, pACC)
        )

        return {"mIoU": miou, "fwIoU": fiou, "pACC": pACC}


class InstanceEval(object):
    def __init__(self, classes, num_classes=35, ignore_labels=[35, 36]) -> None:
        self.ignore_labels = ignore_labels
        self._num_classes = num_classes
        self._class_names = classes
        self.min_obj_score = 0.1
        self.IoU_thres = 0.5

        self.tp_classes = np.zeros(num_classes)
        self.tp_classes_values = np.zeros(num_classes)
        self.fp_classes = np.zeros(num_classes)
        self.fn_classes = np.zeros(num_classes)
        self.thing_class = [i for i in range(30)]
        self.stuff_class = [30, 31, 32, 33, 34]

        self.IoU_threshes = np.arange(0.5, 1, 0.05)
        self.tp_classes_thresh = [np.zeros(num_classes) for _ in range(len(self.IoU_threshes))]
        self.fp_classes_thresh = [np.zeros(num_classes) for _ in range(len(self.IoU_threshes))]
        self.fn_classes_thresh = [np.zeros(num_classes) for _ in range(len(self.IoU_threshes))]
        
    def compute_ap(self, precision, recall):
        # 插值 precision 和 recall
        mrec = [0, recall.tolist(), 1]
        mpre = [0, precision.tolist(), 0]
        
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        
        # 计算 AP
        ap = 0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap
    
    def compute_precision_recall(self, tp, fp, fn):
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        return precision, recall
    
    def compute_apx(self, tp_classes, fp_classes, fn_classes, iou_thresh, logger):
        # 计算每个类别的 Precision 和 Recall
        precision_list = []
        recall_list = []
        ap_list = []
        for i in range(len(self.thing_class)):
            precision, recall = self.compute_precision_recall(tp_classes[i], fp_classes[i], fn_classes[i])
            precision_list.append(precision)
            recall_list.append(recall)
            ap = self.compute_ap(precision, recall)
            ap_list.append(ap)

        # 计算 mAP
        APx = sum(ap_list) / len(ap_list)
        
        # 打印结果
        for i in self.thing_class:
            logger.info('Class_{}  AP{}: {:.3f}'.format(self._class_names[i], int(iou_thresh * 100), ap_list[i] * 100))
            
        logger.info('AP{}: {:.3f}'.format(int(iou_thresh * 100), APx * 100))
        
        return ap_list, APx

    def update_bbox(self, instances, target, logger):
        """
        基于bbox计算AP
        """
        tgt_labels = target["labels"]
        tgt_bboxs = target["bboxs"]
        for tgt_label, tgt_bbox in zip(tgt_labels, tgt_bboxs):
            if tgt_label in self.ignore_labels:
                continue

            flags = np.zeros(self._num_classes, dtype=bool)
            for instance in instances:
                src_label = instance["labels"]
                src_score = instance["scores"]
                if src_label in self.ignore_labels:
                    continue
                if src_score < self.min_obj_score:
                    continue
                src_bbox = instance["bboxs"]
                
                iou = calculate_bbox_iou(src_bbox, tgt_bbox)
                
                for index, iou_thresh in enumerate(self.IoU_threshes):
                    if iou >= iou_thresh:
                        flags[index] = True
                        if tgt_label==src_label:
                            self.tp_classes_thresh[index][tgt_label] += 1
                        else:
                            self.fp_classes_thresh[index][src_label] += 1
            for index, _ in enumerate(self.IoU_threshes):
                if not flags[index]:
                    self.fn_classes_thresh[index][tgt_label] += 1

    def update(self, instances, target, lengths):
        tgt_labels = target["labels"].tolist()
        tgt_masks = target["masks"].transpose(0, 1)
        for tgt_label, tgt_mask in zip(tgt_labels, tgt_masks):
            if tgt_label in self.ignore_labels:
                continue

            flag = False
            for instance in instances:
                src_label = instance["labels"]
                src_score = instance["scores"]
                if src_label in self.ignore_labels:
                    continue
                if src_score < self.min_obj_score:
                    continue
                src_mask = instance["masks"]

                interArea = sum(lengths[np.logical_and(src_mask, tgt_mask)])
                unionArea = sum(lengths[np.logical_or(src_mask, tgt_mask)])
                iou = interArea / (unionArea + 1e-6)

                if iou >= self.IoU_thres:
                    flag = True
                    if tgt_label == src_label:
                        self.tp_classes[tgt_label] += 1
                        self.tp_classes_values[tgt_label] += iou
                    else:
                        self.fp_classes[src_label] += 1
                        
            if not flag:
                self.fn_classes[tgt_label] += 1

    def get_eval(self, logger):
        # each class
        RQ = self.tp_classes / (
            self.tp_classes + 0.5 * self.fp_classes + 0.5 * self.fn_classes + 1e-6
        )
        SQ = self.tp_classes_values / (self.tp_classes + 1e-6)
        PQ = RQ * SQ

        aps = []
        ap50 = 0.
        ap75 = 0.
        for index, iou_thresh in enumerate(self.IoU_threshes):
            ap_list, apx = self.compute_apx(self.tp_classes_thresh[index], self.fp_classes_thresh[index], self.fn_classes_thresh[index], iou_thresh, logger)
            aps.append(apx)
            # logger.info('AP{}: {:.3f}'.format(int(iou_thresh * 100), apx * 100))
            if index == 0:
                ap50 = apx * 100
            elif index == 5:
                ap75 = apx * 100
        
        assert len(aps) == 10
        mAP = sum(aps) / len(aps) * 100
        logger.info('mAP: {:.3f}'.format(mAP))

        # thing
        thing_RQ = sum(self.tp_classes[self.thing_class]) / (
            sum(self.tp_classes[self.thing_class])
            + 0.5 * sum(self.fp_classes[self.thing_class])
            + 0.5 * sum(self.fn_classes[self.thing_class])
            + 1e-6
        )
        thing_SQ = sum(self.tp_classes_values[self.thing_class]) / (
            sum(self.tp_classes[self.thing_class]) + 1e-6
        )
        thing_PQ = thing_RQ * thing_SQ

        # stuff
        stuff_RQ = sum(self.tp_classes[self.stuff_class]) / (
            sum(self.tp_classes[self.stuff_class])
            + 0.5 * sum(self.fp_classes[self.stuff_class])
            + 0.5 * sum(self.fn_classes[self.stuff_class])
            + 1e-6
        )
        stuff_SQ = sum(self.tp_classes_values[self.stuff_class]) / (
            sum(self.tp_classes[self.stuff_class]) + 1e-6
        )
        stuff_PQ = stuff_RQ * stuff_SQ

        # total
        sRQ = sum(self.tp_classes) / (
            sum(self.tp_classes)
            + 0.5 * sum(self.fp_classes)
            + 0.5 * sum(self.fn_classes)
            + 1e-6
        )
        sSQ = sum(self.tp_classes_values) / (sum(self.tp_classes) + 1e-6)
        sPQ = sRQ * sSQ

        for i in range(self._num_classes):
            logger.info(
                "Class_{}  PQ / RQ / SQ : {:.2f} & {:.2f} & {:.2f}".format(self._class_names[i], PQ[i] * 100, RQ[i] *100, SQ[i] * 100)
            )

        # for i, name in enumerate(self._class_names):
        #     logger.info('Class_{}  PQ: {:.3f}'.format(name, PQ[i]*100))

        logger.info(
            "PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}".format(
                sPQ * 100, sRQ * 100, sSQ * 100
            )
        )
        logger.info(
            "thing PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}".format(
                thing_PQ * 100, thing_RQ * 100, thing_SQ * 100
            )
        )
        logger.info(
            "stuff PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}".format(
                stuff_PQ * 100, stuff_RQ * 100, stuff_SQ * 100
            )
        )
        return {"PQ": sPQ * 100, "RQ": sRQ * 100, "SQ": sSQ * 100, "mAP": mAP, "AP50": ap50, "AP75": ap75}

class InstanceEval_speedup(InstanceEval):
    def update(self, instances, target, lengths):
        tgt_labels = target["labels"].tolist()
        tgt_masks = target["masks"].transpose(0, 1)
        matched = [False] * len(instances)
        for tgt_label, tgt_mask in zip(tgt_labels, tgt_masks):
            if tgt_label in self.ignore_labels:
                continue

            flag = False
            for i, instance in enumerate(instances):
                if matched[i]:
                    continue
                src_label = instance["labels"]
                src_score = instance["scores"]
                if src_label in self.ignore_labels:
                    continue
                if src_score < self.min_obj_score:
                    continue
                src_mask = instance["masks"]

                interArea = sum(lengths[np.logical_and(src_mask, tgt_mask)])
                unionArea = sum(lengths[np.logical_or(src_mask, tgt_mask)])
                iou = interArea / (unionArea + 1e-6)

                if iou >= self.IoU_thres:
                    flag = True
                    if tgt_label == src_label:
                        self.tp_classes[tgt_label] += 1
                        self.tp_classes_values[tgt_label] += iou
                    else:
                        self.fp_classes[src_label] += 1
                    matched[i] = True
                    break
                        
            if not flag:
                self.fn_classes[tgt_label] += 1

class InstanceEval2(InstanceEval):
    """
    修改match规则
    """
    def __init__(self, classes, num_classes=35, ignore_labels=[35, 36]) -> None:
        super().__init__(classes, num_classes, ignore_labels)

    def update(self, instances, target, lengths):
        # lengths = np.round(np.log(1 + lengths), 3)
        tgt_labels = target["labels"].tolist()
        tgt_masks = target["masks"].transpose(0, 1)
        for tgt_label, tgt_mask in zip(tgt_labels, tgt_masks):
            if tgt_label in self.ignore_labels:
                continue

            tp_classes_values_list = [[] for _ in range(self._num_classes)]
            flags = np.zeros(self._num_classes, dtype=bool)
            flag = False
            for instance in instances:
                src_label = instance["labels"]
                src_score = instance["scores"]
                if src_label in self.ignore_labels:
                    continue
                if src_score < self.min_obj_score:
                    continue
                src_mask = instance["masks"]

                interArea = sum(lengths[np.logical_and(src_mask, tgt_mask)])
                unionArea = sum(lengths[np.logical_or(src_mask, tgt_mask)])
                iou = interArea / (unionArea + 1e-6)
                
                for index, iou_thresh in enumerate(self.IoU_threshes):
                    if iou >= iou_thresh:
                        flags[index] = True
                        if tgt_label==src_label:
                            self.tp_classes_thresh[index][tgt_label] += 1
                        else:
                            self.fp_classes_thresh[index][src_label] += 1

                if iou >= self.IoU_thres:
                    flag = True
                    if tgt_label == src_label:
                        # self.tp_classes[tgt_label] += 1
                        # self.tp_classes_values[tgt_label] += iou
                        tp_classes_values_list[tgt_label].append(iou)
                    else:
                        self.fp_classes[src_label] += 1
                        
            for index, _ in enumerate(self.IoU_threshes):
                if not flags[index]:
                    self.fn_classes_thresh[index][tgt_label] += 1
            if not flag:
                self.fn_classes[tgt_label] += 1
            if len(tp_classes_values_list[tgt_label]) > 0:
                self.tp_classes[tgt_label] += 1
                self.tp_classes_values[tgt_label] += max(tp_classes_values_list[tgt_label])


def sympoint_semantic_F1_eval(
    gt_labels, seg_preds, ele_weights, classes, ignore_index, logger=None
):
    pointWiseEval = PointWiseEvalF1(classes, ignore_labels=ignore_index)
    for gt, prd, weights in zip(gt_labels, seg_preds, ele_weights):
        assert gt.shape == prd.shape == weights.shape
        pointWiseEval.update(gt, prd, weights)
    res = pointWiseEval.get_eval(logger)
    return res


def sympoint_semantic_eval(gt_labels, seg_preds, classes, ignore_index, logger=None):
    pointWiseEval = PointWiseEval(classes, ignore_labels=ignore_index)
    for gt, prd in zip(gt_labels, seg_preds):
        assert gt.shape == prd.shape
        pointWiseEval.update(gt, prd)
    res = pointWiseEval.get_eval(logger)
    return res


def calculate_bbox_iou(bbox1, bbox2):
    """
    计算两个边界框之间的 IoU。

    参数:
    bbox1 (tuple): 第一个边界框，格式为 (x1, y1, x2, y2)。
    bbox2 (tuple): 第二个边界框，格式为 (x1, y1, x2, y2)。

    返回:
    float: 两个边界框之间的 IoU。
    """
    # 解包边界框坐标
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集的左上角和右下角坐标
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 计算交集面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # 计算两个边界框的面积
    area_bbox1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_bbox2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集面积
    union_area = area_bbox1 + area_bbox2 - inter_area
    
    # 计算 IoU
    iou = inter_area / union_area if union_area != 0 else 0
    
    return iou


def sympoint_instance_eval(
    gt_semantic_masks_inst_task,
    gt_instance_masks_inst_task,
    pred_instance_masks_inst_task,
    gt_semantic_masks_inst_bbox_task,
    gt_instance_masks_inst_bbox_task,
    pred_instance_masks_inst_bbox_task,
    pred_instance_labels,
    pred_instance_scores,
    ele_weights,
    classes,
    ignore_index,
    logger,
):
    # insEval = InstanceEval2(classes, ignore_labels=ignore_index)
    insEval = InstanceEval_speedup(classes, ignore_labels=ignore_index)
    """
    instances: [instance1, instance2, ..., instance_n]
        instance_x: {
            "masks": [True, False, True],
            "labels": 1,
            "scores": 0.3,
        }
    
    target:
        "labels": [1, 2, ...3] list[sem_id]
        "masks": [[001100110], [111000011], [000011001]]
    """
    mask_overlap_rate_arr = []
    for (
        gt_sem,
        gt_ins,
        pred_ins_masks,
        gt_sem_bbox_labels,
        gt_ins_bboxs,
        pred_ins_bboxs,
        pred_ins_labels,
        pred_ins_scores,
        lengths,
    ) in zip(
        gt_semantic_masks_inst_task,
        gt_instance_masks_inst_task,
        pred_instance_masks_inst_task,
        gt_semantic_masks_inst_bbox_task,
        gt_instance_masks_inst_bbox_task,
        pred_instance_masks_inst_bbox_task,
        pred_instance_labels,
        pred_instance_scores,
        ele_weights,
    ):
        # prepare target
        unique_ins_ids = np.unique(gt_ins)
        target = {
            'labels': [],
            'masks': []
        }
        for ins_id in unique_ins_ids:
            mask = (gt_ins == ins_id)
            label = np.unique(gt_sem[mask])
            assert len(label) == 1
            label = label[0].tolist()
            target['labels'].append(label)
            target['masks'].append(mask)

        target['labels'] = np.array(target['labels'])
        target['masks'] = np.stack(target['masks']).astype(np.uint8)
        
        # prepare bbox target
        bbox_target = {}
        bbox_target["labels"] = gt_sem_bbox_labels
        bbox_target["bboxs"] = gt_ins_bboxs

        # prepare instances
        instances = []
        bbox_instances = []
        # check mask
        if len(pred_ins_masks) == 0:
            continue
        check_mask = np.zeros_like(pred_ins_masks[0])
        mask_overlap_count = 0
        for mask, label, score, bbox in zip(pred_ins_masks, pred_ins_labels, pred_ins_scores, pred_ins_bboxs):
            mask_ = mask.cpu().numpy()
            if np.sum(check_mask[mask_]) > 0:
                mask_overlap_count += 1
            check_mask[mask_] = 1
            instance = {
                "masks": mask.cpu().numpy(),
                "labels": label.cpu().numpy(),
                "scores": score.cpu().numpy(),
            }
            instances.append(instance)
            
            # prepare bbox instances
            instance_ = {
                "bboxs": bbox,
                "labels": label.cpu().numpy().tolist(),
                "scores": score.cpu().numpy().tolist(),
            }
            bbox_instances.append(instance_)
        # calculate mask overlap rate
        rate = mask_overlap_count / len(pred_ins_masks)
        mask_overlap_rate_arr.append(rate)
        insEval.update(instances, target, lengths)
        insEval.update_bbox(bbox_instances, bbox_target, logger)
        
    print('Mean mask overlap rate: %.2f' % np.array(mask_overlap_rate_arr).mean())
    res = insEval.get_eval(logger)
    return res


def sympoint_instance_inference_eval(
    gt_semantic_masks_inst_task,
    gt_instance_masks_inst_task,
    pred_instances_list,
    ele_weights,
    classes,
    ignore_index,
    logger,
):
    """
    Using sympoint instance inference function.
    
    pred_instances: [instance1, instance2, ..., instance_n]

    target:
        "labels": [1, 2, ...3] list[sem_id]
        "masks": [[001100110], [111000011], [000011001]]
    """
    insEval = InstanceEval(classes, ignore_labels=ignore_index)
    for (
        gt_sem,
        gt_ins,
        instances,
        lengths,
    ) in zip(
        gt_semantic_masks_inst_task,
        gt_instance_masks_inst_task,
        pred_instances_list,
        ele_weights,
    ):
        # prepare target
        unique_ins_ids = np.unique(gt_ins)
        target = {
            'labels': [],
            'masks': []
        }
        for ins_id in unique_ins_ids:
            mask = (gt_ins == ins_id)
            label = gt_sem[mask][0]
            target['labels'].append(label)
            target['masks'].append(mask)

        target['labels'] = np.array(target['labels'])
        target['masks'] = np.stack(target['masks']).astype(np.uint8)

        # prepare instances
        insEval.update(instances, target, lengths)
    res = insEval.get_eval(logger)
    return res

def sympoint_instance_eval_new(
    gt_semantic_masks_inst_task,
    gt_instance_masks_inst_task,
    pred_instance_masks_inst_task,
    pred_instance_labels,
    pred_instance_scores,
    ele_weights,
    classes,
    ignore_index,
    logger,
):
    insEval = InstanceEval(classes, ignore_labels=ignore_index)
    """
    instances: [instance1, instance2, ..., instance_n]
        instance_x: {
            "masks": [True, False, True],
            "labels": 1,
            "scores": 0.3,
        }
    
    target:
        "labels": [1, 2, ...3] list[sem_id]
        "masks": [[001100110], [111000011], [000011001]]
    """
    for (
        gt_sem,
        gt_ins,
        pred_ins_masks,
        pred_ins_labels,
        pred_ins_scores,
        lengths,
    ) in zip(
        gt_semantic_masks_inst_task,
        gt_instance_masks_inst_task,
        pred_instance_masks_inst_task,
        pred_instance_labels,
        pred_instance_scores,
        ele_weights,
    ):
        # prepare target
        unique_ins_ids = np.unique(gt_ins)
        target = {
            'labels': [],
            'masks': []
        }
        for ins_id in unique_ins_ids:
            mask = (gt_ins == ins_id)
            label = gt_sem[mask][0]
            target['labels'].append(label)
            target['masks'].append(mask)

        target['labels'] = np.array(target['labels'])
        target['masks'] = np.stack(target['masks']).astype(np.uint8)

        # org code
        # target['masks'] = []
        # for sem_id, ins_id in zip(gt_sem, gt_ins):
        #     target_mask = np.zeros(len(gt_ins))
        #     ind1 = np.where(gt_sem==sem_id)[0]
        #     ind2 = np.where(gt_ins==ins_id)[0]
        #     ind = list(set(ind1).intersection(ind2))
        #     target_mask[ind] = 1
        #     target['masks'].append(target_mask)
        # target['masks'] = np.array(target['masks'])

        # prepare instances
        instances = []
        for mask, label, score in zip(pred_ins_masks, pred_ins_labels, pred_ins_scores):
            instance = {
                "masks": mask.cpu().numpy(),
                "labels": label.cpu().numpy(),
                "scores": score.cpu().numpy(),
            }
            instances.append(instance)

        # start_time = time.perf_counter()
        insEval.update(instances, target, lengths)
        
        # """
        #     统计stuff类的预测结果
        # """
        # pred_ins_labels = [item.cpu().numpy() for item in pred_ins_labels]
        # pred_ins_labels = [item for item in pred_ins_labels if item >= 32]
        # values, counts = np.unique(pred_ins_labels, return_counts=True)
        # logger.info(str(values) + "---" + str(counts))
        
        # end_time = time.perf_counter()
        # logger.info("updating an insEval needs " + str(end_time - start_time) + "s")
    res = insEval.get_eval(logger)
    return res

def sympoint_instance_eval_delete(
    gt_semantic_masks_inst_task,
    gt_instance_masks_inst_task,
    pred_instance_masks_inst_task,
    pred_instance_labels,
    pred_instance_scores,
    ele_weights,
    classes,
    ignore_index,
    logger,
):
    insEval = InstanceEval_delete(classes, ignore_labels=ignore_index)
    """
    instances: [instance1, instance2, ..., instance_n]
        instance_x: {
            "masks": [True, False, True],
            "labels": 1,
            "scores": 0.3,
        }
    
    target:
        "labels": [1, 2, ...3] list[sem_id]
        "masks": [[001100110], [111000011], [000011001]]
    """
    for (
        gt_sem,
        gt_ins,
        pred_ins_masks,
        pred_ins_labels,
        pred_ins_scores,
        lengths,
    ) in zip(
        gt_semantic_masks_inst_task,
        gt_instance_masks_inst_task,
        pred_instance_masks_inst_task,
        pred_instance_labels,
        pred_instance_scores,
        ele_weights,
    ):
        # prepare target
        unique_ins_ids = np.unique(gt_ins)
        target = {
            'labels': [],
            'masks': []
        }
        for ins_id in unique_ins_ids:
            mask = (gt_ins == ins_id)
            label = gt_sem[mask][0]
            target['labels'].append(label)
            target['masks'].append(mask)

        target['labels'] = np.array(target['labels'])
        target['masks'] = np.stack(target['masks']).astype(np.uint8)

        # org code
        # target['masks'] = []
        # for sem_id, ins_id in zip(gt_sem, gt_ins):
        #     target_mask = np.zeros(len(gt_ins))
        #     ind1 = np.where(gt_sem==sem_id)[0]
        #     ind2 = np.where(gt_ins==ins_id)[0]
        #     ind = list(set(ind1).intersection(ind2))
        #     target_mask[ind] = 1
        #     target['masks'].append(target_mask)
        # target['masks'] = np.array(target['masks'])

        # prepare instances
        instances = []
        for mask, label, score in zip(pred_ins_masks, pred_ins_labels, pred_ins_scores):
            instance = {
                "masks": mask.cpu().numpy(),
                "labels": label.cpu().numpy(),
                "scores": score.cpu().numpy(),
            }
            instances.append(instance)

        # start_time = time.perf_counter()
        insEval.update(instances, target, lengths)
        
        # """
        #     统计stuff类的预测结果
        # """
        # pred_ins_labels = [item.cpu().numpy() for item in pred_ins_labels]
        # pred_ins_labels = [item for item in pred_ins_labels if item >= 32]
        # values, counts = np.unique(pred_ins_labels, return_counts=True)
        # logger.info(str(values) + "---" + str(counts))
        
        # end_time = time.perf_counter()
        # logger.info("updating an insEval needs " + str(end_time - start_time) + "s")
    res = insEval.get_eval(logger)
    return res
def sympoint_instance_eval_delete_1(
    gt_semantic_masks_inst_task,
    gt_instance_masks_inst_task,
    pred_instance_masks_inst_task,
    pred_instance_labels,
    pred_instance_scores,
    ele_weights,
    classes,
    ignore_index,
    logger,
):
    insEval = InstanceEval_delete_1(classes, ignore_labels=ignore_index)
    """
    instances: [instance1, instance2, ..., instance_n]
        instance_x: {
            "masks": [True, False, True],
            "labels": 1,
            "scores": 0.3,
        }
    
    target:
        "labels": [1, 2, ...3] list[sem_id]
        "masks": [[001100110], [111000011], [000011001]]
    """
    for (
        gt_sem,
        gt_ins,
        pred_ins_masks,
        pred_ins_labels,
        pred_ins_scores,
        lengths,
    ) in zip(
        gt_semantic_masks_inst_task,
        gt_instance_masks_inst_task,
        pred_instance_masks_inst_task,
        pred_instance_labels,
        pred_instance_scores,
        ele_weights,
    ):
        # prepare target
        unique_ins_ids = np.unique(gt_ins)
        target = {
            'labels': [],
            'masks': []
        }
        for ins_id in unique_ins_ids:
            mask = (gt_ins == ins_id)
            label = gt_sem[mask][0]
            target['labels'].append(label)
            target['masks'].append(mask)

        target['labels'] = np.array(target['labels'])
        target['masks'] = np.stack(target['masks']).astype(np.uint8)

        # org code
        # target['masks'] = []
        # for sem_id, ins_id in zip(gt_sem, gt_ins):
        #     target_mask = np.zeros(len(gt_ins))
        #     ind1 = np.where(gt_sem==sem_id)[0]
        #     ind2 = np.where(gt_ins==ins_id)[0]
        #     ind = list(set(ind1).intersection(ind2))
        #     target_mask[ind] = 1
        #     target['masks'].append(target_mask)
        # target['masks'] = np.array(target['masks'])

        # prepare instances
        instances = []
        for mask, label, score in zip(pred_ins_masks, pred_ins_labels, pred_ins_scores):
            instance = {
                "masks": mask.cpu().numpy(),
                "labels": label.cpu().numpy(),
                "scores": score.cpu().numpy(),
            }
            instances.append(instance)

        # start_time = time.perf_counter()
        insEval.update(instances, target, lengths)
        
        # """
        #     统计stuff类的预测结果
        # """
        # pred_ins_labels = [item.cpu().numpy() for item in pred_ins_labels]
        # pred_ins_labels = [item for item in pred_ins_labels if item >= 32]
        # values, counts = np.unique(pred_ins_labels, return_counts=True)
        # logger.info(str(values) + "---" + str(counts))
        
        # end_time = time.perf_counter()
        # logger.info("updating an insEval needs " + str(end_time - start_time) + "s")
    res = insEval.get_eval(logger)
    return res
class InstanceEval_delete(InstanceEval):
    def update(self, instances, target, lengths):
        # lengths = np.round(np.log(1 + lengths), 3)
        tgt_labels = target["labels"].tolist()
        tgt_masks = target["masks"].transpose(0, 1)
        
        # 创建一个标记列表，记录哪些预测实例已被匹配
        matched = [False] * len(instances)

        for tgt_label, tgt_mask in zip(tgt_labels, tgt_masks):
            if tgt_label in self.ignore_labels:
                continue

            flags = np.zeros(len(self.IoU_threshes), dtype=bool)
            flag = False

            # 遍历预测实例
            for i, instance in enumerate(instances):
                if matched[i]:
                    continue

                src_label = instance["labels"]
                src_score = instance["scores"]
                if src_label in self.ignore_labels:
                    continue
                if src_score < self.min_obj_score:
                    continue
                src_mask = instance["masks"]

                interArea = sum(lengths[np.logical_and(src_mask, tgt_mask)])
                unionArea = sum(lengths[np.logical_or(src_mask, tgt_mask)])
                iou = interArea / (unionArea + 1e-6)
                
                for index, iou_thresh in enumerate(self.IoU_threshes):
                    if iou >= iou_thresh:
                        flags[index] = True
                        if tgt_label == src_label:
                            self.tp_classes_thresh[index][tgt_label] += 1
                        else:
                            self.fp_classes_thresh[index][src_label] += 1

                if iou >= self.IoU_thres:
                    flag = True
                    if tgt_label == src_label:
                        self.tp_classes[tgt_label] += 1
                        self.tp_classes_values[tgt_label] += iou
                    else:
                        self.fp_classes[src_label] += 1
                    
                    matched[i] ==True

            for index, _ in enumerate(self.IoU_threshes):
                if not flags[index]:
                    self.fn_classes_thresh[index][tgt_label] += 1
            if not flag:
                self.fn_classes[tgt_label] += 1

class InstanceEval_delete_1(InstanceEval):
    def update(self, instances, target, lengths):
        # lengths = np.round(np.log(1 + lengths), 3)
        tgt_labels = target["labels"].tolist()
        tgt_masks = target["masks"].transpose(0, 1)
        
        # 创建一个标记列表，记录哪些预测实例已被匹配
        matched = [False] * len(instances)

        for tgt_label, tgt_mask in zip(tgt_labels, tgt_masks):
            if tgt_label in self.ignore_labels:
                continue

            flags = np.zeros(len(self.IoU_threshes), dtype=bool)
            flag = False

            # 遍历预测实例
            for i, instance in enumerate(instances):
                if matched[i]:
                    continue  

                src_label = instance["labels"]
                src_score = instance["scores"]
                if src_label in self.ignore_labels:
                    continue
                if src_score < self.min_obj_score:
                    continue
                src_mask = instance["masks"]

                interArea = sum(lengths[np.logical_and(src_mask, tgt_mask)])
                unionArea = sum(lengths[np.logical_or(src_mask, tgt_mask)])
                iou = interArea / (unionArea + 1e-6)
                
                for index, iou_thresh in enumerate(self.IoU_threshes):
                    if iou >= iou_thresh:
                        flags[index] = True
                        if tgt_label == src_label:
                            self.tp_classes_thresh[index][tgt_label] += 1
                        else:
                            self.fp_classes_thresh[index][src_label] += 1

                if iou >= self.IoU_thres:
                    flag = True
                    if tgt_label == src_label:
                        self.tp_classes[tgt_label] += 1
                        self.tp_classes_values[tgt_label] += iou
                    else:
                        self.fp_classes[src_label] += 1
                    
                    matched[i] = True
                    break

            for index, _ in enumerate(self.IoU_threshes):
                if not flags[index]:
                    self.fn_classes_thresh[index][tgt_label] += 1
            if not flag:
                self.fn_classes[tgt_label] += 1