# This is a copy from mmdet/models/layers/matrix_nms.py.
# We just change the input shape of `masks` tensor.
import gc

import torch
import cupy

numpy = cupy
sparse = cupy.sparse
device = "cuda:0"


def sort_cpu(*args, **kwargs):
    args = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
    kwargs = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for (k, v) in kwargs.items()
    }
    rets = torch.sort(*args, **kwargs)
    return tuple([ret.cuda() if isinstance(ret, torch.Tensor) else ret for ret in rets])


def decay_coeff(
    flatten_masks,
    mask_area,
    labels,
    masks,
    eps=1e-6,
    kernel="linear",
):
    assert kernel == "linear", "only linear kernel implemented"

    decay_data, decay_rows, decay_cols = (
        numpy.empty(0, numpy.float32),
        numpy.empty(0, numpy.int64),
        numpy.empty(0, numpy.int64),
    )

    num_masks = len(flatten_masks)
    indices_all = torch.arange(num_masks, device=masks.device)

    labels_unq = torch.unique(labels).cpu()

    len_decay = 0

    # NOTE
    # miou between masks are only computed for pairs with same semantic label
    # while miou's are first computed then masked out by label_matrix
    # in vanilla implementation
    for ll in labels_unq:
        mask_ll = labels == ll

        flatten_masks_ll = sparse.csr_matrix(numpy.asarray(flatten_masks[mask_ll]))
        inter_ll = sparse.triu(flatten_masks_ll @ flatten_masks_ll.T, k=1)
        len_ll = inter_ll.nnz
        if len_ll == 0:
            continue

        mask_area_ll = numpy.asarray(mask_area[mask_ll])
        iou_ll = inter_ll.data / (
            mask_area_ll[inter_ll.row] + mask_area_ll[inter_ll.col] - inter_ll.data
        )

        indices_ll = numpy.asarray(indices_all[mask_ll])
        rows_decay_ll = indices_ll[inter_ll.row]
        cols_decay_ll = indices_ll[inter_ll.col]

        len_decay += len_ll
        decay_data = numpy.resize(decay_data, len_decay)
        decay_rows = numpy.resize(decay_rows, len_decay)
        decay_cols = numpy.resize(decay_cols, len_decay)

        decay_data[-len_ll:] = numpy.clip(iou_ll, eps, 1 - eps)
        decay_rows[-len_ll:] = rows_decay_ll
        decay_cols[-len_ll:] = cols_decay_ll

    decay_coo = sparse.coo_matrix((decay_data, (decay_rows, decay_cols)))
    # MAGIC NUMBER: [0]
    # after .max(axis=0), only one row is left
    compensate_iou_dense = decay_coo.tocsc().max(axis=0).toarray()[0]
    compensate_matrix_coo_neglog = sparse.coo_matrix(
        (
            -numpy.log(1 - decay_coo.data)
            + numpy.log(1 - compensate_iou_dense[decay_coo.row]),
            (decay_coo.row, decay_coo.col),
        )
    )
    decay_coefficient = torch.as_tensor(
        numpy.exp(-compensate_matrix_coo_neglog.tocsc().max(axis=0).toarray())[0],
        device=masks.device,
    )

    return decay_coefficient


def mask_matrix_nms(
    masks,
    labels,
    scores,
    filter_thr=-1,
    nms_pre=-1,
    max_num=-1,
    kernel="gaussian",
    sigma=2.0,
    mask_area=None,
):
    """Matrix NMS for multi-class masks.

    Args:
        masks (Tensor): Has shape (num_instances, m)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, m).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return (
            scores.new_zeros(0),
            labels.new_zeros(0),
            masks.new_zeros(0, *masks.shape[-1:]),
            labels.new_zeros(0),
        )
    if mask_area is None:
        mask_area = masks.sum(1).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    decay_coefficient = decay_coeff(
        flatten_masks, mask_area, labels, masks, kernel=kernel
    )

    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return (
                scores.new_zeros(0),
                labels.new_zeros(0),
                masks.new_zeros(0, *masks.shape[-1:]),
                labels.new_zeros(0),
            )
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True, stable=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds
