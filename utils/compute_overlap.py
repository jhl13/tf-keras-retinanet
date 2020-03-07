# -*- coding:utf-8 -*-
import numpy as np

def compute_overlap(boxes, gt):
    """
    Args
        boxes: (N, 4) ndarray of float
        gt   : (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes = np.asarray(boxes)
    gt = np.asarray(gt)
    
    # 边缘像素也要算进面积
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)

    w_overlap = np.minimum(np.expand_dims(boxes[:, 2], axis=1), gt[:, 2]) \
        - np.maximum(np.expand_dims(boxes[:, 0], 1), gt[:, 0]) + 1
    h_overlap = np.minimum(np.expand_dims(boxes[:, 3], axis=1), gt[:, 3]) \
        - np.maximum(np.expand_dims(boxes[:, 1], 1), gt[:, 1]) + 1

    w_overlap = np.maximum(w_overlap, 0)
    h_overlap = np.maximum(h_overlap, 0)

    ua = np.expand_dims((boxes[:, 2] - boxes[:, 0] + 1) * \
        (boxes[:, 3] - boxes[:, 1] + 1), axis=1) + area_gt - w_overlap * h_overlap

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = w_overlap * h_overlap

    return intersection / ua
