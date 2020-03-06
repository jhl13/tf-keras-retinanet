# -*- coding:utf-8 -*-
from utils.compute_overlap import compute_overlap
import numpy as np


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    recall = np.asarray(recall)
    precision = np.asarray(precision)
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # 对precision进行排序，第i-1个位置的precision是[i-1，i]precision的最大值，i只会到1
    for i in range(mpre.size - 1, 0, -1): 
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value # 找出recall变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec # 计算PR曲线的面积
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
