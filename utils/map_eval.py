# -*- coding:utf-8 -*-
from utils.compute_overlap import compute_overlap
import numpy as np
from tqdm import tqdm
import time
import cv2
import os

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


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections.shape = N*C*5
        N: number of images
        C: number of classes
        5: boxes + scores

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_inferences_time = [None for i in range(generator.size())]
    pbar = tqdm(total = generator.size())

    # 只使用"channel last"
    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        # 返回resize的尺寸 resized_image.shape / original_image.shape
        image, scale = generator.resize_image(image)

        # run network
        # boxes scores labels 一一对应
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        inference_time = time.time() - start

        # 把boxes还原成原来的图片的大小
        boxes /= scale

        # 大于阈值分数索引
        indices = np.where(scores[0, :] > score_threshold)[0]

        # 选取对应索引的分数
        scores = scores[0][indices]

        # argsort 是从小到大排序 
        scores_sort = np.argsort(-scores)[:max_detections]

        # 选取对应索引且重新排序的输出 （对得到的索引排序就可以了）image_boxes是二维的
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        
        # 可视化检测结果
        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue
            # 将检测框拷贝到相应的位置
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        all_inferences_time[i] = inference_time
        pbar.update(1)
    
    return all_detections, all_inferences_time

def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections.shape = N*C*4
        N: number of images
        C: number of classes
        4: boxes

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    pbar = tqdm(total = generator.size())
    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()
        pbar.update(1)
    return all_annotations

def evalution(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    all_detections, all_inferences_time = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    for label in range(generator.num_classes()):
        if not generator.has(label):
            continue
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        num_detections = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            # number of boxes * 4
            num_annotations += annotations.shape[0]
            num_detections  += detections.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    np.append(false_positives, 1)
                    np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d[:-1], axis=0), annotations)
                assigned_labels = np.argmax(overlaps, axis=1)[0]
                max_overlap = overlaps[0, assigned_labels]

                # 如果多个检测框对应某一个GT的IoU都大于阈值？ detections本来就是排过序的
                if max_overlap > iou_threshold and assigned_labels not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_labels)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        if num_annotations == 0:
            if num_detections == 0:
                average_precisions[label] = 1, 0
                continue
            else:
                average_precisions[label] = 0, 0
                continue

        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # np.cumsum([0,1,0,1,1])
        # array([0, 1, 1, 2, 3])
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)
        
        # recall = TP / number of gt
        # prescision = TP / 
        recall = true_positives / num_annotations
        prescision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        average_precision = _compute_ap(recall, prescision)
        average_precisions[label] = average_precision, num_annotations
    
    inference_time_mean = np.sum(all_inferences_time) / generator.size()
    return average_precisions, inference_time_mean
    