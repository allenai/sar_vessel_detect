import math
import numpy as np
import scipy.optimize
import scipy.spatial
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn

from xview3.training.eval.coco_eval import CocoEvaluator
from xview3.training.eval.coco_utils import get_coco_api_from_dataset

"""
Source: https://github.com/pytorch/vision/blob/master/references/detection/engine.py
"""

def copy_target(target):
    out = {}
    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            v = v.clone()
        out[k] = v
    return out

@torch.no_grad()
def get_comparisons(model, data_loader, device, half=False):
    cpu_device = torch.device("cpu")
    model.eval()
    results = []

    for image, targets in data_loader:
        image = list(img.to(device) for img in image)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.cuda.amp.autocast(enabled=half):
            outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        targets = [copy_target(t) for t in targets]
        results.extend(zip(targets, outputs))

    model.train()
    return results

@torch.no_grad()
def evaluate_map(results, data_loader):
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_evaluator.update({
        target["image_id"].item(): output
        for target, output in results
    })

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator.coco_eval['bbox'].stats[0].item()

@torch.no_grad()
def evaluate_f1(results, data_loader):
    thresholds = [0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    tp = {threshold: 0 for threshold in thresholds}
    fp = {threshold: 0 for threshold in thresholds}
    fn = {threshold: 0 for threshold in thresholds}

    for target, output in results:
        gt_array = np.array([
            ((target['boxes'][i, 0] + target['boxes'][i, 2]) / 2, (target['boxes'][i, 1] + target['boxes'][i, 3]) / 2)
            for i in range(target['boxes'].shape[0])
        ])

        for threshold in thresholds:
            pred_array = np.array([
                ((output['boxes'][i, 0] + output['boxes'][i, 2]) / 2, (output['boxes'][i, 1] + output['boxes'][i, 3]) / 2)
                for i in range(output['boxes'].shape[0])
                if output['scores'][i] > threshold
            ], dtype='float32')

            if gt_array.shape[0] == 0:
                fp[threshold] += pred_array.shape[0]
                continue
            if pred_array.shape[0] == 0:
                fn[threshold] += gt_array.shape[0]
                continue

            dist_mat = scipy.spatial.distance_matrix(pred_array, gt_array, p=2)
            rows, cols = scipy.optimize.linear_sum_assignment(dist_mat)
            cur_tp = len([i for i in range(len(rows)) if dist_mat[rows[i], cols[i]] < 20])
            tp[threshold] += cur_tp
            fp[threshold] += pred_array.shape[0] - cur_tp
            fn[threshold] += gt_array.shape[0] - cur_tp

    best_f1 = 0.0
    for threshold in thresholds:
        if tp[threshold] == 0:
            continue
        precision = tp[threshold] / (tp[threshold] + fp[threshold])
        recall = tp[threshold] / (tp[threshold] + fn[threshold])
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1
