import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment


class UnionFind:
    """Used by 3D_segmentation to merge touching instances across xy block boundaries."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


#========================================================================================
# Metrics
#========================================================================================

@njit(cache=True)
def _label_overlap(x, y):
    """Pixel overlap matrix between two integer label arrays."""
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint64)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    overlap       = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou           = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    return (iou[true_ind, pred_ind] >= th).sum()

def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """IoU-based average precision.

    Returns (ap, tp, fp, fn) each of shape [N, T] for N images and T thresholds.
    AP = TP / (TP + FP + FN)  per image per threshold.
    """
    if not isinstance(masks_true, list):
        masks_true, masks_pred = [masks_true], [masks_pred]
    if not isinstance(threshold, (list, np.ndarray)):
        threshold = [threshold]
    N, T = len(masks_true), len(threshold)
    ap = np.zeros((N, T), np.float32)
    tp = np.zeros((N, T), np.float32)
    fp = np.zeros((N, T), np.float32)
    fn = np.zeros((N, T), np.float32)
    n_true = np.array([m.max() for m in masks_true])
    n_pred = np.array([m.max() for m in masks_pred])
    for n in range(N):
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])
    return ap, tp, fp, fn
