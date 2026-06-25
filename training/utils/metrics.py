"""Small metric helpers."""

from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    truth_bin = truth.astype(bool)
    denom = int(pred_bin.sum() + truth_bin.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred_bin, truth_bin).sum() / denom)


def iou_score(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    truth_bin = truth.astype(bool)
    union = int(np.logical_or(pred_bin, truth_bin).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_bin, truth_bin).sum() / union)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)

