from __future__ import annotations

import math
from typing import Any

import numpy as np


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.float32).reshape(-1)
    y_prob = np.clip(y_prob.astype(np.float32).reshape(-1), 1e-6, 1.0 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bce = -float(np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))

    return {
        "binary_cross_entropy": bce,
        "accuracy": float(np.mean(y_pred == y_true)),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "calibration": calibration_bins(y_true, y_prob),
    }


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    scores = np.concatenate([positives, negatives])
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    pos_ranks = ranks[labels == 1].sum()
    auc = (pos_ranks - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives))
    if math.isnan(auc):
        return None
    return float(auc)


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    output = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < bins - 1 else y_prob <= high)
        if not np.any(mask):
            continue
        output.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": int(np.sum(mask)),
                "mean_predicted_probability": float(np.mean(y_prob[mask])),
                "observed_win_rate": float(np.mean(y_true[mask])),
            }
        )
    return output
