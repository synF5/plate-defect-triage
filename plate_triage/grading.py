"""Metrics used in both training and the app."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def one_vs_rest_auc(y_true, probabilities: np.ndarray, classes) -> float:
    # macro one-vs-rest AUC. Done per class because sklearn's version errors
    # if an uploaded file doesn't have all 7 classes. Same result on the full test set.
    y_true = np.asarray(y_true)
    per_class = []
    for column, label in enumerate(classes):
        is_label = y_true == label
        if 0 < is_label.sum() < len(y_true):
            per_class.append(roc_auc_score(is_label, probabilities[:, column]))
    return float(np.mean(per_class)) if per_class else float("nan")


def grade(y_true, y_pred, probabilities: np.ndarray, classes) -> dict[str, float]:
    # macro averages so the small classes count as much as the big ones
    shared = {"average": "macro", "zero_division": 0}
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": one_vs_rest_auc(y_true, probabilities, classes),
        "Precision": float(precision_score(y_true, y_pred, **shared)),
        "Recall": float(recall_score(y_true, y_pred, **shared)),
        "F1": float(f1_score(y_true, y_pred, **shared)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def mcc_interval(y_true, y_pred, rounds: int = 2000, seed: int = 5781) -> tuple[float, float]:
    # 95% bootstrap interval for MCC
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    draws = [
        matthews_corrcoef(y_true[idx], y_pred[idx])
        for idx in rng.integers(0, len(y_true), size=(rounds, len(y_true)))
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)
