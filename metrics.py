"""Grading metrics. Grades are ordinal, so agreement is measured with weighted kappa."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from blastograde import EXPANSION_CLASSES, QUALITY_CLASSES
from blastograde.data.dataset import IGNORE_INDEX


def task_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Accuracy, within one grade accuracy, macro F1 and quadratic weighted kappa."""
    keep = y_true != IGNORE_INDEX
    y_true, y_pred = y_true[keep], y_pred[keep]
    labels = list(range(n_classes))
    return {
        "n": int(keep.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "within_one_grade": float((np.abs(y_true - y_pred) <= 1).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def gardner_string(expansion: int, icm: int, te: int) -> str:
    """Compose the conventional Gardner label, for example ``4AB``."""
    grade = EXPANSION_CLASSES[expansion]
    return grade if expansion < 2 else f"{grade}{QUALITY_CLASSES[icm]}{QUALITY_CLASSES[te]}"


def is_good_quality(expansion: np.ndarray, icm: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Widely used transfer criterion: expansion 3 or more with ICM and TE of B or better."""
    return (expansion >= 2) & (icm >= 0) & (icm <= 1) & (te >= 0) & (te <= 1)
