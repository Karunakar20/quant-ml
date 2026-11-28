"""Model evaluation utilities."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)


def evaluate_classifier(y_true, y_pred) -> dict[str, Any]:
    """Calculate appropriate metrics for trend classes."""
    labels = [-1, 0, 1]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def evaluate_regressor(y_true, y_pred) -> dict[str, float]:
    """Calculate common forward-return estimation metrics."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": float(r2_score(y_true_array, y_pred_array)),
        "direction_accuracy": float(np.mean(np.sign(y_true_array) == np.sign(y_pred_array))),
    }
