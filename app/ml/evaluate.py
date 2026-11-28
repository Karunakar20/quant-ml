"""Model evaluation utilities."""

from typing import Any

import numpy as np
import pandas as pd
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
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": float(r2_score(y_true_array, y_pred_array)),
        "direction_accuracy": float(np.mean(np.sign(y_true_array) == np.sign(y_pred_array))),
    }


def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    initial_train_size: int,
    test_size: int,
    target_type: str = "classification",
    y_encoding: dict | None = None,
) -> dict[str, Any]:
    """Evaluate a model using expanding-window chronological folds."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    if initial_train_size >= len(X):
        raise ValueError("initial_train_size must be smaller than the dataset")
    if test_size <= 0:
        raise ValueError("test_size must be positive")

    fold_predictions = []
    fold_truth = []
    fold_metrics = []

    start = initial_train_size
    fold = 0
    while start < len(X):
        end = min(start + test_size, len(X))
        if end <= start:
            break

        X_train, y_train = X.iloc[:start], y.iloc[:start]
        X_test, y_test = X.iloc[start:end], y.iloc[start:end]

        model = model_factory()
        if y_encoding:
            model.fit(X_train, y_train.map(y_encoding))
            encoded_predictions = model.predict(X_test)
            decode = {value: key for key, value in y_encoding.items()}
            predictions = pd.Series(encoded_predictions).map(decode).to_numpy()
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        fold_truth.extend(y_test.tolist())
        fold_predictions.extend(predictions.tolist())

        if target_type == "classification":
            metrics = evaluate_classifier(y_test, predictions)
        else:
            metrics = evaluate_regressor(y_test, predictions)
        fold_metrics.append({"fold": fold, "train_end": str(X.index[start - 1]), "test_end": str(X.index[end - 1]), **metrics})
        fold += 1
        start = end

    if not fold_metrics:
        raise ValueError("No walk-forward folds were created")

    if target_type == "classification":
        aggregate = evaluate_classifier(fold_truth, fold_predictions)
    else:
        aggregate = evaluate_regressor(fold_truth, fold_predictions)

    return {"aggregate": aggregate, "folds": fold_metrics}


def market_regime_metrics(y_true, y_pred, returns, target_type: str = "classification") -> dict[str, dict]:
    """Evaluate model performance in low-, normal-, and high-volatility regimes."""
    returns = pd.Series(np.asarray(returns)).reset_index(drop=True)
    volatility = returns.rolling(20, min_periods=20).std()
    volatility = volatility.fillna(volatility.median())

    low = volatility <= volatility.quantile(0.33)
    high = volatility >= volatility.quantile(0.67)
    normal = ~(low | high)

    regimes = {"low_volatility": low, "normal_volatility": normal, "high_volatility": high}
    results = {}
    for name, mask in regimes.items():
        if not mask.any():
            continue
        selected_true = np.asarray(y_true)[mask.to_numpy()]
        selected_pred = np.asarray(y_pred)[mask.to_numpy()]
        results[name] = (
            evaluate_classifier(selected_true, selected_pred)
            if target_type == "classification"
            else evaluate_regressor(selected_true, selected_pred)
        )
    return results
