"""Probability calibration and model interpretation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


def calibrate_classifier(model, X_train, y_train, method: str = "sigmoid"):
    """Fit a calibrated copy of a probability-capable classifier."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be sigmoid or isotonic")

    calibrated = CalibratedClassifierCV(model, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Return normalized feature importance for tree or linear classifiers."""
    estimator = getattr(model, "model", model)

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    result = pd.DataFrame({"feature": feature_names, "importance": values})
    result = result.sort_values("importance", ascending=False).reset_index(drop=True)
    total = result["importance"].sum()
    if total:
        result["importance"] /= total
    return result


def try_shap_importance(model, X) -> pd.DataFrame:
    """Calculate SHAP importance when shap is installed; otherwise return empty data."""
    try:
        import shap
    except ImportError:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    estimator = getattr(model, "model", model)
    explainer = shap.Explainer(estimator)
    shap_values = explainer(X)
    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        importance = np.mean(np.abs(values), axis=(0, 2))
    else:
        importance = np.mean(np.abs(values), axis=0)

    return pd.DataFrame({
        "feature": list(X.columns),
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
