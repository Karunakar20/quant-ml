"""Week 4 hyperparameter tuning and walk-forward optimization workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.evaluate import evaluate_classifier, evaluate_regressor
from app.ml.models.random_forest import build_random_forest_classifier
from app.ml.models.xgboost_model import build_xgboost_classifier, build_xgboost_regressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "week4_tuning"


def tune_classifier(
    model,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 4,
    encoded_target: bool = False,
):
    """Select a classifier by balanced accuracy using chronological CV."""
    target = y.map({-1: 0, 0: 1, 1: 2}) if encoded_target else y
    best = {"score": float("-inf")}

    for params in ParameterGrid(param_grid):
        candidate = clone(model).set_params(**params)
        scores = cross_val_score(
            candidate,
            X,
            target,
            cv=TimeSeriesSplit(n_splits=cv_splits),
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        mean_score = float(scores.mean())
        if mean_score > best["score"]:
            best = {"model": candidate, "params": params, "score": mean_score}

    best["model"].fit(X, target)
    return best


def walk_forward_classification(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable,
    encoded_target: bool = False,
    initial_train_size: int = 900,
    test_size: int = 126,
) -> dict:
    """Evaluate one fitted configuration using expanding chronological folds."""
    truth: list = []
    predictions: list = []
    folds: list = []

    start = initial_train_size
    fold_number = 0
    while start < len(X):
        end = min(start + test_size, len(X))
        X_train, y_train = X.iloc[:start], y.iloc[:start]
        X_test, y_test = X.iloc[start:end], y.iloc[start:end]

        model = model_factory()
        if encoded_target:
            model.fit(X_train, y_train.map({-1: 0, 0: 1, 1: 2}))
            encoded = model.predict(X_test)
            fold_pred = pd.Series(encoded).map({0: -1, 1: 0, 2: 1}).to_numpy()
        else:
            model.fit(X_train, y_train)
            fold_pred = model.predict(X_test)

        truth.extend(y_test.tolist())
        predictions.extend(fold_pred.tolist())
        metrics = evaluate_classifier(y_test, fold_pred)
        folds.append({
            "fold": fold_number,
            "train_end": str(X.index[start - 1]),
            "test_end": str(X.index[end - 1]),
            **metrics,
        })
        fold_number += 1
        start = end

    return {"aggregate": evaluate_classifier(truth, predictions), "folds": folds}


def walk_forward_regression(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable,
    initial_train_size: int = 900,
    test_size: int = 126,
) -> dict:
    """Evaluate a tuned regressor using expanding chronological folds."""
    truth: list = []
    predictions: list = []
    folds: list = []

    start = initial_train_size
    fold_number = 0
    while start < len(X):
        end = min(start + test_size, len(X))
        model = model_factory()
        model.fit(X.iloc[:start], y.iloc[:start])
        fold_pred = model.predict(X.iloc[start:end])

        truth.extend(y.iloc[start:end].tolist())
        predictions.extend(fold_pred.tolist())
        metrics = evaluate_regressor(y.iloc[start:end], fold_pred)
        folds.append({
            "fold": fold_number,
            "train_end": str(X.index[start - 1]),
            "test_end": str(X.index[end - 1]),
            **metrics,
        })
        fold_number += 1
        start = end

    return {"aggregate": evaluate_regressor(truth, predictions), "folds": folds}



def tune_regressor(model, param_grid: dict, X: pd.DataFrame, y: pd.Series, cv_splits: int = 4):
    """Select a regressor by negative chronological MAE."""
    best = {"score": float("-inf")}
    for params in ParameterGrid(param_grid):
        candidate = clone(model).set_params(**params)
        scores = cross_val_score(
            candidate, X, y,
            cv=TimeSeriesSplit(n_splits=cv_splits),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        mean_score = float(scores.mean())
        if mean_score > best["score"]:
            best = {"model": candidate, "params": params, "score": mean_score}
    best["model"].fit(X, y)
    return best

def tuning_configurations() -> dict:
    """Return bounded grids suitable for a local Week 4 experiment."""
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)),
    ])

    return {
        "logistic_regression": (
            logistic,
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__solver": ["lbfgs", "saga"],
            },
            False,
        ),
        "random_forest": (
            build_random_forest_classifier(),
            {
                "n_estimators": [200, 300],
                "max_depth": [5, 8, 12],
                "min_samples_leaf": [3, 5, 10],
            },
            False,
        ),
        "xgboost": (
            build_xgboost_classifier(),
            {
                "n_estimators": [200, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.05],
                "subsample": [0.85],
            },
            True,
        ),
    }


def save_optimization_report(report: dict, filename: str = "optimization_results.json") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
