"""Machine learning training workflows."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.datasets import chronological_split, feature_combinations, prepare_dataset
from app.ml.evaluate import evaluate_classifier, evaluate_regressor
from app.ml.models.random_forest import build_random_forest_classifier
from app.ml.models.xgboost_model import build_xgboost_classifier, build_xgboost_regressor


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def logistic_regression_model() -> Pipeline:
    """Build a scaled Logistic Regression baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])


def train_classification_experiment(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Compare baseline and ensemble classifiers across feature combinations."""
    results: dict[str, Any] = {}

    for feature_name, features in feature_combinations(df).items():
        X, y = prepare_dataset(df, target_column, features)
        X_train, X_test, y_train, y_test = chronological_split(X, y, test_size)

        if set(y_train) - {-1, 0, 1}:
            raise ValueError("Classification target must contain only -1, 0, and 1")

        models = {
            "logistic_regression": logistic_regression_model(),
            "random_forest": build_random_forest_classifier(),
            "xgboost": build_xgboost_classifier(),
        }

        for model_name, model in models.items():
            if model_name == "xgboost":
                model.fit(X_train, y_train.map({-1: 0, 0: 1, 1: 2}))
                predictions = pd.Series(model.predict(X_test)).map({0: -1, 1: 0, 2: 1}).to_numpy()
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            metrics = evaluate_classifier(y_test, predictions)

            key = f"{feature_name}/{model_name}"
            results[key] = metrics

            model_dir = ARTIFACT_DIR / "classification" / target_column
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / f"{feature_name}_{model_name}.joblib")
            pd.Series(features).to_json(model_dir / f"{feature_name}_features.json", orient="values")

            logger.info("%s: %s", key, metrics)

    output_path = ARTIFACT_DIR / "classification" / target_column / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def train_regression_experiment(
    df: pd.DataFrame,
    return_column: str,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train XGBoost regressors for forward returns across feature combinations."""
    results: dict[str, Any] = {}

    for feature_name, features in feature_combinations(df).items():
        X, y = prepare_dataset(df, return_column, features)
        X_train, X_test, y_train, y_test = chronological_split(X, y, test_size)

        model = build_xgboost_regressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_regressor(y_test, predictions)
        results[feature_name] = metrics

        model_dir = ARTIFACT_DIR / "regression" / return_column
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / f"{feature_name}_xgboost.joblib")
        pd.Series(features).to_json(model_dir / f"{feature_name}_features.json", orient="values")
        logger.info("%s: %s", feature_name, metrics)

    output_path = ARTIFACT_DIR / "regression" / return_column / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
