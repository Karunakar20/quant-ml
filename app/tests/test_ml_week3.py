import numpy as np
import pandas as pd

from app.ml.datasets import chronological_split, feature_combinations, prepare_dataset
from app.ml.evaluate import evaluate_classifier, evaluate_regressor
from app.ml.train import logistic_regression_model


def make_ml_frame(rows=100):
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100, 120, rows)
    return pd.DataFrame({
        "Open": close,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1000,
        "return_1d": np.random.default_rng(42).normal(0, 0.01, rows),
        "rsi_14": np.random.default_rng(43).uniform(30, 70, rows),
        "market_beta_20": np.random.default_rng(44).normal(1, 0.1, rows),
        "target_1d": np.resize([-1, 0, 1], rows),
        "forward_return_1d": np.random.default_rng(45).normal(0, 0.01, rows),
    }, index=index)


def test_prepare_dataset_and_chronological_split():
    df = make_ml_frame()
    X, y = prepare_dataset(df, "target_1d", ["return_1d", "rsi_14"])
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.2)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert X_train.index.max() < X_test.index.min()


def test_feature_combinations():
    combinations = feature_combinations(make_ml_frame())
    assert "all" in combinations
    assert "technical" in combinations
    assert "relative" in combinations


def test_logistic_baseline_can_fit():
    df = make_ml_frame()
    X, y = prepare_dataset(df, "target_1d")
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    model = logistic_regression_model()
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(y_test, model.predict(X_test))
    assert set(metrics) >= {"accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix"}


def test_regression_metrics():
    metrics = evaluate_regressor([0.01, -0.02, 0.03], [0.008, -0.01, 0.02])
    assert set(metrics) >= {"mae", "rmse", "r2", "direction_accuracy"}
