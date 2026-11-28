import numpy as np
import pandas as pd

from app.ml.evaluate import market_regime_metrics, walk_forward_validation
from app.ml.models.random_forest import build_random_forest_classifier


def test_walk_forward_validation():
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    X = pd.DataFrame({"feature": np.linspace(0, 1, 100)}, index=index)
    y = pd.Series(np.resize([-1, 0, 1], 100), index=index)

    result = walk_forward_validation(
        X, y, build_random_forest_classifier,
        initial_train_size=70, test_size=10,
        target_type="classification",
    )

    assert "aggregate" in result
    assert len(result["folds"]) == 3
    assert "accuracy" in result["aggregate"]


def test_market_regime_metrics():
    rows = 100
    y_true = np.resize([1, -1, 0], rows)
    y_pred = np.resize([1, -1, 1, 0], rows)
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, rows)

    regimes = market_regime_metrics(y_true, y_pred, returns)

    assert {"low_volatility", "normal_volatility", "high_volatility"}.issubset(regimes)
