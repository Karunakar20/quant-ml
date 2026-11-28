"""Resume and run the Week 4 tuning workflow."""

from __future__ import argparse

import argparse
import json

import numpy as np
import pandas as pd

from app.ml.models.xgboost_model import build_xgboost_regressor
from app.ml.optimization import (
    OUTPUT_DIR,
    tune_classifier,
    tune_regressor,
    tuning_configurations,
    walk_forward_classification,
    walk_forward_regression,
)


def synthetic_dataset(rows: int = 1292) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range("2021-01-01", periods=rows, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.00015, 0.012, rows)))
    forward = np.r_[np.diff(close) / close[:-1], 0.0]

    df = pd.DataFrame(index=index)
    df["return_1d"] = pd.Series(forward, index=index).shift(1).fillna(0.0)
    df["rsi_14"] = 50 + 20 * np.sin(np.arange(rows) / 14) + rng.normal(0, 3, rows)
    df["volatility_20"] = df["return_1d"].rolling(20).std().fillna(0.01)
    df["market_beta_20"] = rng.normal(1.0, 0.1, rows)
    df["market_correlation_20"] = np.clip(rng.normal(0.45, 0.15, rows), -1, 1)
    df["forward_return_1d"] = forward
    df["target_1d"] = np.select(
        [forward > 0.005, forward < -0.005],
        [1, -1],
        default=0,
    )
    return df


def load_data(input_path: str | None) -> pd.DataFrame:
    if not input_path:
        return synthetic_dataset()

    df = pd.read_csv(input_path)
    date_column = next((column for column in ("Date", "date", "Unnamed: 0") if column in df.columns), None)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
    df = df.replace([np.inf, -np.inf], np.nan).sort_index()
    return df


def run(input_path: str | None = None) -> dict:
    df = load_data(input_path)
    feature_columns = [
        column for column in (
            "return_1d", "rsi_14", "volatility_20",
            "market_beta_20", "market_correlation_20",
        )
        if column in df.columns
    ]
    X = df[feature_columns].dropna()
    y = df.loc[X.index, "target_1d"]

    initial_train = min(900, len(df) // 2)
    report = {"source": input_path or "synthetic_week4_dataset"}

    for name, (model, grid, encoded) in tuning_configurations().items():
        tuned = tune_classifier(model, grid, X, y, cv_splits=4, encoded_target=encoded)
        walk = walk_forward_classification(
            X,
            y,
            lambda config=tuned["model"]: config.__class__(**config.get_params()),
            encoded_target=encoded,
            initial_train_size=initial_train,
            test_size=126,
        )
        report[name] = {
            "best_params": tuned["params"],
            "cv_balanced_accuracy": tuned["score"],
            "walk_forward": walk,
        }

    if "forward_return_1d" in df.columns:
        return_y = df.loc[X.index, "forward_return_1d"]
        tuned = tune_regressor(
            build_xgboost_regressor(),
            {
                "n_estimators": [200, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.05],
                "subsample": [0.85],
            },
            X,
            return_y,
            cv_splits=4,
        )
        report["xgboost_regressor"] = {
            "best_params": tuned["params"],
            "cv_neg_mae": tuned["score"],
            "walk_forward": walk_forward_regression(
                X,
                return_y,
                lambda config=tuned["model"]: config.__class__(**config.get_params()),
                initial_train_size=initial_train,
                test_size=126,
            ),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "optimization_results.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume Week 4 tuning")
    parser.add_argument("--input", help="Processed CSV dataset")
    args = parser.parse_args()
    report = run(args.input)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
