"""Tune Week 4 models using real RELIANCE.NS historical_candles data."""

import json

import pandas as pd
from sklearn.base import clone

from app.database.models import HistoricalCandle, Stock
from app.database.session import SessionLocal
from app.features.technical import add_technical_indicators
from app.features.target_generator import add_forward_targets
from app.ml.datasets import get_feature_columns
from app.ml.models.xgboost_model import build_xgboost_regressor
from app.ml.optimization import (
    OUTPUT_DIR,
    tune_classifier,
    tune_regressor,
    tuning_configurations,
    walk_forward_classification,
    walk_forward_regression,
)


def load_reliance() -> pd.DataFrame:
    with SessionLocal() as db:
        stock = db.query(Stock).filter(Stock.symbol == "RELIANCE.NS").first()
        if stock is None:
            raise ValueError("RELIANCE.NS not found")
        candles = db.query(HistoricalCandle).filter(
            HistoricalCandle.stock_id == stock.id,
            HistoricalCandle.timeframe == "D",
        ).order_by(HistoricalCandle.timestamp).all()

    df = pd.DataFrame([{
        "Date": c.timestamp,
        "Open": c.open,
        "High": c.high,
        "Low": c.low,
        "Close": c.close,
        "Volume": c.volume,
    } for c in candles]).set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    return df.sort_index()


def main() -> None:
    df = load_reliance()
    df = add_technical_indicators(df, "daily")
    df = add_forward_targets(df, "daily")

    X = df[get_feature_columns(df)].replace([float("inf"), float("-inf")], pd.NA)
    X = X.select_dtypes(include=["number"]).astype(float).dropna()
    y = df.loc[X.index, "target_1d"]
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    return_y = df.loc[X.index, "forward_return_1d"]

    report = {"source": "historical_candles:RELIANCE.NS", "rows": len(X), "features": len(X.columns)}
    initial_train = min(900, len(X) // 2)

    for name, (model, grid, encoded) in tuning_configurations().items():
        tuned = tune_classifier(model, grid, X, y, cv_splits=4, encoded_target=encoded)
        report[name] = {
            "best_params": tuned["params"],
            "cv_balanced_accuracy": tuned["score"],
            "walk_forward": walk_forward_classification(
                X, y, lambda config=tuned["model"]: clone(config),
                encoded_target=encoded, initial_train_size=initial_train, test_size=126,
            ),
        }

    tuned_regressor = tune_regressor(
        build_xgboost_regressor(),
        {"n_estimators": [200, 300], "max_depth": [3, 5],
         "learning_rate": [0.03, 0.05], "subsample": [0.85]},
        X, return_y, cv_splits=4,
    )
    report["xgboost_regressor"] = {
        "best_params": tuned_regressor["params"],
        "cv_neg_mae": tuned_regressor["score"],
        "walk_forward": walk_forward_regression(
            X, return_y, lambda config=tuned_regressor["model"]: clone(config),
            initial_train_size=initial_train, test_size=126,
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "historical_RELIANCE_NS.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
