"""Tune Week 4 models from real historical_candles database rows."""

import argparse
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


def load_candles(symbol: str, timeframe: str = "D") -> pd.DataFrame:
    """Load one symbol's candles and return an OHLCV DataFrame."""
    with SessionLocal() as db:
        stock = db.query(Stock).filter(Stock.symbol == symbol).first()
        if stock is None:
            raise ValueError(f"Symbol not found in stocks table: {symbol}")

        candles = (
            db.query(HistoricalCandle)
            .filter(
                HistoricalCandle.stock_id == stock.id,
                HistoricalCandle.timeframe == timeframe,
            )
            .order_by(HistoricalCandle.timestamp)
            .all()
        )

    if not candles:
        raise ValueError(f"No {timeframe} candles found for {symbol}")

    df = pd.DataFrame([
        {
            "Date": candle.timestamp,
            "Open": candle.open,
            "High": candle.high,
            "Low": candle.low,
            "Close": candle.close,
            "Volume": candle.volume,
        }
        for candle in candles
    ]).set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    return df.sort_index()


def run(symbol: str) -> dict:
    df = load_candles(symbol)
    df = add_technical_indicators(df, timeframe="daily")
    df = add_forward_targets(df, timeframe="daily")

    X = df[get_feature_columns(df)].replace([float("inf"), float("-inf")], pd.NA).dropna()
    y = df.loc[X.index, "target_1d"]
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    return_y = df.loc[X.index, "forward_return_1d"]
    initial_train = min(900, len(df) // 2)
    report = {"source": f"historical_candles:{symbol}"}

    for name, (model, grid, encoded) in tuning_configurations().items():
        tuned = tune_classifier(model, grid, X, y, cv_splits=4, encoded_target=encoded)
        walk = walk_forward_classification(
            X,
            y,
            lambda config=tuned["model"]: clone(config),
            encoded_target=encoded,
            initial_train_size=initial_train,
            test_size=126,
        )
        report[name] = {
            "best_params": tuned["params"],
            "cv_balanced_accuracy": tuned["score"],
            "walk_forward": walk,
        }

    tuned_regressor = tune_regressor(
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
        "best_params": tuned_regressor["params"],
        "cv_neg_mae": tuned_regressor["score"],
        "walk_forward": walk_forward_regression(
            X,
            return_y,
            lambda config=tuned_regressor["model"]: clone(config),
            initial_train_size=initial_train,
            test_size=126,
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"historical_{symbol.replace('.', '_')}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune models using historical_candles")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    args = parser.parse_args()
    report = run(args.symbol)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
