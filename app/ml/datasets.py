"""Machine learning dataset preparation utilities."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


EXCLUDED_FEATURES = {
    "Open", "High", "Low", "Close", "Volume", "ema_12", "ema_26",
    "ema_50", "atr_14", "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns excluding OHLCV and targets."""
    return [
        column for column in df.columns
        if column not in EXCLUDED_FEATURES
        and not column.startswith(("target_", "forward_return_"))
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def feature_combinations(df: pd.DataFrame) -> dict[str, list[str]]:
    """Create named feature subsets for model experiments."""
    all_features = get_feature_columns(df)
    technical = [
        column for column in all_features
        if column.startswith(("return_", "log_return", "volatility_", "candle_",
                              "high_low", "close_to", "rsi_", "close_ema",
                              "macd_normalized", "atr_normalized", "bb_", "volume_"))
    ]
    relative = [
        column for column in all_features
        if column.startswith(("market_", "relative_return", "sector_"))
    ]

    combinations = {"all": all_features}
    if technical:
        combinations["technical"] = technical
    if relative:
        combinations["relative"] = relative
    return combinations


def prepare_dataset(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and align an ML feature matrix and target."""
    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")

    features = list(feature_columns or get_feature_columns(df))
    missing = set(features) - set(df.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {sorted(missing)}")

    clean = df.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=features + [target_column]).copy()
    if clean.empty:
        raise ValueError("No rows remain after removing missing values")

    return clean[features].astype(float), clean[target_column].copy()


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split samples in time so future rows are never used for training."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    if len(X) < 10:
        raise ValueError("At least 10 records are required")

    split_index = int(len(X) * (1 - test_size))
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:] 
