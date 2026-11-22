
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_candles(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "open", "high", "low", "close", "volume"
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # Reject invalid or missing values.
    result = result.dropna(
        subset=REQUIRED_COLUMNS
    )

    # Reject infinite values.
    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.dropna(
        subset=REQUIRED_COLUMNS
    )

    # Prices must be positive.
    price_columns = [
        "open", "high", "low", "close"
    ]

    result = result[
        (result[price_columns] > 0).all(axis=1)
    ]

    # OHLC consistency.
    result = result[
        (result["high"] >= result["low"])
        & (result["high"] >= result["open"])
        & (result["high"] >= result["close"])
        & (result["low"] <= result["open"])
        & (result["low"] <= result["close"])
    ]

    # Volume must be non-negative and integral.
    result = result[
        (result["volume"] >= 0)
        & (result["volume"] % 1 == 0)
    ]

    # Remove duplicate candle timestamps.
    result = result.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    result = result.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    result["volume"] = result["volume"].astype("int64")

    return result
