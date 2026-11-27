
import pandas as pd


HORIZONS = {
    "daily": {
        1: 0.005,
        5: 0.015,
        10: 0.020,
    },
    "weekly": {
        1: 0.010,
        2: 0.020,
        4: 0.030,
    },
}


def add_forward_targets(
    df: pd.DataFrame,
    timeframe: str = "daily",
) -> pd.DataFrame:
    """
    Generate forward returns and three-class targets.

    Bullish  = 1
    Sideways = 0
    Bearish  = -1

    The final rows will have missing targets because
    their future closing prices are not yet available.
    """

    if timeframe not in HORIZONS:
        raise ValueError("Invalid timeframe")

    if "Close" not in df.columns:
        raise ValueError("Close column is required")

    if (df["Close"] <= 0).any():
        raise ValueError("Close prices must be positive")

    df = df.copy()

    for horizon, neutral_band in HORIZONS[timeframe].items():

        suffix = "d" if timeframe == "daily" else "w"

        future_return = (
            df["Close"].shift(-horizon) / df["Close"]
        ) - 1

        return_column = (
            f"forward_return_{horizon}{suffix}"
        )

        target_column = (
            f"target_{horizon}{suffix}"
        )

        df[return_column] = future_return

        target = pd.Series(
            float("nan"),
            index=df.index,
            dtype="float64",
        )

        valid = future_return.notna()

        bullish = valid & (
            future_return > neutral_band
        )

        bearish = valid & (
            future_return < -neutral_band
        )

        sideways = valid & ~bullish & ~bearish

        target.loc[bullish] = 1
        target.loc[sideways] = 0
        target.loc[bearish] = -1

        df[target_column] = target

    return df
