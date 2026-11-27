
import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


def add_technical_indicators(
    df: pd.DataFrame,
    timeframe: str = "daily",
) -> pd.DataFrame:
    """
    Add technical indicators and time-series features.

    Input:
        OHLCV dataframe indexed by Date.

    Output:
        Original OHLCV + engineered features.

    No backward filling is performed.
    """

    if timeframe not in {"daily", "weekly"}:
        raise ValueError("timeframe must be daily or weekly")

    required = {"Open", "High", "Low", "Close", "Volume"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    if df.empty:
        raise ValueError("Input dataframe is empty")

    df = df.copy().sort_index()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    if (close <= 0).any():
        raise ValueError("Close prices must be positive")

    # Periods are expressed in trading sessions for daily
    # data and trading weeks for weekly data.

    periods = {
        "short": 5,
        "medium": 10 if timeframe == "daily" else 4,
        "long": 20 if timeframe == "daily" else 12,
        "volatility": 60 if timeframe == "daily" else 26,
    }

    # --------------------------------------------------
    # 1. Returns
    # --------------------------------------------------

    for period in [1, 2, 5, 10, 20]:
        if timeframe == "weekly" and period == 2:
            continue

        suffix = "d" if timeframe == "daily" else "w"

        df[f"return_{period}{suffix}"] = (
            close.pct_change(period)
        )

    df["log_return_1"] = np.log(
        close / close.shift(1)
    )

    # --------------------------------------------------
    # 2. Rolling volatility
    # --------------------------------------------------

    annualization = 252 if timeframe == "daily" else 52

    for label, window in periods.items():

        rolling_std = close.pct_change().rolling(
            window=window,
            min_periods=window,
        ).std()

        df[f"volatility_{label}"] = (
            rolling_std * np.sqrt(annualization)
        )

    # --------------------------------------------------
    # 3. Price movement features
    # --------------------------------------------------

    df["candle_body"] = (
        close - df["Open"]
    ) / close

    df["high_low_range"] = (
        high - low
    ) / close

    df["close_to_high"] = (
        close / high
    ) - 1

    df["close_to_low"] = (
        close / low
    ) - 1

    # --------------------------------------------------
    # 4. RSI
    # --------------------------------------------------

    df["rsi_14"] = RSIIndicator(
        close=close,
        window=14,
        fillna=False,
    ).rsi()

    # --------------------------------------------------
    # 5. EMA
    # --------------------------------------------------

    for window in [12, 26, 50]:

        ema = EMAIndicator(
            close=close,
            window=window,
            fillna=False,
        ).ema_indicator()

        df[f"ema_{window}"] = ema

        # Normalized distance from EMA.
        df[f"close_ema_{window}_distance"] = (
            close / ema
        ) - 1

    # --------------------------------------------------
    # 6. MACD
    # --------------------------------------------------

    macd_indicator = MACD(
        close=close,
        window_slow=26,
        window_fast=12,
        window_sign=9,
        fillna=False,
    )

    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    df["macd_histogram"] = macd_indicator.macd_diff()

    # Normalize to make values comparable across stocks.
    df["macd_normalized"] = df["macd"] / close
    df["macd_signal_normalized"] = df["macd_signal"] / close
    df["macd_histogram_normalized"] = (
        df["macd_histogram"] / close
    )

    # --------------------------------------------------
    # 7. ATR
    # --------------------------------------------------

    atr_indicator = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=14,
        fillna=False,
    )

    df["atr_14"] = atr_indicator.average_true_range()

    df["atr_normalized"] = (
        df["atr_14"] / close
    )

    # --------------------------------------------------
    # 8. Bollinger Bands
    # --------------------------------------------------

    bb = BollingerBands(
        close=close,
        window=20,
        window_dev=2,
        fillna=False,
    )

    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    df["bb_width"] = bb.bollinger_wband()

    df["bb_percent"] = bb.bollinger_pband()

    # --------------------------------------------------
    # 9. Volume features
    # --------------------------------------------------

    volume_available = (
        volume.notna().any()
        and volume.fillna(0).sum() > 0
    )

    if volume_available:

        df["volume_change"] = volume.pct_change()

        volume_mean = volume.rolling(
            window=20,
            min_periods=20,
        ).mean()

        volume_std = volume.rolling(
            window=20,
            min_periods=20,
        ).std()

        df["volume_zscore"] = (
            volume - volume_mean
        ) / volume_std.replace(0, np.nan)

        df["volume_ratio"] = (
            volume / volume_mean
        )

    else:
        df["volume_change"] = np.nan
        df["volume_zscore"] = np.nan
        df["volume_ratio"] = np.nan

    return df
