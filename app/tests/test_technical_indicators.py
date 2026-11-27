
import numpy as np
import pandas as pd

from app.features.technical import (
    add_technical_indicators,
)


def create_sample_data(rows=300):

    index = pd.bdate_range(
        start="2024-01-01",
        periods=rows,
    )

    close = pd.Series(
        np.linspace(100, 150, rows),
        index=index,
    )

    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.random.default_rng(42).integers(
                10000,
                50000,
                rows,
            ),
        },
        index=index,
    )


def test_indicators_are_generated():

    df = create_sample_data()

    result = add_technical_indicators(df)

    expected_columns = [
        "rsi_14",
        "ema_12",
        "ema_26",
        "ema_50",
        "macd",
        "macd_signal",
        "atr_14",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_percent",
        "volume_ratio",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_indicator_values_are_valid():

    df = create_sample_data()

    result = add_technical_indicators(df)

    valid_rsi = result["rsi_14"].dropna()

    assert valid_rsi.between(0, 100).all()

    assert (
        result["bb_upper"].dropna()
        >= result["bb_lower"].dropna()
    ).all()

    assert (result["atr_14"].dropna() >= 0).all()


def test_input_dataframe_is_not_modified():

    df = create_sample_data()

    original_columns = list(df.columns)

    add_technical_indicators(df)

    assert list(df.columns) == original_columns
