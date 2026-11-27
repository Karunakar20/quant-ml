
import pandas as pd

from app.features.target_generator import (
    add_forward_targets,
)


def test_daily_targets():

    df = pd.DataFrame({
        "Close": [
            100,
            102,
            104,
            106,
            108,
            110,
            112,
            114,
            116,
            118,
            120,
        ]
    })

    result = add_forward_targets(
        df,
        timeframe="daily",
    )

    assert result.loc[0, "target_1d"] == 1

    # assert result.loc[0, "forward_return_1d"] == 0.02
    actual = result.loc[0, "forward_return_1d"]
    assert round(float(actual), 12) == 0.02

    # No future price is available for the final row.
    assert pd.isna(result.iloc[-1]["target_1d"])


def test_weekly_targets():

    df = pd.DataFrame({
        "Close": range(100, 151)
    })

    result = add_forward_targets(
        df,
        timeframe="weekly",
    )

    assert "target_1w" in result.columns
    assert "target_2w" in result.columns
    assert "target_4w" in result.columns

    assert pd.isna(result.iloc[-1]["target_4w"])


def test_sideways_target():

    df = pd.DataFrame({
        "Close": [100, 100.1, 100.2, 100.3]
    })

    result = add_forward_targets(
        df,
        timeframe="daily",
    )

    assert result.loc[0, "target_1d"] == 0
