
import numpy as np
import pandas as pd

from app.features.relative_features import (
    add_relative_features,
)


def test_relative_features():

    dates = pd.bdate_range(
        start="2024-01-01",
        periods=100,
    )

    benchmark = pd.DataFrame(
        {
            "Close": np.linspace(100, 120, 100),
        },
        index=dates,
    )

    stock = pd.DataFrame(
        {
            "Open": np.linspace(100, 125, 100),
            "High": np.linspace(102, 127, 100),
            "Low": np.linspace(99, 124, 100),
            "Close": np.linspace(101, 126, 100),
            "Volume": 10000,
        },
        index=dates,
    )

    result = add_relative_features(
        stock_df=stock,
        benchmark_df=benchmark,
    )

    assert "market_return_short" in result.columns

    assert "relative_return_market_short" in result.columns

    assert "market_beta_20" in result.columns

    assert "market_correlation_20" in result.columns

    assert result["market_correlation_20"].dropna().between(
        -1,
        1,
    ).all()
