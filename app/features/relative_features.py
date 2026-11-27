
import pandas as pd


def calculate_return(
    df: pd.DataFrame,
    periods: int,
) -> pd.Series:

    return df["Close"].pct_change(periods)


def add_relative_features(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    sector_df: pd.DataFrame | None = None,
    timeframe: str = "daily",
) -> pd.DataFrame:
    """
    Add market-relative and sector-relative features.

    Rolling correlation and beta use historical returns only.
    """

    if timeframe not in {"daily", "weekly"}:
        raise ValueError("Invalid timeframe")

    stock_df = stock_df.copy().sort_index()
    benchmark_df = benchmark_df.copy().sort_index()

    periods = {
        "short": 1,
        "medium": 5 if timeframe == "daily" else 2,
        "long": 20 if timeframe == "daily" else 4,
    }

    stock_returns = stock_df["Close"].pct_change()
    market_returns = benchmark_df["Close"].pct_change()

    market_returns = market_returns.reindex(
        stock_df.index
    )

    result = stock_df.copy()

    # --------------------------------------------------
    # 1. Market-relative features
    # --------------------------------------------------

    for label, period in periods.items():

        stock_return = stock_df["Close"].pct_change(period)
        market_return = benchmark_df["Close"].pct_change(
            period
        ).reindex(stock_df.index)

        result[f"market_return_{label}"] = market_return

        result[f"relative_return_market_{label}"] = (
            stock_return - market_return
        )

    # --------------------------------------------------
    # 2. Market beta and correlation
    # --------------------------------------------------

    windows = (
        [20, 60]
        if timeframe == "daily"
        else [12, 26]
    )

    for window in windows:

        covariance = stock_returns.rolling(
            window,
            min_periods=window,
        ).cov(market_returns)

        market_variance = market_returns.rolling(
            window,
            min_periods=window,
        ).var()

        result[f"market_beta_{window}"] = (
            covariance / market_variance.replace(0, pd.NA)
        )

        result[f"market_correlation_{window}"] = (
            stock_returns.rolling(
                window,
                min_periods=window,
            ).corr(market_returns)
        )

    # --------------------------------------------------
    # 3. Sector-relative features
    # --------------------------------------------------

    if sector_df is not None:

        sector_df = sector_df.copy().sort_index()

        for label, period in periods.items():

            stock_return = stock_df["Close"].pct_change(
                period
            )

            sector_return = (
                sector_df["Close"]
                .pct_change(period)
                .reindex(stock_df.index)
            )

            result[f"sector_return_{label}"] = sector_return

            result[f"relative_return_sector_{label}"] = (
                stock_return - sector_return
            )

        sector_returns = (
            sector_df["Close"]
            .pct_change()
            .reindex(stock_df.index)
        )

        for window in windows:

            covariance = stock_returns.rolling(
                window,
                min_periods=window,
            ).cov(sector_returns)

            sector_variance = sector_returns.rolling(
                window,
                min_periods=window,
            ).var()

            result[f"sector_beta_{window}"] = (
                covariance / sector_variance.replace(0, pd.NA)
            )

            result[f"sector_correlation_{window}"] = (
                stock_returns.rolling(
                    window,
                    min_periods=window,
                ).corr(sector_returns)
            )

    return result
