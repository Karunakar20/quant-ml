
def calculate_eda_metrics(
    df: pd.DataFrame,
    timeframe: str = "daily",
) -> dict:

    returns = df["Close"].pct_change().dropna()

    annualization = (
        252 if timeframe == "daily" else 52
    )

    rolling_volatility = (
        returns.rolling(20).std()
        * np.sqrt(annualization)
    )

    return {
        "records": len(df),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "mean_return": returns.mean(),
        "median_return": returns.median(),
        "return_std": returns.std(),
        "annualized_volatility": (
            returns.std() * np.sqrt(annualization)
        ),
        "best_period_return": returns.max(),
        "worst_period_return": returns.min(),
        "max_close": df["Close"].max(),
        "min_close": df["Close"].min(),
        "average_volume": df["Volume"].mean(),
        "missing_values": int(df.isna().sum().sum()),
        "latest_20_period_volatility": rolling_volatility.iloc[-1],
    }
