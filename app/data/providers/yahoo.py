
import logging
from datetime import date

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooFinanceProvider:

    def fetch_history(self,symbol: str,start_date: date,end_date: date,interval: str = "1d") -> pd.DataFrame:

        yahoo_symbol = symbol

        logger.info("Downloading %s from %s to %s",yahoo_symbol,start_date,end_date)

        ticker = yf.Ticker(yahoo_symbol)

        df = ticker.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval=interval,
            auto_adjust=False,
            actions=False,
        )

        if df.empty:
            logger.warning(
                "No historical data returned for %s",
                yahoo_symbol,
            )

            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )

        df = df.reset_index()

        timestamp_column = ("Date" if "Date" in df.columns else "Datetime")

        df = df.rename(
            columns={
                timestamp_column: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

        df = df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]

        df = df.drop_duplicates(subset=["timestamp"],keep="last")

        return df.sort_values("timestamp").reset_index(drop=True)
