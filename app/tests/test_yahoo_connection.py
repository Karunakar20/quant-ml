
from datetime import date, timedelta

from app.data.providers.yahoo import YahooFinanceProvider


def main():

    provider = YahooFinanceProvider()

    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    df = provider.fetch_history(
        symbol="RELIANCE.NS",
        start_date=start_date,
        end_date=end_date,
    )

    print(df.head())
    print("\nTotal candles:", len(df))
    print("\nData types:")
    print(df.dtypes)


if __name__ == "__main__":
    main()


# PYTHONPATH=. python app/tests/test_yahoo_connection.py

