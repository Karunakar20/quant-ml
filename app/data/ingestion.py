
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.database.models import (
    Stock,
    HistoricalCandle,
    IngestionLog,
)
from app.data.providers.yahoo import YahooFinanceProvider
from app.data.validation import validate_candles


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STOCK_CSV = (PROJECT_ROOT / "scripts" / "nifty50_symbols.csv")


def load_stock_universe(
    db: Session,
    csv_path: str | Path = DEFAULT_STOCK_CSV,
) -> int:
    """
    Load NIFTY 50 stock symbols from CSV into PostgreSQL.

    Existing stocks are skipped.
    Only new stocks are inserted.

    Returns:
        Number of new stocks inserted.
    """

    csv_path = Path(csv_path)
    logger.info("Loading stock universe from %s", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Stock universe CSV not found: {csv_path}"
        )

    stocks_df = pd.read_csv(csv_path)

    required_columns = {
        "symbol",
        "company_name",
        "exchange",
    }

    missing_columns = required_columns - set(stocks_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    if stocks_df.empty:
        raise ValueError("Stock universe CSV is empty")

    stocks_df = stocks_df.dropna(
        subset=["symbol", "exchange"]
    )

    stocks_df["symbol"] = (
        stocks_df["symbol"].astype(str).str.strip()
    )

    stocks_df["exchange"] = (
        stocks_df["exchange"].astype(str).str.strip()
    )

    stocks_df["company_name"] = (
        stocks_df["company_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if stocks_df["symbol"].eq("").any():
        raise ValueError("Empty stock symbols found")

    if stocks_df["exchange"].eq("").any():
        raise ValueError("Empty exchanges found")

    if stocks_df["symbol"].duplicated().any():
        raise ValueError(
            "Duplicate stock symbols found in CSV"
        )

    records = [
        {
            "symbol": row.symbol,
            "company_name": row.company_name or None,
            "exchange": row.exchange,
            "is_active": True,
        }
        for row in stocks_df.itertuples(index=False)
    ]

    try:
        stmt = insert(Stock).values(records)

        stmt = stmt.on_conflict_do_nothing(
            index_elements=["symbol"]
        )

        result = db.execute(stmt)
        db.commit()

        inserted_count = result.rowcount or 0
        skipped_count = len(records) - inserted_count

        logger.info(
            "Stock universe loaded: %s inserted, %s skipped",
            inserted_count,
            skipped_count,
        )

        return inserted_count

    except Exception:
        db.rollback()
        logger.exception("Failed to load stock universe")
        raise


def save_candles(
    db: Session,
    stock_id: int,
    df: pd.DataFrame,
    timeframe: str = "D",
) -> int:
    """
    Insert historical candles or update existing candles.

    Unique key:
        stock_id + timeframe + timestamp
    """

    if df.empty:
        logger.debug("Skipping empty candle upsert for stock_id=%s", stock_id)
        return 0

    records = []

    for row in df.to_dict(orient="records"):

        timestamp = pd.Timestamp(
            row["timestamp"]
        ).to_pydatetime()

        records.append({
            "stock_id": stock_id,
            "timeframe": timeframe,
            "timestamp": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
        })

    logger.info(
        "Upserting %s candles for stock_id=%s timeframe=%s",
        len(records),
        stock_id,
        timeframe,
    )

    stmt = insert(HistoricalCandle).values(records)

    stmt = stmt.on_conflict_do_update(
        constraint="uq_stock_timeframe_timestamp",
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
        },
    )

    db.execute(stmt)
    db.commit()
    logger.info("Persisted %s candles for stock_id=%s", len(records), stock_id)

    return len(records)


def ingest_stock_history(
    db: Session,
    stock: Stock,
    start_date: date,
    end_date: date,
    resolution: str = "D",
) -> int:
    """
    Fetch Yahoo Finance historical data,
    validate OHLCV candles, and store in PostgreSQL.

    Returns:
        Number of processed candles.
    """

    started_at = datetime.now(timezone.utc)

    log = IngestionLog(
        symbol=stock.symbol,
        started_at=started_at,
        status="RUNNING",
        records_inserted=0,
    )

    db.add(log)
    db.commit()
    db.refresh(log)

    try:
        if start_date > end_date:
            raise ValueError(
                "start_date cannot be after end_date"
            )

        logger.info(
            "Starting history ingestion symbol=%s exchange=%s start=%s end=%s resolution=%s",
            stock.symbol,
            stock.exchange,
            start_date,
            end_date,
            resolution,
        )

        provider = YahooFinanceProvider()

        raw_df = provider.fetch_history(
            symbol=stock.symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1d" if resolution == "D" else resolution,
        )
        logger.info("%s: provider returned %s rows", stock.symbol, len(raw_df))

        validated_df = validate_candles(raw_df)
        logger.info(
            "%s: %s valid candles after validation",
            stock.symbol,
            len(validated_df),
        )

        processed_count = save_candles(
            db=db,
            stock_id=stock.id,
            df=validated_df,
            timeframe=resolution,
        )

        log.status = "SUCCESS"
        log.records_inserted = processed_count
        log.completed_at = datetime.now(timezone.utc)

        db.commit()

        logger.info(
            "%s: processed %s candles",
            stock.symbol,
            processed_count,
        )

        return processed_count

    except Exception as exc:
        db.rollback()

        log = db.get(IngestionLog, log.id)

        if log:
            log.status = "FAILED"
            log.error_message = str(exc)[:2000]
            log.completed_at = datetime.now(timezone.utc)

            db.commit()

        logger.exception(
            "Historical ingestion failed for %s",
            stock.symbol,
        )

        raise
