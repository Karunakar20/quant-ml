
import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.session import get_db
from app.schema.pipline_schema import IngestionResponse,IngestionRequest,StockIngestionResult
from app.database.models import Stock
from app.data.ingestion import (
    load_stock_universe,
    ingest_stock_history,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion",tags=["Data Ingestion"],)


@router.post("/run", response_model=IngestionResponse)
def run_ingestion(
    request: IngestionRequest,
    db: Session = Depends(get_db),
):
    """
    Load the stock universe and ingest historical stock data.
    Existing stocks are skipped by load_stock_universe().
    """

    start_date = (
        request.start_date
        or date.fromisoformat(settings.HISTORY_START_DATE)
    )

    end_date = (
        request.end_date
        or (
            date.fromisoformat(settings.HISTORY_END_DATE)
            if settings.HISTORY_END_DATE
            else date.today()
        )
    )

    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date cannot be greater than end_date",
        )

    if request.resolution != "D":
        raise HTTPException(
            status_code=400,
            detail="Only daily resolution ('D') is currently supported",
        )

    try:
        newly_inserted = load_stock_universe(db)
    except Exception as exc:
        logger.exception("Failed to load stock universe")
        raise HTTPException(
            status_code=500,
            detail="Failed to load stock universe",
        ) from exc

    logger.info(
        "New stocks inserted: %s",
        newly_inserted,
    )

    # return True


    # Read IDs and symbols before processing individual stocks.
    stock_rows = db.execute(
        select(Stock.id, Stock.symbol).where(
            Stock.is_active.is_(True)
        )
    ).all()

    successful = 0
    failed = 0
    results = []

    for stock_id, symbol in stock_rows:
        try:
            stock = db.get(Stock, stock_id)

            if stock is None:
                raise ValueError(
                    f"Stock record not found: {symbol}"
                )

            count = ingest_stock_history(
                db=db,
                stock=stock,
                start_date=start_date,
                end_date=end_date,
                resolution=request.resolution,
            )

            successful += 1

            results.append(
                StockIngestionResult(
                    symbol=symbol,
                    status="SUCCESS",
                    candles=count,
                )
            )

            logger.info(
                "%s: %s candles",
                symbol,
                count,
            )

        except Exception as exc:
            db.rollback()
            failed += 1

            results.append(
                StockIngestionResult(
                    symbol=symbol,
                    status="FAILED",
                    error=str(exc)[:500],
                )
            )

            logger.exception(
                "Failed stock: %s",
                symbol,
            )

    logger.info("========== INGESTION SUMMARY ==========")
    logger.info("Total stocks: %s", len(stock_rows))
    logger.info("Successful: %s", successful)
    logger.info("Failed: %s", failed)

    return IngestionResponse(
        message="Historical data ingestion completed",
        total_stocks=len(stock_rows),
        newly_inserted_stocks=newly_inserted,
        successful=successful,
        failed=failed,
        results=results,
    )
