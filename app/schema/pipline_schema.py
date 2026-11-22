from datetime import date
from pydantic import BaseModel

class IngestionRequest(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    resolution: str = "D"

class StockIngestionResult(BaseModel):
    symbol: str
    status: str
    candles: int = 0
    error: str | None = None

class IngestionResponse(BaseModel):
    message: str
    total_stocks: int
    newly_inserted_stocks: int
    successful: int
    failed: int
    results: list[StockIngestionResult]