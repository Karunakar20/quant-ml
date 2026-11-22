
import logging
from time import perf_counter

from fastapi import FastAPI
from fastapi import Request

from app.core.logging import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quant ML",
    description="Stock Market Data and ML Platform",
    version="1.0.0",
)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    started_at = perf_counter()
    method = request.method
    path = request.url.path
    logger.info("HTTP request started method=%s path=%s", method, path)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("HTTP request failed method=%s path=%s", method, path)
        raise

    duration_ms = (perf_counter() - started_at) * 1000
    logger.info(
        "HTTP request completed method=%s path=%s status=%s duration_ms=%.2f",
        method,
        path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/")
def root():
    return {
        "message": "Welcome to BharatQuant ML API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


from app.routers.pipline import router as pipeline_router

app.include_router(pipeline_router, prefix="/api/v1")

logger.info("Quant ML API initialized")