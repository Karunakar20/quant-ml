
from fastapi import FastAPI

from app.routers.pipline import router as pipeline_router
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="Quant ML",
    description="Stock Market Data and ML Platform",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Welcome to BharatQuant ML API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


app.include_router(pipeline_router, prefix="/api/v1")