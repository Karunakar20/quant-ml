
import logging
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import PROJECT_ROOT, settings


def configure_logging() -> None:
    log_file = settings.LOG_FILE.expanduser()
    if not log_file.is_absolute():
        log_file = PROJECT_ROOT / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level = logging.getLevelNamesMapping().get(settings.LOG_LEVEL.upper())
    if level is None:
        raise ValueError(f"Invalid LOG_LEVEL: {settings.LOG_LEVEL}")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
        force=True,
    )

    for logger_name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "sqlalchemy.engine",
    ):
        named_logger = logging.getLogger(logger_name)
        named_logger.handlers.clear()
        named_logger.propagate = True
        named_logger.setLevel(level)

    # Request middleware logs access details once; avoid duplicate Uvicorn access lines.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # SQL is opt-in because it can be very verbose; bound parameters stay hidden.
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.SQL_ECHO else logging.WARNING
    )
    logging.captureWarnings(True)

    logging.getLogger(__name__).info("Logging initialized; file=%s", log_file)


logger = logging.getLogger("quant")
