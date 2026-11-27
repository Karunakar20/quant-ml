
import json
import logging

import numpy as np
import pandas as pd

from src.config.settings import settings
from src.data.symbols import (
    get_market_symbols,
    INDEX_SYMBOLS,
    STOCK_SYMBOLS,
    SECTOR_INDEX_SYMBOLS,
    STOCK_SECTOR_MAP,
)
from src.features.technical_indicators import (
    add_technical_indicators,
)
from src.features.relative_features import (
    add_relative_features,
)
from src.features.target_generator import (
    add_forward_targets,
    HORIZONS,
)


logger = logging.getLogger(__name__)


# Absolute price-level columns are retained for analysis
# but excluded from the initial ML feature matrix.
EXCLUDED_FEATURES = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "ema_12",
    "ema_26",
    "ema_50",
    "atr_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_middle",
    "bb_lower",
}


def load_raw_data(
    name: str,
    timeframe: str,
) -> pd.DataFrame:

    path = (
        settings.raw_dir
        / timeframe
        / f"{name}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        index_col="Date",
        parse_dates=["Date"],
    )

    df = df.sort_index()

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    return df


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    excluded_prefixes = (
        "forward_return_",
        "target_",
    )

    return [
        column
        for column in df.columns
        if column not in EXCLUDED_FEATURES
        and not column.startswith(excluded_prefixes)
    ]


def get_target_columns(
    timeframe: str,
) -> list[str]:

    suffix = "d" if timeframe == "daily" else "w"

    return [
        f"target_{horizon}{suffix}"
        for horizon in HORIZONS[timeframe]
    ]


def build_dataset(
    name: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:

    logger.info(
        "Building features for %s | %s",
        name,
        timeframe,
    )

    df = load_raw_data(name, timeframe)

    # 1. Technical indicators
    df = add_technical_indicators(
        df,
        timeframe,
    )

    # 2. Market-relative features
    # NIFTY 50 itself does not need to be compared
    # against itself.
    if name != "NIFTY50":

        benchmark = load_raw_data(
            "NIFTY50",
            timeframe,
        )

        sector = None

        if name in STOCK_SECTOR_MAP:

            sector_name = STOCK_SECTOR_MAP[name]

            if sector_name in SECTOR_INDEX_SYMBOLS:

                try:
                    sector = load_raw_data(
                        sector_name,
                        timeframe,
                    )

                except FileNotFoundError:
                    logger.warning(
                        "Sector data unavailable for %s",
                        name,
                    )

        df = add_relative_features(
            stock_df=df,
            benchmark_df=benchmark,
            sector_df=sector,
            timeframe=timeframe,
        )

    # 3. Forward-return targets
    df = add_forward_targets(
        df,
        timeframe,
    )

    # 4. Clean infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # 5. Identify model input columns
    feature_columns = get_feature_columns(df)

    target_columns = get_target_columns(timeframe)

    # 6. Historical training dataset:
    # all features and all target horizons must be available.
    ml_df = df.dropna(
        subset=feature_columns + target_columns
    ).copy()

    if ml_df.empty:
        raise ValueError(
            f"No valid ML records available for {name}"
        )

    # 7. Save the ML-ready dataset
    output_dir = (
        settings.processed_dir / timeframe
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"{name}.csv"

    temporary_path = output_path.with_suffix(".tmp")

    ml_df.to_csv(
        temporary_path,
        index=True,
        index_label="Date",
    )

    temporary_path.replace(output_path)

    logger.info(
        "%s: raw=%s ML-ready=%s features=%s",
        name,
        len(df),
        len(ml_df),
        len(feature_columns),
    )

    return ml_df


def run_feature_pipeline(
    timeframe: str = "daily",
) -> None:

    symbols = get_market_symbols()

    feature_schema = {}

    success_count = 0
    failed_symbols = []

    for name, symbol in symbols.items():

        try:

            ml_df = build_dataset(
                name=name,
                symbol=symbol,
                timeframe=timeframe,
            )

            feature_schema[name] = {
                "features": get_feature_columns(ml_df),
                "targets": get_target_columns(timeframe),
            }

            success_count += 1

        except Exception:

            logger.exception(
                "Feature pipeline failed for %s",
                name,
            )

            failed_symbols.append(name)

    schema_path = (
        settings.processed_dir
        / timeframe
        / "feature_schema.json"
    )

    schema_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    schema_path.write_text(
        json.dumps(
            feature_schema,
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info(
        "%s pipeline complete. Success=%s Failed=%s",
        timeframe,
        success_count,
        len(failed_symbols),
    )

    if failed_symbols:
        logger.warning(
            "Failed symbols: %s",
            failed_symbols,
        )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    run_feature_pipeline("daily")
    run_feature_pipeline("weekly")
