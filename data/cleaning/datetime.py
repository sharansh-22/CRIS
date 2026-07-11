import logging

import pandas as pd

from data.errors import DatasetContractError

logger = logging.getLogger(__name__)

def parse_datetimes(df: pd.DataFrame, date_column: str, date_format: str) -> pd.DataFrame:
    """Parse the date column using an explicit format and standardize to UTC."""
    if df.empty or date_column not in df.columns:
        return df

    if not date_format:
        raise DatasetContractError(f"explicit date format is required for column {date_column}")

    try:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format, utc=True, exact=True, errors="raise")
    except Exception as exc:
        logger.error(f"Failed to parse datetimes in column {date_column}: {exc}")
        raise DatasetContractError(f"ambiguous or invalid datetime values in {date_column}") from exc

    return df
