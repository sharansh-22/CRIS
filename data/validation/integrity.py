import pandas as pd
import numpy as np


def check_timezone_utc(df: pd.DataFrame, date_column: str) -> bool:
    if df.empty or date_column not in df.columns:
        return False
    series = pd.to_datetime(df[date_column], errors="coerce")
    if series.isna().any():
        return False
    tz = getattr(series.dt, "tz", None)
    return str(tz) == "UTC"


def check_numeric_finiteness(df: pd.DataFrame, numeric_columns: list[str] | None = None) -> bool:
    columns = numeric_columns or [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(series.dropna().to_numpy()).all():
            return False
    return True


def check_expected_coverage(df: pd.DataFrame, date_column: str, minimum_rows: int, max_gap_days: int | None = None) -> bool:
    if df.empty or date_column not in df.columns:
        return False
    if len(df) < minimum_rows:
        return False
    if max_gap_days is None:
        return True
    series = pd.to_datetime(df[date_column], utc=True, errors="coerce").dropna().sort_values()
    if len(series) < 2:
        return False
    gaps = series.diff().dropna().dt.days
    return bool((gaps <= max_gap_days).all())

def check_is_chronological(df: pd.DataFrame, date_column: str) -> bool:
    """Check if the dataframe is sorted chronologically by the date column."""
    if df.empty or date_column not in df.columns:
        return False
        
    return df[date_column].is_monotonic_increasing

def check_duplicate_timestamps(df: pd.DataFrame, date_column: str) -> int:
    """Count duplicate timestamps."""
    if df.empty or date_column not in df.columns:
        return 0
        
    return df.duplicated(subset=[date_column]).sum()
    
def check_missing_values(df: pd.DataFrame) -> int:
    """Count total missing values in the dataframe."""
    return df.isna().sum().sum()
