import pandas as pd
import logging
from dataclasses import dataclass, field


@dataclass(slots=True)
class DuplicateHandlingResult:
    dataframe: pd.DataFrame
    strategy: str
    duplicate_rows_removed: int = 0
    duplicate_timestamps_removed: int = 0
    affected_row_indices: list[int] = field(default_factory=list)

logger = logging.getLogger(__name__)

def handle_duplicates(df: pd.DataFrame, date_column: str, strategy: str) -> DuplicateHandlingResult:
    """Remove duplicate rows and duplicate timestamps."""
    if df.empty:
        return DuplicateHandlingResult(dataframe=df, strategy=strategy)

    initial_len = len(df)
    
    # Remove exact duplicate rows
    exact_duplicates = df.index[df.duplicated()].tolist()
    df = df.drop_duplicates()
    
    # Remove duplicate timestamps
    timestamp_removed_indices: list[int] = []
    if date_column in df.columns:
        keep_strat = "first" if strategy == "drop_first" else "last" if strategy == "drop_last" else False
        timestamp_removed_indices = df.index[df.duplicated(subset=[date_column], keep=keep_strat)].tolist()
        df = df.drop_duplicates(subset=[date_column], keep=keep_strat)
        
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows/timestamps using strategy '{strategy}'.")

    return DuplicateHandlingResult(
        dataframe=df,
        strategy=strategy,
        duplicate_rows_removed=len(exact_duplicates),
        duplicate_timestamps_removed=len(timestamp_removed_indices),
        affected_row_indices=exact_duplicates + timestamp_removed_indices,
    )
