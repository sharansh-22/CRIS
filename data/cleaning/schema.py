import pandas as pd
from typing import List
import logging
from dataclasses import dataclass, field


@dataclass(slots=True)
class SchemaStandardizationResult:
    dataframe: pd.DataFrame
    added_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    preserved_columns: list[str] = field(default_factory=list)

logger = logging.getLogger(__name__)

def standardize_columns(df: pd.DataFrame, expected_columns: List[str], allow_extra_columns: bool = False) -> SchemaStandardizationResult:
    """Ensure the dataframe has the expected columns and report schema drift."""
    if df.empty:
        return SchemaStandardizationResult(dataframe=df)

    original_columns = list(df.columns)
    missing = [col for col in expected_columns if col not in df.columns]
    extras = [col for col in df.columns if col not in expected_columns and not col.startswith("__")]
    if missing:
        logger.warning(f"Missing expected columns: {missing}. Filling with NaNs to match schema.")
        for m in missing:
            df[m] = pd.NA

    dropped_columns: list[str] = []
    if extras:
        if allow_extra_columns:
            preserved = list(expected_columns) + [column for column in extras if column not in expected_columns] + [column for column in original_columns if column.startswith("__")]
            df = df[preserved]
            return SchemaStandardizationResult(
                dataframe=df,
                added_columns=missing,
                dropped_columns=[],
                preserved_columns=extras,
            )
        dropped_columns = extras
        logger.warning(f"Unexpected columns will be removed by contract: {extras}")

    # Reorder and filter columns
    df = df[expected_columns]
    return SchemaStandardizationResult(
        dataframe=df,
        added_columns=missing,
        dropped_columns=dropped_columns,
        preserved_columns=[column for column in original_columns if column in expected_columns or column.startswith("__")],
    )
