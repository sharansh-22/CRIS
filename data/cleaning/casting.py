"""Strict datatype casting helpers for the CRIS Data Layer."""

from __future__ import annotations

import pandas as pd

from data.errors import DatasetContractError


def cast_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Cast numeric columns strictly and reject mixed types."""

    if df.empty:
        return df

    for column in numeric_columns:
        if column not in df.columns:
            continue
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except Exception as exc:
            raise DatasetContractError(f"column {column} cannot be safely cast to numeric") from exc
    return df
