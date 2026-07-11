from __future__ import annotations

import io
import logging
from datetime import datetime
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from data.contracts import RawDataset
from data.errors import DataSourceError

logger = logging.getLogger(__name__)

def _fetch_fred_csv(series_id: str, retries: int = 3, timeout: int = 30) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=timeout) as response:
                payload = response.read()
            return pd.read_csv(io.BytesIO(payload))
        except (URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            logger.warning(f"Attempt {attempt} failed for FRED series {series_id}: {exc}")
    raise DataSourceError(f"failed to fetch FRED series {series_id}") from last_error


def load_fred_data(dataset_name: str, series_ids: list[str], retries: int = 3) -> RawDataset:
    """Fetch macroeconomic data from FRED using the public CSV endpoint."""
    if not series_ids:
        raise DataSourceError(f"no FRED series ids configured for {dataset_name}")

    logger.info(f"Fetching {len(series_ids)} FRED series for dataset: {dataset_name}")
    merged: pd.DataFrame | None = None
    source_urls: list[str] = []

    for series_id in series_ids:
        frame = _fetch_fred_csv(series_id, retries=retries)
        if frame.empty:
            raise DataSourceError(f"FRED returned empty data for series {series_id}")
        if "DATE" not in frame.columns:
            raise DataSourceError(f"FRED payload for {series_id} is missing DATE column")
        value_columns = [column for column in frame.columns if column != "DATE"]
        if len(value_columns) != 1:
            raise DataSourceError(f"FRED payload for {series_id} has unexpected columns: {list(frame.columns)}")

        value_column = value_columns[0]
        frame = frame.rename(columns={"DATE": "Date", value_column: series_id})
        frame["Date"] = pd.to_datetime(frame["Date"], format="%Y-%m-%d", utc=True, errors="raise")
        source_urls.append(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")

        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="Date", how="outer")

    if merged is None:
        raise DataSourceError(f"failed to fetch any FRED series for {dataset_name}")

    merged = merged.sort_values("Date").reset_index(drop=True)
    merged["__row_id"] = [f"fred:{idx}" for idx in range(len(merged))]

    return RawDataset(
        dataset_name=dataset_name,
        data=merged,
        fetch_timestamp=datetime.utcnow(),
        provider="fred",
        provider_version="fred-graph-csv",
        source_url=", ".join(source_urls),
        source_metadata={"series_ids": series_ids, "source_urls": source_urls},
        source_row_count=len(merged),
        source_column_count=len(merged.columns),
    )
