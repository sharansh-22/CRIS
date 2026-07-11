from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.contracts import RawDataset
from data.errors import DataSourceError

logger = logging.getLogger(__name__)

def load_csv_data(dataset_name: str, file_path: Path, max_file_size_mb: int | None = None, chunksize_rows: int | None = None) -> RawDataset:
    """Load static CSV file."""
    logger.info(f"Loading CSV data for {dataset_name} from {file_path}")
    file_size_bytes = file_path.stat().st_size if file_path.exists() else None
    try:
        if not file_path.exists():
            raise DataSourceError(f"CSV file does not exist: {file_path}")
        if max_file_size_mb is not None and file_size_bytes > max_file_size_mb * 1024 * 1024 and chunksize_rows is None:
            raise DataSourceError(
                f"CSV file {file_path} exceeds the configured in-memory limit of {max_file_size_mb} MB"
            )
        if chunksize_rows is not None:
            frames = [chunk for chunk in pd.read_csv(file_path, chunksize=chunksize_rows)]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise DataSourceError(f"failed to load CSV dataset {dataset_name}") from e
        
    return RawDataset(
        dataset_name=dataset_name,
        data=df,
        fetch_timestamp=datetime.utcnow(),
        provider="csv",
        provider_version="pandas-read-csv",
        source_url=str(file_path),
        source_metadata={"file_path": str(file_path)},
        source_row_count=len(df),
        source_column_count=len(df.columns),
        source_file_size_bytes=file_size_bytes,
    )
