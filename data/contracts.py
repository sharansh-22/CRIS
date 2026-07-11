from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
import pandas as pd


class PublicationState(str, Enum):
    RAW = "raw"
    STAGED = "staged"
    VALIDATED = "validated"
    PUBLISHED = "published"
    QUARANTINED = "quarantined"


@dataclass
class TransformationStep:
    """Record of a single transformation or validation step."""

    step_name: str
    rows_before: int
    rows_after: int
    columns_before: List[str]
    columns_after: List[str]
    outcome: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetMetadata:
    """Metadata detailing the dataset version, source, and shape."""
    dataset_name: str
    provider: str
    version: str
    timestamp_generated: datetime
    num_rows: int
    num_columns: int
    columns: List[str]
    missing_value_strategy: str
    schema_version: str = "1.0.0"
    cleaning_version: str = "1.0.0"
    provider_version: str = "unknown"
    fetch_timestamp: Optional[datetime] = None
    source_url: str = ""
    expected_columns: List[str] = field(default_factory=list)
    duplicate_strategy: str = "drop_first"
    output_format: str = "parquet"
    publication_state: PublicationState = PublicationState.STAGED
    transformation_history: List[TransformationStep] = field(default_factory=list)
    row_lineage_column: str = "__row_id"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Report detailing the post-cleaning data quality."""
    dataset_name: str
    is_valid: bool
    missing_values_count: int
    duplicate_rows_count: int
    duplicate_timestamps_count: int
    is_chronological: bool
    schema_matched: bool
    schema_version: str = "1.0.0"
    expected_frequency: str = "daily"
    coverage_window: str = "unspecified"
    missing_value_strategy: str = "none"
    duplicate_strategy: str = "drop_first"
    dropped_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    schema_drift_detected: bool = False
    semantic_errors: List[str] = field(default_factory=list)
    semantic_warnings: List[str] = field(default_factory=list)
    outlier_columns: List[str] = field(default_factory=list)
    outlier_diagnostics: Dict[str, Any] = field(default_factory=dict)
    missingness_ratio: float = 0.0
    duplicate_rows_affected: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RawDataset:
    """A wrapper around raw ingested data before cleaning."""
    dataset_name: str
    data: pd.DataFrame
    fetch_timestamp: datetime
    provider: str = "unknown"
    provider_version: str = "unknown"
    source_url: str = ""
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    source_row_count: int = 0
    source_column_count: int = 0
    source_file_size_bytes: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataPackage:
    """The final production-ready artifact exported by the Data Layer."""
    dataset_name: str
    data: pd.DataFrame
    metadata: DatasetMetadata
    validation_report: ValidationReport
    publication_state: PublicationState = PublicationState.STAGED
    quarantine_path: Optional[str] = None
    published_path: Optional[str] = None
    quarantine_reason: Optional[str] = None

def load_published_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Downstream read contract for Signal Extractors.
    Loads published datasets while enforcing the PyArrow schema and preventing type guessing.
    """
    from data.config import PROCESSED_DIR
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    path = PROCESSED_DIR / f"{dataset_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"CRIS Data Layer Error: Published dataset not found at {path}")
        
    # Read using pyarrow to ensure no silent float/datetime casting occurs
    return pd.read_parquet(path, engine="pyarrow")
