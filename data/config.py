from pathlib import Path
from typing import Dict, Any, List

from data.registry import DatasetDefinition, DatasetRegistry, DuplicatePolicy, MissingValuePolicy, SemanticRule

# Define root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
METADATA_DIR = DATA_DIR / "metadata"
QUARANTINE_DIR = DATA_DIR / "quarantine"
STAGED_DIR = DATA_DIR / "staged"

# Default Formats
DEFAULT_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # ISO 8601 UTC
DEFAULT_DATE_COLUMN = "Date"

DATASET_REGISTRY = DatasetRegistry(
    {
        "market_indices": DatasetDefinition(
            name="market_indices",
            provider="yahoo",
            expected_columns=("Date", "Open", "High", "Low", "Close", "Volume", "Ticker"),
            provider_args={"tickers": ["SPY", "^NSEI", "^VIX"]},
            source_url_template="https://finance.yahoo.com/quote/{ticker}/history",
            missing_value_policy=MissingValuePolicy("forward_fill", max_missing_ratio=0.02),
            duplicate_policy=DuplicatePolicy(row_strategy="drop", timestamp_strategy="drop_first"),
            semantic_rules=(
                SemanticRule("Open", min_value=0.0),
                SemanticRule("High", min_value=0.0),
                SemanticRule("Low", min_value=0.0),
                SemanticRule("Close", min_value=0.0),
                SemanticRule("Volume", min_value=0.0, reject_negative=True),
            ),
            expected_frequency="daily",
            coverage_window="2005-present",
            allowed_gap_policy="weekend_and_market_holiday",
            outlier_policy="winsorize",
            max_rows=20000,
            max_file_size_mb=200,
            source_metadata={"provider_family": "equity_index"},
        ),
        "market_equities": DatasetDefinition(
            name="market_equities",
            provider="yahoo",
            expected_columns=("Date", "Open", "High", "Low", "Close", "Volume", "Ticker"),
            provider_args={"tickers": ["NVDA", "TSLA", "HDFCBANK.NS"]},
            source_url_template="https://finance.yahoo.com/quote/{ticker}/history",
            missing_value_policy=MissingValuePolicy("forward_fill", max_missing_ratio=0.02),
            duplicate_policy=DuplicatePolicy(row_strategy="drop", timestamp_strategy="drop_first"),
            semantic_rules=(
                SemanticRule("Open", min_value=0.0),
                SemanticRule("High", min_value=0.0),
                SemanticRule("Low", min_value=0.0),
                SemanticRule("Close", min_value=0.0),
                SemanticRule("Volume", min_value=0.0, reject_negative=True),
            ),
            expected_frequency="daily",
            coverage_window="2005-present",
            allowed_gap_policy="weekend_and_market_holiday",
            outlier_policy="winsorize",
            max_rows=30000,
            max_file_size_mb=250,
            source_metadata={"provider_family": "equity"},
        ),
        "market_macro": DatasetDefinition(
            name="market_macro",
            provider="yahoo",
            expected_columns=("Date", "Open", "High", "Low", "Close", "Volume", "Ticker"),
            provider_args={"tickers": ["GLD", "^TNX"]},
            source_url_template="https://finance.yahoo.com/quote/{ticker}/history",
            missing_value_policy=MissingValuePolicy("forward_fill", max_missing_ratio=0.02),
            duplicate_policy=DuplicatePolicy(row_strategy="drop", timestamp_strategy="drop_first"),
            semantic_rules=(
                SemanticRule("Open", min_value=0.0),
                SemanticRule("High", min_value=0.0),
                SemanticRule("Low", min_value=0.0),
                SemanticRule("Close", min_value=0.0),
                SemanticRule("Volume", min_value=0.0, reject_negative=True),
            ),
            expected_frequency="daily",
            coverage_window="2005-present",
            allowed_gap_policy="weekend_and_market_holiday",
            outlier_policy="winsorize",
            max_rows=20000,
            max_file_size_mb=200,
            source_metadata={"provider_family": "macro_proxy"},
        ),
        "market_structure_etfs": DatasetDefinition(
            name="market_structure_etfs",
            provider="yahoo",
            expected_columns=("Date", "Open", "High", "Low", "Close", "Volume", "Ticker"),
            provider_args={"tickers": ["XLY", "XLI", "XLB", "XLF", "XLE", "XLU", "XLP", "XLV", "XLRE"]},
            source_url_template="https://finance.yahoo.com/quote/{ticker}/history",
            missing_value_policy=MissingValuePolicy("forward_fill", max_missing_ratio=0.02),
            duplicate_policy=DuplicatePolicy(row_strategy="drop", timestamp_strategy="drop_first"),
            semantic_rules=(
                SemanticRule("Open", min_value=0.0),
                SemanticRule("High", min_value=0.0),
                SemanticRule("Low", min_value=0.0),
                SemanticRule("Close", min_value=0.0),
                SemanticRule("Volume", min_value=0.0, reject_negative=True),
            ),
            expected_frequency="daily",
            coverage_window="2005-present",
            allowed_gap_policy="weekend_and_market_holiday",
            outlier_policy="winsorize",
            max_rows=50000,
            max_file_size_mb=300,
            source_metadata={"provider_family": "sector_etf"},
        ),
    }
)

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = DATASET_REGISTRY.as_config_map()
