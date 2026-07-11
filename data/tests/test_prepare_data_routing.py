import pytest
import pandas as pd
from unittest.mock import patch
from pathlib import Path

from data.pipeline.prepare_data import prepare_data
from data.contracts import PublicationState
from data.registry import DatasetDefinition
from data.config import PROCESSED_DIR, QUARANTINE_DIR, DATASET_REGISTRY


@pytest.fixture
def mock_dataset_registry():
    definition = DatasetDefinition(
        name="test_macro",
        provider="csv",
        expected_columns=("Date", "Open", "High", "Low", "Close", "Volume", "Ticker"),
        output_format="parquet",
        provider_args={"file_path": "dummy.csv"}
    )
    # Patch the global registry definitions for the duration of the test
    with patch.dict(DATASET_REGISTRY._definitions, {"test_macro": definition}):
        yield definition


@patch("data.pipeline.prepare_data._load_raw_dataset")
@patch("data.pipeline.prepare_data._write_frame")
def test_valid_data_publishes_to_parquet(mock_write, mock_load, mock_dataset_registry):
    mock_load.return_value.data = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"], utc=True),
        "Open": [10.0, 11.0, 10.5, 12.0, 11.5],
        "High": [11.0, 12.0, 11.5, 13.0, 12.5],
        "Low": [9.0, 10.0, 9.5, 11.0, 10.5],
        "Close": [10.5, 11.5, 11.0, 12.5, 12.0],
        "Volume": [1000.0, 1100.0, 1050.0, 1200.0, 1150.0],
        "Ticker": ["SPY", "SPY", "SPY", "SPY", "SPY"]
    })
    mock_load.return_value.provider_version = "1.0"
    mock_load.return_value.fetch_timestamp = pd.Timestamp.utcnow()
    mock_load.return_value.source_url = "dummy"
    mock_load.return_value.source_metadata = {}
    mock_load.return_value.source_row_count = 5
    mock_load.return_value.source_column_count = 7
    
    result = prepare_data("test_macro")
    
    assert result.publication_state == PublicationState.PUBLISHED
    assert result.published_path.endswith("test_macro.parquet")
    
    # Verify _write_frame was called with "parquet" format
    mock_write.assert_any_call(result.data, PROCESSED_DIR / "test_macro.parquet", "parquet")


@patch("data.pipeline.prepare_data._load_raw_dataset")
@patch("data.pipeline.prepare_data._write_frame")
def test_invalid_data_routes_to_quarantine(mock_write, mock_load, mock_dataset_registry):
    # Missing 'Close' column should fail validation
    mock_load.return_value.data = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"], utc=True),
        "Open": [10.0, 11.0, 10.5, 12.0, 11.5],
        "High": [11.0, 12.0, 11.5, 13.0, 12.5],
        "Low": [9.0, 10.0, 9.5, 11.0, 10.5],
        "Volume": [1000.0, 1100.0, 1050.0, 1200.0, 1150.0],
        "Ticker": ["SPY", "SPY", "SPY", "SPY", "SPY"]
    })
    mock_load.return_value.provider_version = "1.0"
    mock_load.return_value.fetch_timestamp = pd.Timestamp.utcnow()
    mock_load.return_value.source_url = "dummy"
    mock_load.return_value.source_metadata = {}
    mock_load.return_value.source_row_count = 5
    mock_load.return_value.source_column_count = 6
    
    result = prepare_data("test_macro")
    
    assert result.publication_state == PublicationState.QUARANTINED
    assert result.quarantine_reason is not None
    assert "Pandera Schema Violation" in result.quarantine_reason
    
    # Quarantine saves raw and staged versions
    write_calls = [call.args[2] for call in mock_write.mock_calls]
    assert all(fmt == "parquet" for fmt in write_calls)


@patch("data.pipeline.prepare_data._load_raw_dataset")
@patch("data.pipeline.prepare_data._write_frame")
def test_pandera_schema_violation_routes_to_quarantine(mock_write, mock_load, mock_dataset_registry):
    # Setup malicious data: 'Open' contains a negative value (violating 'ge=0.0')
    # and 'Date' is not localized to UTC.
    mock_load.return_value.data = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]), # Missing UTC
        "Open": [-5.0, 11.0, 10.5, 12.0, 11.5], # Negative value
        "Close": [10.5, 11.5, 11.0, 12.5, 12.0],
        "High": [11.0, 12.0, 11.5, 13.0, 12.5],
        "Low": [9.0, 10.0, 9.5, 11.0, 10.5],
        "Volume": [1000.0, 1100.0, 1050.0, 1200.0, 1150.0],
        "Ticker": ["SPY", "SPY", "SPY", "SPY", "SPY"]
    })
    mock_load.return_value.provider_version = "1.0"
    mock_load.return_value.fetch_timestamp = pd.Timestamp.utcnow()
    mock_load.return_value.source_url = "dummy"
    mock_load.return_value.source_metadata = {}
    mock_load.return_value.source_row_count = 5
    mock_load.return_value.source_column_count = 7
    
    # Run pipeline
    result = prepare_data("test_macro")
    
    # Assert Pandera caught it and quarantined it loudly
    assert result.publication_state == PublicationState.QUARANTINED
    assert result.quarantine_reason is not None
    assert "Pandera Schema Violation" in result.quarantine_reason
