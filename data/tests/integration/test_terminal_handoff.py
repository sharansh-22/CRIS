import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from data.contracts import load_published_dataset
from data.pipeline.prepare_data import _publish_dataset

def test_terminal_handoff_fidelity(tmp_path):
    """
    Execute the Terminal Handoff Round-Trip Guarantee test.
    Mathematically prove the integrity of the CRIS Data Layer serialization boundary.
    """
    # Isolate the test environment using a temporary directory
    with patch("data.pipeline.prepare_data.PROCESSED_DIR", tmp_path), \
         patch("data.contracts.PROCESSED_DIR", tmp_path):
        
        # 1. Edge Case Injection
        # Construct a synthetic payload containing NaN values, float64 precision limits,
        # and strict datetime64[ns, UTC] timestamps.
        df_original = pd.DataFrame({
            "timestamp": pd.date_range(start="2026-01-01", periods=3, tz="UTC"),
            "value_float64": np.array([1.123456789012345, np.nan, 3.141592653589793], dtype=np.float64),
            "categorical_id": ["alpha", "beta", "gamma"]
        })
        
        dataset_name = "synthetic_handoff_test"
        
        # 2. Write across the Serialization Boundary (Pandas -> PyArrow -> Parquet)
        _publish_dataset(dataset_name, df_original, output_format="parquet")
        
        # 3. Read back across the Deserialization Boundary (Parquet -> PyArrow -> Pandas)
        df_loaded = load_published_dataset(dataset_name)
        
        # 4. Strict Equality Assertion
        # Violations of CRIS Constitution (e.g., timezone stripping, float downcasting)
        # will cause this assertion to fail loudly.
        pd.testing.assert_frame_equal(
            df_original, 
            df_loaded, 
            check_exact=True, 
            check_dtype=True, 
            check_datetimelike_compat=False
        )

if __name__ == "__main__":
    pytest.main([__file__])
