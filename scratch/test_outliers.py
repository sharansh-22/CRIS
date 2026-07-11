import polars as pl
import pytest
from data.cleaning.outliers import handle_outliers

def test_outliers_clip():
    df = pl.DataFrame({"A": [1, 2, 3, 4, 5, 100], "B": [10, 20, 30, 40, 50, -100]})
    result = handle_outliers(df, ["A", "B"], method="clip")
    
    # 1st and 99th percentiles for A: 1.05 and 95.25
    # 1st and 99th percentiles for B: 9.45 and 49.5
    # (polars uses linear interpolation by default for quantiles)
    clipped = result.dataframe
    assert clipped.select(pl.col("A").max()).item() <= 95.25
    assert clipped.select(pl.col("B").min()).item() >= 9.45

def test_outliers_robust_zscore():
    # Construct an array where one value is a clear outlier
    df = pl.DataFrame({"A": [10, 12, 11, 13, 12, 100, 11, 10, 12, 11]})
    
    # Median is 11.0, MAD is 1.0
    # robust z-score for 100 is: 0.6745 * (100 - 11) / 1.0 = 60.03 > 3.5
    result = handle_outliers(df, ["A"], method="robust_zscore_filter")
    filtered = result.dataframe
    
    assert len(filtered) == 9
    assert 100 not in filtered["A"].to_list()

def test_strict_typing_enforcement():
    df = pl.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    
    with pytest.raises(TypeError, match="CRIS Constitution Violation: Column 'B' is not numeric"):
        handle_outliers(df, ["A", "B"], method="clip")

if __name__ == "__main__":
    pytest.main([__file__])
