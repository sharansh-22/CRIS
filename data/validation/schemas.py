import pandera as pa
from pandera.typing import Series
import pandas as pd

class CRISBaseSchema(pa.DataFrameModel):
    """Base schema enforcing UTC and chronological constraints on the Date column."""
    Date: Series[pd.DatetimeTZDtype] = pa.Field(
        nullable=False, 
        dtype_kwargs={"unit": "ns", "tz": "UTC"}
    )

    @pa.check("Date", name="is_monotonic_increasing")
    def check_monotonic(cls, s: pd.Series) -> bool:
        return s.is_monotonic_increasing

    class Config:
        coerce = False  # Strictly forbid silent type casting
        strict = True   # Disallow unexpected columns


class MarketDataSchema(CRISBaseSchema):
    """Standard OHLCV schema for Yahoo Finance equities and indices."""
    Open: Series[float] = pa.Field(ge=0.0, nullable=False)
    High: Series[float] = pa.Field(ge=0.0, nullable=False)
    Low: Series[float] = pa.Field(ge=0.0, nullable=False)
    Close: Series[float] = pa.Field(ge=0.0, nullable=False)
    Volume: Series[float] = pa.Field(ge=0.0, nullable=False)
    Ticker: Series[str] = pa.Field(nullable=False)


class MacroDataSchema(CRISBaseSchema):
    """Standard TimeSeries schema for macroeconomic indicators."""
    Value: Series[float] = pa.Field(nullable=False)
