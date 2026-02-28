"""
fred_macro_puller.py — Federal Reserve Economic Data Ingestion

Pulls macroeconomic time-series (yield curves, credit spreads,
unemployment claims, etc.) from the FRED API for use as leading
indicators of systemic stress.
"""


def pull_fred_series(series_ids: list[str], start: str, end: str) -> None:
    """Download one or more FRED time-series."""
    pass


def compute_yield_curve_slope() -> None:
    """Compute the 10Y-2Y yield curve slope from FRED data."""
    pass
