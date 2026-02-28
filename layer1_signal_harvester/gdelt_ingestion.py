"""
gdelt_ingestion.py — GDELT Event Stream Ingestion

Connects to the GDELT 2.0 Event Database to pull geopolitical
event records, filtering for conflict, protest, and economic-
instability themes relevant to tail-risk detection.
"""


def fetch_gdelt_events(start_date: str, end_date: str) -> None:
    """Download GDELT events for the given date range."""
    pass


def filter_risk_events(events, theme_codes: list[str] | None = None) -> None:
    """Filter raw GDELT events to retain only risk-relevant themes."""
    pass
