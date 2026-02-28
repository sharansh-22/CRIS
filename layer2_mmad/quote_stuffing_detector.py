"""
quote_stuffing_detector.py — Quote Stuffing Detection

Detects abnormal bursts of order submissions and cancellations
(quote stuffing) that may indicate algorithmic manipulation or
deteriorating market quality.
"""


def detect_quote_stuffing(messages, window_ms: int = 100) -> list:
    """Flag time windows with abnormal message rates."""
    pass
