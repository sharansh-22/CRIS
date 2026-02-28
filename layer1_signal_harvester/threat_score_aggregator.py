"""
threat_score_aggregator.py — Layer 1 Threat Score Aggregation

Fuses outputs from GDELT, FRED, FinBERT, and the geopolitical
index into a single Layer 1 threat score that is passed to the
Convergence Engine.
"""


def aggregate_threat_score(
    geopolitical_score: float,
    macro_score: float,
    sentiment_score: float,
) -> float:
    """Produce a weighted composite threat score for Layer 1."""
    pass
