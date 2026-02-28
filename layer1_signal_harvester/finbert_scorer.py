"""
finbert_scorer.py — FinBERT Sentiment Scoring

Applies FinBERT (a financial-domain BERT model) to news headlines
and earnings transcripts to produce sentiment scores that feed
into the threat-score aggregator.
"""


def score_headlines(headlines: list[str]) -> list[float]:
    """Return sentiment scores for a batch of headlines."""
    pass


def score_transcript(transcript: str) -> float:
    """Return an aggregate sentiment score for an earnings transcript."""
    pass
