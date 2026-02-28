"""
hmm_regime_classifier.py — Hidden Markov Model Regime Classification

Fits a Gaussian HMM to microstructure features to classify the
market into latent regimes (e.g., calm, stressed, crisis).
"""


def fit_hmm(features, n_states: int = 3) -> None:
    """Fit a Gaussian HMM and return the model."""
    pass


def classify_regime(model, features) -> int:
    """Return the most likely current regime label."""
    pass
