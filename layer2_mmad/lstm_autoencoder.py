"""
lstm_autoencoder.py — LSTM Autoencoder for Sequence Anomaly Detection

Trains an LSTM-based autoencoder on "normal" microstructure
sequences and flags high-reconstruction-error windows as anomalous.
"""


def build_autoencoder(input_dim: int, seq_len: int) -> None:
    """Construct the LSTM autoencoder architecture."""
    pass


def detect_sequence_anomalies(model, sequences, threshold: float = 2.0) -> list:
    """Return indices of anomalous sequences."""
    pass
