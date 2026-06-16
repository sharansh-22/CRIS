"""
entropy.py — Attribution entropy and concentration analysis.

Measures how concentrated or distributed informational value is
across the signal universe. Low entropy means a few signals dominate;
high entropy means many contribute.
"""

import numpy as np
from typing import Dict

from signal_attribution.schema import EntropyAnalysis


def compute_attribution_entropy(weights: Dict[str, float]) -> EntropyAnalysis:
    """Compute Shannon entropy and concentration metrics for the attribution distribution.

    Parameters
    ----------
    weights : dict
        Signal name -> attribution weight. Must sum to ~1.0.

    Returns
    -------
    EntropyAnalysis
        Typed entropy analysis result.
    """
    values = np.array(list(weights.values()))
    values = values[values > 0]  # Exclude zero weights for log safety
    n_signals = len(weights)

    if n_signals == 0:
        return EntropyAnalysis(
            attribution_entropy=0.0,
            max_possible_entropy=0.0,
            normalized_entropy=0.0,
            concentration_ratio_top3=0.0,
            concentration_ratio_top5=0.0,
            interpretation="No signals available.",
        )

    # Shannon entropy: H = -Σ p_i * log2(p_i)
    entropy = float(-np.sum(values * np.log2(values)))
    max_entropy = float(np.log2(n_signals))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    # Concentration ratios
    sorted_weights = sorted(weights.values(), reverse=True)
    top3 = sum(sorted_weights[:3])
    top5 = sum(sorted_weights[:5])

    # Interpretation
    if normalized > 0.85:
        interpretation = (
            "HIGHLY DISTRIBUTED: Information is spread broadly across many signals. "
            "No single signal dominates. The environmental state is multi-dimensional."
        )
    elif normalized > 0.70:
        interpretation = (
            "MODERATELY DISTRIBUTED: Several signals carry meaningful weight, "
            "but some differentiation exists. A core group of signals provides "
            "most of the information."
        )
    elif normalized > 0.50:
        interpretation = (
            "MODERATELY CONCENTRATED: A subset of signals provides most of the "
            "informational value. The environmental state has clear dominant dimensions."
        )
    else:
        interpretation = (
            "HIGHLY CONCENTRATED: Very few signals carry most of the information. "
            "The attribution landscape is sparse — a small number of environmental "
            "dimensions drive most of the credit deterioration signal."
        )

    return EntropyAnalysis(
        attribution_entropy=round(entropy, 4),
        max_possible_entropy=round(max_entropy, 4),
        normalized_entropy=round(normalized, 4),
        concentration_ratio_top3=round(top3, 4),
        concentration_ratio_top5=round(top5, 4),
        interpretation=interpretation,
    )
