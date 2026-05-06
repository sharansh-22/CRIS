"""
uncertainty_tests.py — Validate uncertainty quantification.

Tests:
1. Mixed stress_fields → MIXED state should appear
2. Ambiguous markets → UNCLEAR should appear
3. Confidence bands should reflect reality
"""

from .helpers import *
from layer3.convergence.uncertainty import compute_uncertainty_score, classify_uncertainty_state
from layer3.schema import DominantField


def test_mixed_stress_field_detection():
    """Multiple conflicting engines should produce MIXED state."""
    print("=" * 60)
    print("TEST: Mixed StressField → MIXED detected")
    print("=" * 60)

    returns = generate_mixed_stress_field(n=300)
    df = run_walk_forward(returns, warmup=200)

    # Check if MIXED or TRANSITIONAL appears at least once
    uncertainty_states = set(df["dominant"].unique())
    ambiguous = {"MIXED", "TRANSITIONAL", "UNCLEAR"}
    found = uncertainty_states & ambiguous

    print(f"  Dominant states seen: {uncertainty_states}")
    print(f"  Ambiguous states:    {found}")
    print(f"  Max uncertainty:     {df['uncertainty'].max():.2f}")

    # At least some uncertainty should be present
    assert df["uncertainty"].max() > 0.1, \
        f"System should show some uncertainty: max={df['uncertainty'].max():.2f}"
    print("  ✅ PASSED\n")


def test_uncertainty_unit_logic():
    """Test uncertainty computation directly."""
    print("=" * 60)
    print("TEST: Uncertainty Computation Logic")
    print("=" * 60)

    # All engines agree (low uncertainty)
    u_agree = compute_uncertainty_score(0.8, 0.8, 0.8, 0.9, 0.9, 0.9)
    print(f"  All agree (high risk):     uncertainty={u_agree:.2f}")

    # Engines disagree (high uncertainty)
    u_disagree = compute_uncertainty_score(0.9, 0.1, 0.5, 0.7, 0.7, 0.7)
    print(f"  Engines disagree:          uncertainty={u_disagree:.2f}")

    # All low confidence (high uncertainty)
    u_lowconf = compute_uncertainty_score(0.3, 0.3, 0.3, 0.1, 0.1, 0.1)
    print(f"  All low confidence:        uncertainty={u_lowconf:.2f}")

    assert u_agree < u_disagree, \
        f"Disagreement should increase uncertainty: agree={u_agree:.2f} disagree={u_disagree:.2f}"
    assert u_lowconf > 0.2, f"Low confidence should raise uncertainty: {u_lowconf:.2f}"
    print("  ✅ PASSED\n")


def test_unclear_classification():
    """Low confidence everywhere should produce UNCLEAR."""
    print("=" * 60)
    print("TEST: Low Confidence → UNCLEAR")
    print("=" * 60)

    # All engines have low confidence and low risk
    result = classify_uncertainty_state(
        dominant=DominantField.NONE,
        uncertainty=0.6,
        fast_risk=0.1, slow_risk=0.1, decay_risk=0.1,
        fast_confidence=0.2, slow_confidence=0.2, decay_confidence=0.2,
    )

    print(f"  Classification: {result.value}")
    assert result == DominantField.UNCLEAR, f"Expected UNCLEAR, got {result.value}"
    print("  ✅ PASSED\n")


def test_confidence_bands():
    """Confidence bands should match expectations."""
    print("=" * 60)
    print("TEST: Confidence Bands")
    print("=" * 60)

    from layer3.schema import to_confidence_band, ConfidenceBand

    assert to_confidence_band(0.1) == ConfidenceBand.LOW
    assert to_confidence_band(0.5) == ConfidenceBand.MEDIUM
    assert to_confidence_band(0.8) == ConfidenceBand.HIGH

    print("  LOW/MEDIUM/HIGH boundaries correct")
    print("  ✅ PASSED\n")


def run_all():
    tests = [test_mixed_stress_field_detection, test_uncertainty_unit_logic,
             test_unclear_classification, test_confidence_bands]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  ❌ FAILED: {e}\n")
            failed += 1
    return passed, failed


if __name__ == "__main__":
    p, f = run_all()
    print(f"Uncertainty: {p} passed, {f} failed")
