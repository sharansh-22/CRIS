"""
oscillation_tests.py — Validate stability under oscillating conditions.

Tests:
1. Fake volatility spikes → system should remain stable
2. Oscillating markets → probabilities should remain bounded
"""

from .helpers import *


def test_fake_spike_stability():
    """Single-day spikes should NOT cause escalation."""
    print("=" * 60)
    print("TEST: Fake Spike → Stability")
    print("=" * 60)

    returns = generate_fake_spike(n=280, spike_day=260)
    df = run_walk_forward(returns, warmup=200)

    post = df.iloc[-15:]
    max_risk = post["overall_risk"].max()
    late_risk = post.iloc[-5:]["overall_risk"].mean()

    print(f"  Max risk after spike: {max_risk:.2f}")
    print(f"  Risk 5d later:       {late_risk:.2f}")

    assert late_risk < 0.5, f"System should stabilize: {late_risk}"
    print("  ✅ PASSED\n")


def test_oscillating_stability():
    """Oscillating markets should NOT cause unbounded risk."""
    print("=" * 60)
    print("TEST: Oscillating Market → Bounded risk")
    print("=" * 60)

    returns = generate_oscillating(n=400)
    df = run_walk_forward(returns, warmup=100)

    # Risk should not permanently ratchet upward
    first_half = df.iloc[:len(df)//2]
    second_half = df.iloc[len(df)//2:]

    first_mean = first_half["overall_risk"].mean()
    second_mean = second_half["overall_risk"].mean()
    max_risk = df["overall_risk"].max()

    print(f"  First half avg risk:  {first_mean:.2f}")
    print(f"  Second half avg risk: {second_mean:.2f}")
    print(f"  Max risk:             {max_risk:.2f}")

    # Risk should not systematically increase
    assert second_mean < first_mean + 0.15, \
        f"Risk ratcheting up: first={first_mean:.2f} second={second_mean:.2f}"
    # Risk should remain bounded
    assert max_risk < 0.85, f"Risk unbounded: {max_risk}"
    print("  ✅ PASSED\n")


def test_no_runaway_feedback():
    """Partner influence should be bounded and non-circular."""
    print("=" * 60)
    print("TEST: No Runaway Feedback Loops")
    print("=" * 60)

    from harvesters.macro.convergence.arbitration import apply_partner_influence, validate_no_circular_feedback

    cases = [
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ]

    for fr, fc, sr, sc, dr, dc in cases:
        af, as_, ad = apply_partner_influence(fr, fc, sr, sc, dr, dc)

        assert validate_no_circular_feedback(fr, sr, dr, af, as_, ad), \
            f"Circular feedback for ({fr}, {sr}, {dr})"
        assert abs(af - fr) < 1e-6, f"Fast modified: {fr} → {af}"
        assert abs(as_ - sr) <= 0.10, f"Slow too large: {sr} → {as_}"
        assert abs(ad - dr) <= 0.10, f"Decay too large: {dr} → {ad}"

    print("  All bounds validated")
    print("  ✅ PASSED\n")


def run_all():
    tests = [test_fake_spike_stability, test_oscillating_stability, test_no_runaway_feedback]
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
    print(f"Oscillation: {p} passed, {f} failed")
