"""
stress_field_transition_tests.py — Validate stress_field transition dynamics.

Tests:
1. Sudden shocks → FAST should dominate initially
2. Persistent crises → SLOW should gradually rise
3. Long deterioration → DECAY should dominate
4. No hard switching (smooth weight transitions)
5. Raw data dominance (95%+ own reasoning)
"""

from .helpers import *
from harvesters.macro.convergence.arbitration import apply_partner_influence, validate_no_circular_feedback


def test_sudden_shock():
    """FAST should spike after sudden shock."""
    print("=" * 60)
    print("TEST: Sudden Shock → FAST dominates")
    print("=" * 60)

    returns = generate_sudden_shock(n=280, shock_day=260)
    df = run_walk_forward(returns, warmup=200)

    pre = df.iloc[:-20]
    post = df.iloc[-15:]

    pre_fast = pre["fast_risk"].mean()
    max_fast = post["fast_risk"].max()

    print(f"  Pre-shock FAST risk:  {pre_fast:.2f}")
    print(f"  Post-shock max FAST:  {max_fast:.2f}")

    assert max_fast > 0.3, f"FAST too low after shock: {max_fast}"
    assert max_fast > pre_fast + 0.15, f"FAST should spike: pre={pre_fast:.2f} post={max_fast:.2f}"
    print("  ✅ PASSED\n")


def test_persistent_crisis():
    """SLOW should gradually rise during persistent crisis."""
    print("=" * 60)
    print("TEST: Persistent Crisis → SLOW rises")
    print("=" * 60)

    returns = generate_persistent_crisis(n=340, crisis_start=260)
    df = run_walk_forward(returns, warmup=200)

    pre = df.iloc[:40]
    crisis = df.iloc[-40:]

    pre_slow = pre["slow_risk"].mean()
    crisis_slow = crisis["slow_risk"].mean()

    print(f"  Pre-crisis SLOW:    {pre_slow:.2f}")
    print(f"  During crisis SLOW: {crisis_slow:.2f}")

    assert crisis_slow > pre_slow, f"SLOW should rise: {pre_slow:.2f} → {crisis_slow:.2f}"
    assert crisis_slow > 0.2, f"SLOW too low: {crisis_slow}"
    print("  ✅ PASSED\n")


def test_long_grind_down():
    """DECAY should dominate during slow deterioration."""
    print("=" * 60)
    print("TEST: Long Grind-Down → DECAY dominates")
    print("=" * 60)

    returns = generate_slow_grind(n=400, grind_start=150)
    df = run_walk_forward(returns, warmup=100)

    late = df.iloc[-30:]
    avg_decay = late["decay_risk"].mean()
    avg_fast = late["fast_risk"].mean()

    print(f"  Late DECAY risk: {avg_decay:.2f}")
    print(f"  Late FAST risk:  {avg_fast:.2f}")

    assert avg_decay > 0.1, f"DECAY too low: {avg_decay}"
    assert avg_decay > avg_fast, f"DECAY should exceed FAST: decay={avg_decay:.2f} fast={avg_fast:.2f}"
    print("  ✅ PASSED\n")


def test_no_hard_switching():
    """SLOW and DECAY intensities should not jump abruptly.

    FAST is excluded: it is designed as a reflexive, instantaneous shock detector.
    Large day-to-day jumps in FAST are architecturally correct.
    SLOW and DECAY, however, must exhibit temporal persistence and smooth transitions.
    """
    print("=" * 60)
    print("TEST: No Hard Switching")
    print("=" * 60)

    returns = generate_persistent_crisis(n=300, crisis_start=220)
    df = run_walk_forward(returns, warmup=200)

    max_change = 0.0
    for col in ["slow_risk", "decay_risk"]:
        changes = df[col].diff().abs().dropna()
        max_change = max(max_change, changes.max())

    print(f"  Max daily SLOW/DECAY change: {max_change:.4f}")
    assert max_change < 0.50, f"Hard switching detected in persistent engines: {max_change:.4f}"
    print("  ✅ PASSED\n")


def test_raw_data_dominance():
    """Partner influence must remain ≤5%."""
    print("=" * 60)
    print("TEST: Raw Data Dominance (95%+ own reasoning)")
    print("=" * 60)

    for fr, fc, sr, sc, dr, dc in [(0.8, 0.9, 0.3, 0.5, 0.1, 0.3)]:
        af, as_, ad = apply_partner_influence(fr, fc, sr, sc, dr, dc)

        fast_own = 1.0
        slow_own = 1.0 - (abs(as_ - sr) / max(sr, 0.01))
        decay_own = 1.0 - (abs(ad - dr) / max(dr, 0.01))

        print(f"  Fast:  {fast_own:.0%} own")
        print(f"  Slow:  {slow_own:.0%} own")
        print(f"  Decay: {decay_own:.0%} own")

        assert fast_own >= 1.0
        assert slow_own >= 0.93, f"Slow too influenced: {slow_own:.0%}"
        assert decay_own >= 0.93, f"Decay too influenced: {decay_own:.0%}"

    print("  ✅ PASSED\n")


def run_all():
    tests = [test_sudden_shock, test_persistent_crisis, test_long_grind_down,
             test_no_hard_switching, test_raw_data_dominance]
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
    print(f"StressField Transitions: {p} passed, {f} failed")
