"""
recovery_tests.py — Validate recovery dynamics.

Tests:
1. Recovery after crash → probabilities should gradually normalize
2. Double-dip → recovery logic should detect failed recovery
3. System should NOT remain permanently fearful
"""

from .helpers import *


def test_recovery_relaxation():
    """Risk should decrease during sustained recovery."""
    print("=" * 60)
    print("TEST: Recovery → Risk relaxes")
    print("=" * 60)

    returns = generate_recovery(n=280, crash_at=200, recovery_at=230)
    df = run_walk_forward(returns, warmup=150)

    crash = df[(df.index >= len(df) - 60) & (df.index < len(df) - 30)]
    recovery = df.iloc[-20:]

    crash_risk = crash["overall_risk"].mean() if len(crash) > 0 else 0
    recovery_risk = recovery["overall_risk"].mean() if len(recovery) > 0 else 0

    print(f"  Crash risk:    {crash_risk:.2f}")
    print(f"  Recovery risk: {recovery_risk:.2f}")

    if crash_risk > 0.05:
        assert recovery_risk < crash_risk, \
            f"Risk should relax: crash={crash_risk:.2f} recovery={recovery_risk:.2f}"
    print("  ✅ PASSED\n")


def test_double_dip():
    """Double-dip should show non-trivial risk during second crash."""
    print("=" * 60)
    print("TEST: Double-Dip → Second crash detected")
    print("=" * 60)

    returns = generate_double_dip(n=350)
    df = run_walk_forward(returns, warmup=100)

    # During bounce period (should be lower risk)
    bounce_range = df.iloc[70:110]
    # During second crash (should re-escalate)
    second_crash = df.iloc[110:150]

    if len(bounce_range) > 0 and len(second_crash) > 0:
        bounce_risk = bounce_range["overall_risk"].mean()
        second_risk = second_crash["overall_risk"].mean()

        print(f"  Bounce risk:       {bounce_risk:.2f}")
        print(f"  Second crash risk: {second_risk:.2f}")

        # Risk should be non-trivial during second crash
        assert second_risk > 0.05, f"System should detect second crash: {second_risk}"
    print("  ✅ PASSED\n")


def test_not_permanently_fearful():
    """After full recovery, system should eventually normalize."""
    print("=" * 60)
    print("TEST: Not permanently fearful")
    print("=" * 60)

    # Long calm period after a crash
    returns = generate_recovery(n=350, crash_at=150, recovery_at=180)
    df = run_walk_forward(returns, warmup=100)

    # Very late in recovery (50+ days of calm)
    late_recovery = df.iloc[-30:]
    late_risk = late_recovery["overall_risk"].mean()
    late_fast = late_recovery["fast_risk"].mean()
    late_slow = late_recovery["slow_risk"].mean()

    print(f"  Late overall risk: {late_risk:.2f}")
    print(f"  Late fast risk:    {late_fast:.2f}")
    print(f"  Late slow risk:    {late_slow:.2f}")

    # Fast should relax quickly (5d horizon)
    assert late_fast < 0.45, f"FAST still too high after long recovery: {late_fast}"
    # Slow relaxes slower by design (30d horizon), but should still come down
    assert late_slow < 0.50, f"SLOW still too high after long recovery: {late_slow}"
    # Overall risk should be moderate-to-low
    assert late_risk < 0.40, f"Overall risk too high after long recovery: {late_risk}"
    print("  ✅ PASSED\n")


def run_all():
    tests = [test_recovery_relaxation, test_double_dip, test_not_permanently_fearful]
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
    print(f"Recovery: {p} passed, {f} failed")
