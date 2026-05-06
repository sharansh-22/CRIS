"""
behavioral_suite.py — Master validation runner for Layer 3 probabilistic interpretation framework.

Runs all 4 test suites, then generates visualization outputs:
  - Probability evolution plots
  - Stress interpretation timelines
  - Confidence smoothing visualization
  - Recovery behavior visualization
  - Inter-layer influence diagnostics
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from layer3.validation.helpers import (
    generate_sudden_shock, generate_persistent_crisis, generate_slow_grind,
    generate_recovery, generate_double_dip, generate_oscillating,
    run_walk_forward, returns_to_prices,
)
from layer3.validation import stress_interpretation_tests
from layer3.validation import recovery_tests
from layer3.validation import oscillation_tests
from layer3.validation import uncertainty_tests


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "validation" / "output"


# ──────────────────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────────────────

DARK_BG = "#0d1117"
DARK_PANEL = "#161b22"
FAST_COLOR = "#f97583"
SLOW_COLOR = "#58a6ff"
DECAY_COLOR = "#56d364"
OVERALL_COLOR = "#e3b341"
TEXT_COLOR = "#c9d1d9"


def _setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_PANEL,
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "grid.color": "#21262d",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 10,
    })


def plot_probability_evolution(df: pd.DataFrame, title: str, filename: str):
    """Plot risk probability evolution for all three engines + overall."""
    _setup_dark_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    x = range(len(df))

    # Panel 1: Risk probabilities
    ax = axes[0]
    ax.plot(x, df["fast_risk"], color=FAST_COLOR, label="FAST", linewidth=1.5)
    ax.plot(x, df["slow_risk"], color=SLOW_COLOR, label="SLOW", linewidth=1.5)
    ax.plot(x, df["decay_risk"], color=DECAY_COLOR, label="DECAY", linewidth=1.5)
    ax.plot(x, df["overall_risk"], color=OVERALL_COLOR, label="Overall", linewidth=2, linestyle="--")
    ax.set_ylabel("Risk Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)
    ax.set_title("Stress Intensity Evolution", fontsize=11)

    # Panel 2: Confidence
    ax = axes[1]
    ax.plot(x, df["fast_conf"], color=FAST_COLOR, alpha=0.7, label="FAST conf")
    ax.plot(x, df["slow_conf"], color=SLOW_COLOR, alpha=0.7, label="SLOW conf")
    ax.plot(x, df["decay_conf"], color=DECAY_COLOR, alpha=0.7, label="DECAY conf")
    ax.set_ylabel("Confidence")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)
    ax.set_title("Engine Confidence Evolution", fontsize=11)

    # Panel 3: Uncertainty + Dominant stress_field
    ax = axes[2]
    ax.plot(x, df["uncertainty"], color="#bc8cff", linewidth=1.5, label="Uncertainty")
    ax.set_ylabel("Uncertainty")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)
    ax.set_title("Uncertainty Pressure", fontsize=11)

    # Color background by dominant stress_field
    stress_field_colors = {
        "NONE": "#0d111700", "FAST_SHOCK": FAST_COLOR + "20",
        "SLOW_STRUCTURAL": SLOW_COLOR + "20", "TRAJECTORY_DEGRADATION": DECAY_COLOR + "20",
        "MIXED": "#bc8cff20", "TRANSITIONAL": "#e3b34120", "UNCLEAR": "#8b949e20",
    }
    for i in range(len(df) - 1):
        dom = df["dominant"].iloc[i]
        color = stress_field_colors.get(dom, "#00000000")
        ax.axvspan(i, i + 1, color=color, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_recovery_behavior(df: pd.DataFrame, title: str, filename: str):
    """Plot recovery dynamics using stabilization_strength."""
    _setup_dark_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    x = range(len(df))

    ax = axes[0]
    ax.plot(x, df["overall_risk"], color=OVERALL_COLOR, linewidth=2, label="Overall Risk")
    ax.axhline(y=0.30, color="#f97583", linestyle="--", alpha=0.5, label="Recovery threshold")
    ax.set_ylabel("Risk")
    ax.legend(fontsize=8)
    ax.grid(True)
    ax.set_title("Risk with Recovery Relaxation", fontsize=11)

    ax = axes[1]
    ax.plot(x, df["stab_strength"], color=DECAY_COLOR, linewidth=1.5, label="Stabilization Strength")
    ax.plot(x, df["coherence"], color=SLOW_COLOR, linewidth=1.5, alpha=0.7, label="Signal Coherence")
    ax.set_ylabel("Meta Dynamics")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True)
    ax.set_title("Recovery & Coherence Dynamics", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_smoothing(df: pd.DataFrame, title: str, filename: str):
    """Plot confidence band evolution."""
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    x = range(len(df))
    ax.plot(x, df["fast_conf"], color=FAST_COLOR, alpha=0.7, label="FAST conf")
    ax.plot(x, df["slow_conf"], color=SLOW_COLOR, alpha=0.7, label="SLOW conf")
    ax.plot(x, df["decay_conf"], color=DECAY_COLOR, alpha=0.7, label="DECAY conf")
    ax.plot(x, df["overall_conf"], color=OVERALL_COLOR, linewidth=2, label="Overall conf")

    # Band boundaries
    ax.axhline(y=0.35, color="#8b949e", linestyle=":", alpha=0.5)
    ax.axhline(y=0.65, color="#8b949e", linestyle=":", alpha=0.5)
    ax.text(len(df) - 1, 0.20, "LOW", color="#8b949e", fontsize=8)
    ax.text(len(df) - 1, 0.48, "MEDIUM", color="#8b949e", fontsize=8)
    ax.text(len(df) - 1, 0.75, "HIGH", color="#8b949e", fontsize=8)

    ax.set_ylabel("Confidence")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────
#  Main Runner
# ──────────────────────────────────────────────────────────

def run_all_tests():
    print("\n" + "━" * 60)
    print("  LAYER 3 PROBABILISTIC INTERPRETATION — FULL VALIDATION SUITE")
    print("━" * 60 + "\n")

    total_passed = 0
    total_failed = 0

    for name, suite in [
        ("STRESS INTERPRETATIONS", stress_interpretation_tests),
        ("RECOVERY DYNAMICS", recovery_tests),
        ("OSCILLATION STABILITY", oscillation_tests),
        ("UNCERTAINTY", uncertainty_tests),
    ]:
        print(f"\n{'─' * 60}")
        print(f"  SUITE: {name}")
        print(f"{'─' * 60}\n")
        p, f = suite.run_all()
        total_passed += p
        total_failed += f

    print("\n" + "━" * 60)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed, {total_passed + total_failed} total")
    print("━" * 60 + "\n")
    return total_failed == 0


def generate_all_visualizations():
    print("Generating visualization outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("Sudden Shock", generate_sudden_shock(n=280, shock_day=260), "sudden_shock"),
        ("Persistent Crisis", generate_persistent_crisis(n=340, crisis_start=260), "persistent_crisis"),
        ("Slow Grind-Down", generate_slow_grind(n=400, grind_start=150), "slow_grind"),
        ("Recovery", generate_recovery(n=280, crash_at=200, recovery_at=230), "recovery"),
        ("Double-Dip", generate_double_dip(n=350), "double_dip"),
        ("Oscillating", generate_oscillating(n=400), "oscillating"),
    ]

    for title, returns, tag in scenarios:
        print(f"  → {title}...")
        warmup = min(200, len(returns) // 2)
        df = run_walk_forward(returns, warmup=warmup)
        if len(df) > 0:
            plot_probability_evolution(df, f"Probability Evolution — {title}", f"prob_evolution_{tag}.png")
            plot_confidence_smoothing(df, f"Confidence Smoothing — {title}", f"confidence_{tag}.png")
            if tag in ("recovery", "double_dip"):
                plot_recovery_behavior(df, f"Recovery Dynamics — {title}", f"recovery_{tag}.png")

    print(f"  All visualizations saved to {OUTPUT_DIR}\n")


if __name__ == "__main__":
    success = run_all_tests()
    generate_all_visualizations()
    sys.exit(0 if success else 1)
