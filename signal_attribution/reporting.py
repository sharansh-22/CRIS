"""
reporting.py — Visual and textual reporting for Signal Attribution Engine.

Generates publication-quality charts and a structured text report.
All outputs are saved to the designated output directory.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import logging
from pathlib import Path
from typing import List, Dict

from signal_attribution.schema import AttributionReport

logger = logging.getLogger("CRIS.SAE.reporting")

# ── Visual configuration ──
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.size": 10,
})

PALETTE = [
    "#58a6ff", "#3fb950", "#d29922", "#f85149",
    "#bc8cff", "#79c0ff", "#56d364", "#e3b341",
    "#ff7b72", "#d2a8ff", "#a5d6ff", "#7ee787",
    "#f0883e", "#db6d28", "#f778ba", "#b392f0",
    "#39d353", "#26a641",
]


def plot_attribution_ranking(report: AttributionReport, output_dir: Path) -> Path:
    """Horizontal bar chart of signal attribution weights, ranked."""
    signals = report.signals
    names = [s.signal_name for s in signals]
    weights = [s.attribution_weight for s in signals]
    sources = [s.source for s in signals]

    # Color by source
    source_colors = {
        "Layer3.Fast": "#f85149",
        "Layer3.Slow": "#d29922",
        "Layer3.Decay": "#bc8cff",
        "Layer3.Meta": "#58a6ff",
        "MarketStructure": "#3fb950",
        "Composite": "#79c0ff",
    }
    colors = [source_colors.get(s, "#8b949e") for s in sources]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, weights, color=colors, edgecolor="#30363d", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution Weight", fontsize=11)
    ax.set_title("CRIS Signal Attribution Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, w in zip(bars, weights):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.1%}", va="center", fontsize=8, color="#c9d1d9")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in source_colors.items() if l in set(sources)]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    path = output_dir / "attribution_ranking.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved attribution ranking chart → {path}")
    return path


def plot_attribution_through_time(report: AttributionReport, output_dir: Path) -> Path:
    """Stacked area chart showing how attribution shifts across temporal windows."""
    windows = report.temporal_windows
    if not windows:
        return output_dir / "attribution_through_time.png"

    # Build DataFrame: rows = windows, columns = signals
    labels = [w.window_label for w in windows]
    all_signals = list(windows[0].signal_weights.keys())

    data = pd.DataFrame(index=labels)
    for signal in all_signals:
        data[signal] = [w.signal_weights.get(signal, 0.0) for w in windows]

    # Sort signals by overall weight for visual clarity
    global_weights = {s.signal_name: s.attribution_weight for s in report.signals}
    sorted_signals = sorted(all_signals, key=lambda s: global_weights.get(s, 0), reverse=True)
    data = data[sorted_signals]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = PALETTE[:len(sorted_signals)]
    data.plot.bar(stacked=True, ax=ax, color=colors, edgecolor="#30363d", linewidth=0.3, width=0.65)

    ax.set_ylabel("Attribution Weight", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Signal Attribution Through Time", fontsize=14, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, framealpha=0.7)
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "attribution_through_time.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved attribution through time chart → {path}")
    return path


def plot_stability_analysis(report: AttributionReport, output_dir: Path) -> Path:
    """Scatter plot: temporal stability vs. regime stability, sized by weight."""
    signals = report.signals

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, s in enumerate(signals):
        size = max(s.attribution_weight * 2000, 30)
        ax.scatter(
            s.temporal_stability, s.regime_stability,
            s=size, c=PALETTE[i % len(PALETTE)], alpha=0.8,
            edgecolors="#c9d1d9", linewidth=0.5, zorder=3,
        )
        ax.annotate(
            s.signal_name, (s.temporal_stability, s.regime_stability),
            fontsize=7, ha="center", va="bottom",
            xytext=(0, 8), textcoords="offset points",
            color="#c9d1d9",
        )

    ax.set_xlabel("Temporal Stability", fontsize=11)
    ax.set_ylabel("Regime Stability", fontsize=11)
    ax.set_title("Signal Stability Analysis\n(bubble size ∝ attribution weight)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="#30363d", linestyle="--", alpha=0.5)
    ax.axvline(0.5, color="#30363d", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = output_dir / "stability_analysis.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved stability analysis chart → {path}")
    return path


def plot_correlation_heatmap(report: AttributionReport, output_dir: Path) -> Path:
    """Bar chart of raw correlation strength per signal."""
    signals = report.signals
    names = [s.signal_name for s in signals]
    corrs = [s.correlation_strength for s in signals]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars = ax.bar(range(len(names)), corrs, color=colors, edgecolor="#30363d", linewidth=0.5, width=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|Spearman ρ| with Default Rate", fontsize=10)
    ax.set_title("Signal Correlation with Credit Deterioration", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    for bar, c in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{c:.3f}", ha="center", fontsize=7, color="#c9d1d9")

    plt.tight_layout()
    path = output_dir / "correlation_strength.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved correlation strength chart → {path}")
    return path


def generate_text_report(report: AttributionReport, validation_results: Dict, output_dir: Path) -> Path:
    """Generate a comprehensive markdown research report."""
    lines = []
    lines.append("# CRIS Signal Attribution Engine — Research Report V1\n")
    lines.append("---\n")

    # Part 1: Attribution Distribution
    lines.append("## 1. Final Attribution Distribution\n")
    lines.append("| Rank | Signal | Source | Weight | Correlation | AUC Lift | Temporal Stab. | Regime Stab. |")
    lines.append("|------|--------|--------|--------|-------------|----------|----------------|--------------|")
    for i, s in enumerate(report.signals, 1):
        lines.append(
            f"| {i} | {s.signal_name} | {s.source} | {s.attribution_weight:.4f} | "
            f"{s.correlation_strength:.4f} | {s.predictive_lift_auc:+.6f} | "
            f"{s.temporal_stability:.2f} | {s.regime_stability:.2f} |"
        )

    cumulative = 0.0
    lines.append(f"\n**Σ weights = {sum(s.attribution_weight for s in report.signals):.4f}**\n")

    # Part 2: Entropy
    lines.append("## 2. Attribution Entropy Analysis\n")
    e = report.entropy
    lines.append(f"- **Shannon Entropy**: {e.attribution_entropy:.4f} bits")
    lines.append(f"- **Maximum Entropy**: {e.max_possible_entropy:.4f} bits (uniform over {len(report.signals)} signals)")
    lines.append(f"- **Normalized Entropy**: {e.normalized_entropy:.4f}")
    lines.append(f"- **Top-3 Concentration**: {e.concentration_ratio_top3:.1%}")
    lines.append(f"- **Top-5 Concentration**: {e.concentration_ratio_top5:.1%}")
    lines.append(f"- **Interpretation**: {e.interpretation}\n")

    # Part 3: Temporal Windows
    lines.append("## 3. Attribution Through Time\n")
    for w in report.temporal_windows:
        lines.append(f"### {w.window_label}")
        lines.append(f"- Loans: {w.n_loans:,} | Defaults: {w.n_defaults:,} | Default Rate: {w.default_rate:.2%}")
        top5 = sorted(w.signal_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for sig, wt in top5:
            lines.append(f"  - {sig}: {wt:.4f}")
        lines.append("")

    # Part 4: Validation
    lines.append("## 4. Walk-Forward Validation\n")
    for check, result in validation_results.items():
        icon = "✓" if "PASS" in result else ("⚠" if "WARNING" in result else "✗")
        lines.append(f"- {icon} **{check}**: {result}")

    lines.append(f"\n**Overall Status**: `{report.validation_status}`\n")

    # Part 5: Dataset Summary
    lines.append("## 5. Dataset Summary\n")
    lines.append(f"- Total Loans: {report.n_total_loans:,}")
    lines.append(f"- Total Defaults: {report.n_total_defaults:,}")
    lines.append(f"- Overall Default Rate: {report.overall_default_rate:.2%}\n")

    report_text = "\n".join(lines)
    path = output_dir / "signal_attribution_report.md"
    path.write_text(report_text)
    logger.info(f"Saved text report → {path}")
    return path
