"""
entropy_comparison.py — Empirical Validation of Entropy Methods

This script runs a full empirical comparison of four entropy methods:
  - Shannon Entropy
  - Permutation Entropy
  - Sample Entropy
  - Tsallis Entropy

It evaluates them across three historical market events using six metrics:
  1. Crisis Lead Time (days)
  2. False Positive Rate (breaches per calm month)
  3. Signal-to-Noise Ratio
  4. Directional Consistency
  5. Threshold Breach Duration
  6. Inter-Method Correlation

Produces three plots and outputs the raw data for `final_test_results.md`.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path

# Use Agg backend for headless generation
if os.environ.get("CI") or (os.name == "posix" and not os.environ.get("DISPLAY")):
    matplotlib.use("Agg")

from layer3_bssc.engine.entropy import (
    compute_rolling_multi_entropy,
    _load_ohlcv,
    run_entropy_method_selection
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "simulation_output"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EVENTS = {
    "COVID Crash": {"start": "2020-02-01", "end": "2020-03-31", "color": "#f85149"},
    "Q4 2018 Selloff": {"start": "2018-10-01", "end": "2018-12-31", "color": "#f0883e"},
    "2022 Bear Market": {"start": "2022-01-01", "end": "2022-06-30", "color": "#d29922"}
}

METHODS = ["shannon", "permutation", "sample", "tsallis"]

COLORS = {
    "shannon": "#f0883e",
    "permutation": "#3fb950",
    "sample": "#2f81f7",
    "tsallis": "#d29922",
    "bg": "#0d0d0d",
    "grid": "#21262d",
    "text": "#c9d1d9",
    "price": "#58a6ff"
}

def is_calm_period(date_idx, events_dict):
    """Check if a date is outside 60 days of any event window."""
    for ev in events_dict.values():
        e_start = pd.Timestamp(ev["start"]) - pd.Timedelta(days=60)
        e_end = pd.Timestamp(ev["end"]) + pd.Timedelta(days=60)
        if e_start <= date_idx <= e_end:
            return False
    return True

# ---------------------------------------------------------------------------
# Metric Computations
# ---------------------------------------------------------------------------

def compute_metrics(csv_path: str):
    print("Loading data and computing rolling entropies...")
    df = _load_ohlcv(csv_path)
    close = df["Close"].dropna().astype(float)
    log_returns = np.log(close / close.shift(1)).dropna()
    log_returns = log_returns.loc[~log_returns.index.duplicated(keep="first")]
    
    # Needs to be 30 day rolling to match simulation
    rolling_df = compute_rolling_multi_entropy(log_returns, window=30)
    
    print("Defining calm periods and baselines...")
    calm_mask = rolling_df.index.map(lambda d: is_calm_period(d, EVENTS))
    calm_df = rolling_df.loc[calm_mask].dropna()
    
    if len(calm_df) == 0:
        calm_df = rolling_df.dropna().iloc[:252]
        
    baselines = calm_df.mean()
    calm_months = max(1, len(calm_df) / 21.0)
    
    # Metric 2: False Positive Rate
    fp_rate = {}
    for m in METHODS:
        breaches = (calm_df[m] > baselines[m] + 0.15).sum()
        fp_rate[m] = breaches / calm_months

    lead_times = {m: [] for m in METHODS}
    snrs = {m: [] for m in METHODS}
    directions = {m: [] for m in METHODS}
    breach_durations = {m: [] for m in METHODS}
    corr_matrices = []

    print("Evaluating events...")
    for ev_name, ev_dates in EVENTS.items():
        start = pd.Timestamp(ev_dates["start"])
        end = pd.Timestamp(ev_dates["end"])
        
        ev_df = rolling_df.loc[start:end]
        ev_close = close.loc[start:end]
        
        if len(ev_df) == 0:
            continue
            
        corr_matrices.append(ev_df.corr())
        min_date = ev_close.idxmin()
        
        # Pre-event data for directionality (30 days before start)
        pre_start = start - pd.Timedelta(days=45) # grab extra to ensure 30 trading days
        pre_df = rolling_df.loc[pre_start:start].iloc[-30:] if len(rolling_df.loc[:start]) > 0 else calm_df
        pre_means = pre_df.mean()
        
        for m in METHODS:
            m_base = baselines[m]
            m_thresh = m_base + 0.15
            
            # Metric 3: SNR
            m_ev_mean = ev_df[m].mean()
            snr = m_ev_mean / (m_base if m_base > 0 else 1.0)
            snrs[m].append(snr)
            
            # Metric 4 inputs: Direction
            if m_ev_mean > pre_means[m]:
                directions[m].append(1)  # UP
            else:
                directions[m].append(-1) # DOWN
                
            # Metric 1: Lead Time
            pre_min_df = ev_df.loc[:min_date]
            breaches = pre_min_df[pre_min_df[m] > m_thresh]
            if len(breaches) > 0:
                first_cross = breaches.index[0]
                days = len(ev_close.loc[first_cross:min_date]) - 1
                lead_times[m].append(max(0, days))
            else:
                lead_times[m].append(0)
                
            # Metric 5: Breach Duration
            ev_breaches = ev_df[m] > m_thresh
            if not ev_breaches.any():
                breach_durations[m].append(0)
            else:
                # Count consecutive True values
                consec = ev_breaches * (ev_breaches.groupby((ev_breaches != ev_breaches.shift()).cumsum()).cumcount() + 1)
                breach_durations[m].append(consec.max())

    # Aggregate means
    avg_lt = {m: np.mean(lead_times[m]) if len(lead_times[m]) > 0 else 0 for m in METHODS}
    avg_snr = {m: np.mean(snrs[m]) if len(snrs[m]) > 0 else 0 for m in METHODS}
    avg_dur = {m: np.mean(breach_durations[m]) if len(breach_durations[m]) > 0 else 0 for m in METHODS}
    
    # Compute consistency
    consistency = {}
    for m in METHODS:
        if not directions[m]:
            consistency[m] = 0.0
            continue
        from collections import Counter
        modal = Counter(directions[m]).most_common(1)[0][1]
        consistency[m] = modal / len(directions[m])
        
    avg_corr = sum(corr_matrices) / len(corr_matrices) if len(corr_matrices) > 0 else pd.DataFrame()

    def norm(d, invert=False):
        vals = list(d.values())
        if max(vals) == min(vals):
            return {m: 1.0 for m in METHODS}
        if invert:
            return {m: (max(vals) - d[m]) / (max(vals) - min(vals)) for m in METHODS}
        return {m: (d[m] - min(vals)) / (max(vals) - min(vals)) for m in METHODS}

    n_lt = norm(avg_lt)
    n_fp = norm(fp_rate, invert=True)
    n_snr = norm(avg_snr)
    n_dur = norm(avg_dur)
    
    scores = {}
    for m in METHODS:
        scores[m] = (
            0.35 * n_lt[m] +
            0.30 * n_fp[m] +
            0.15 * n_snr[m] +
            0.10 * consistency[m] +
            0.10 * n_dur[m]
        )
        
    sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner = sorted_methods[0][0]
    
    runner_up = None
    for m, sc in sorted_methods[1:]:
        c = avg_corr.loc[winner, m] if not avg_corr.empty else 0.0
        if c < 0.70:
            runner_up = m
            break
            
    if not runner_up and len(sorted_methods) > 1:
        runner_up = sorted_methods[1][0]
        
    metrics = {
        "lead_time": avg_lt,
        "fp_rate": fp_rate,
        "snr": avg_snr,
        "consistency": consistency,
        "duration": avg_dur,
        "scores": scores,
        "winner": winner,
        "runner_up": runner_up,
        "correlation": avg_corr
    }
    
    return close, rolling_df, baselines, metrics

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def style_ax(ax):
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.grid(True, alpha=0.15, color=COLORS["grid"], linestyle="--")

def plot_overview(close, rolling_df, baselines):
    fig, axes = plt.subplots(6, 1, figsize=(20, 24), sharex=True, dpi=150)
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in axes: style_ax(ax)
    
    ax = axes[0]
    ax.plot(close.index, close.values, color=COLORS["price"], linewidth=1.2)
    ax.set_ylabel("Price", color=COLORS["text"])
    ax.set_title("Entropy Method Comparison — CRIS Empirical Validation", color=COLORS["text"], fontsize=16, fontweight="bold", pad=15)
    
    sub_titles = ["Shannon Entropy", "Permutation Entropy", "Sample Entropy", "Tsallis Entropy"]
    for i, m in enumerate(METHODS):
        ax = axes[i+1]
        ax.plot(rolling_df.index, rolling_df[m], color=COLORS[m], linewidth=1.2)
        ax.axhline(baselines[m], color="white", linestyle="--", alpha=0.5, label="Baseline")
        ax.axhline(baselines[m]+0.15, color="#d29922", linestyle="--", label="Stress")
        ax.axhline(baselines[m]+0.30, color="#f85149", linestyle="--", label="Black Swan")
        ax.set_ylabel(m.capitalize(), color=COLORS["text"])
        ax.set_title(sub_titles[i], color=COLORS["text"], fontsize=11)
        ax.legend(loc="upper left", facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
        
    ax = axes[5]
    for m in METHODS:
        ax.plot(rolling_df.index, rolling_df[m], color=COLORS[m], linewidth=1.2, label=m.capitalize())
    ax.set_ylabel("Normalized", color=COLORS["text"])
    ax.set_title("All Methods Overlaid", color=COLORS["text"], fontsize=11)
    ax.legend(loc="upper left", facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    
    # Shade events across all axes
    for ev in EVENTS.values():
        start = pd.Timestamp(ev["start"])
        end = pd.Timestamp(ev["end"])
        for ax in axes:
            ax.axvspan(start, end, alpha=0.15, color=ev["color"])
            
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "entropy_comparison_overview.png"
    fig.savefig(plot_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")

def plot_metrics(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in axes.flatten(): style_ax(ax)
    
    # Plot 1: Lead Time
    ax = axes[0, 0]
    bars = []
    for i, m in enumerate(METHODS):
        bars.append(ax.bar(i, metrics["lead_time"][m], color=COLORS[m]))
    ax.set_xticks(range(4))
    ax.set_xticklabels([m.capitalize() for m in METHODS])
    ax.set_ylabel("Days", color=COLORS["text"])
    ax.set_title("Mean Crisis Lead Time", color=COLORS["text"])
    
    # Plot 2: FPR
    ax = axes[0, 1]
    for i, m in enumerate(METHODS):
        ax.bar(i, metrics["fp_rate"][m], color=COLORS[m])
    ax.set_xticks(range(4))
    ax.set_xticklabels([m.capitalize() for m in METHODS])
    ax.set_ylabel("Breaches / Month", color=COLORS["text"])
    ax.set_title("False Positive Rate (Lower is Better)", color=COLORS["text"])
    
    # Plot 3: SNR
    ax = axes[1, 0]
    for i, m in enumerate(METHODS):
        ax.bar(i, metrics["snr"][m], color=COLORS[m])
    ax.set_xticks(range(4))
    ax.set_xticklabels([m.capitalize() for m in METHODS])
    ax.set_ylabel("SNR Ratio", color=COLORS["text"])
    ax.set_title("Mean Signal-to-Noise Ratio", color=COLORS["text"])
    
    # Plot 4: Composite Score
    ax = axes[1, 1]
    scores = metrics["scores"]
    sorted_m = sorted(scores.items(), key=lambda x: x[1])
    names = [x[0].capitalize() for x in sorted_m]
    vals = [x[1] for x in sorted_m]
    
    y_pos = np.arange(len(names))
    rects = ax.barh(y_pos, vals, color=COLORS["grid"])
    
    # Color winner/runner-up
    for i, name in enumerate(names):
        if name.lower() == metrics["winner"]:
            rects[i].set_color("#ffd700") # gold
        elif name.lower() == metrics["runner_up"]:
            rects[i].set_color("#c0c0c0") # silver
            
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Composite Score [0-1]", color=COLORS["text"])
    ax.set_title("Final Method Ranking", color=COLORS["text"])
    
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", color=COLORS["text"], va='center')

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "entropy_metrics_bar_chart.png"
    fig.savefig(plot_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")

def plot_correlation(corr_df):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["text"])
    
    cax = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=COLORS["text"], labelcolor=COLORS["text"])
    
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels([c.capitalize() for c in corr_df.columns])
    ax.set_yticklabels([c.capitalize() for c in corr_df.index])
    
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr_df.iloc[i, j]) > 0.5 else "black")
            
    ax.set_title("Entropy Method Correlation During Crisis Windows\nLow correlation = complementary signals", color=COLORS["text"], pad=20)
    
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "entropy_correlation_heatmap.png"
    fig.savefig(plot_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = DATA_DIR / "Indices" / "SPY.csv"
    
    if not csv_file.exists():
        print(f"Error: Required data file {csv_file} not found.")
        exit(1)
        
    close, rolling_df, baselines, metrics = compute_metrics(str(csv_file))
    
    print("\n--- Generating Plots ---")
    plot_overview(close, rolling_df, baselines)
    plot_metrics(metrics)
    plot_correlation(metrics["correlation"])
    
    # Generate the JSON output
    print("\n--- Generating JSON Payload ---")
    run_entropy_method_selection("SPY", str(csv_file))
    
    print("\n--- Final Metrics Summary ---")
    print(f"Winner: {metrics['winner']}")
    print(f"Runner-up: {metrics['runner_up']}")
    
    # Store for formatting the markdown
    with open(OUTPUT_DIR / "empirical_metrics.csv", "w") as f:
        f.write("Method,LeadTime,FPRate,SNR,Consistency,BreachDur,Composite\n")
        for m in METHODS:
            lt = metrics["lead_time"][m]
            fpr = metrics["fp_rate"][m]
            snr = metrics["snr"][m]
            cons = metrics["consistency"][m]
            dur = metrics["duration"][m]
            comp = metrics["scores"][m]
            f.write(f"{m},{lt:.1f},{fpr:.3f},{snr:.3f},{cons:.2f},{dur:.1f},{comp:.3f}\n")
