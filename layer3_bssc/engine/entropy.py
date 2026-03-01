"""
entropy.py — Shannon Entropy Engine for Black Swan Detection

Measures disorder in market return distributions using Shannon Entropy.
Normal markets exhibit low entropy (returns cluster predictably around
the mean), while Black Swan events produce high entropy (the return
distribution flattens toward uniform, meaning anything becomes equally
probable).

This module is the mathematical confirmation layer inside BSSC (Layer 3).
When ``simulation.py`` generates a jump-diffusion price path, this module
classifies whether observed market behaviour represents normal volatility
or a true structural breakdown.

Output feeds directly into ``auditor/report.py`` for the final BSSC risk
report.

Mathematical Background
-----------------------
Shannon Entropy:
    H(X) = -Σ p(x) · log₂(p(x))

Normalised Entropy (used throughout this module):
    H_norm = H(X) / log₂(N)

where N is the number of bins.  H_norm ∈ [0, 1]:
    0 = perfectly predictable (all mass in one bin)
    1 = maximum disorder (uniform distribution)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Use Agg backend when no display is available (consistent with simulation.py)
if os.environ.get("CI") or (
    sys.platform.startswith("linux")
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "simulation_output"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

logger = logging.getLogger(__name__)


def _load_config_thresholds() -> tuple[float, float]:
    """Attempt to load entropy thresholds from config.yaml.

    Returns
    -------
    stress_threshold, black_swan_threshold : tuple[float, float]
        Falls back to (0.15, 0.30) if config is missing or malformed.
    """
    default = (0.15, 0.30)
    if not CONFIG_PATH.exists():
        return default
    try:
        import yaml  # noqa: F811 — optional, only used if available
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        layer3 = cfg.get("layer3", {})
        return (
            float(layer3.get("entropy_stress_threshold", default[0])),
            float(layer3.get("entropy_black_swan_threshold", default[1])),
        )
    except Exception:
        return default


# ---------------------------------------------------------------------------
# 1. Core Shannon Entropy Computation
# ---------------------------------------------------------------------------


def compute_shannon_entropy(
    returns: pd.Series | np.ndarray,
    n_bins: int = 50,
) -> float:
    """Compute normalised Shannon Entropy of a return distribution.

    Discretises the continuous return series into ``n_bins`` equal-width
    bins, computes the empirical probability mass function, and applies
    the Shannon formula normalised to [0, 1].

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Log-return series.  NaN values are removed automatically.
    n_bins : int, optional
        Number of bins for the histogram (default 50).

    Returns
    -------
    float
        Normalised entropy in [0, 1].
        0 = perfectly predictable, 1 = maximum disorder.

    Edge Cases
    ----------
    - Empty input → 0.0
    - All identical values → 0.0
    - Fewer unique values than ``n_bins`` → ``n_bins`` is reduced
      automatically to ``min(n_bins, n_unique)``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return 0.0

    n_unique = len(np.unique(arr))
    if n_unique <= 1:
        return 0.0

    # Reduce bins if data has fewer unique values than requested
    effective_bins = min(n_bins, n_unique)

    counts, _ = np.histogram(arr, bins=effective_bins)
    total = counts.sum()

    if total == 0:
        return 0.0

    probs = counts / total
    # Remove zero-probability bins (log(0) is undefined)
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))

    # Normalise by maximum possible entropy for this bin count
    max_entropy = np.log2(effective_bins)
    if max_entropy == 0:
        return 0.0

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 2. Rolling Entropy Time Series
# ---------------------------------------------------------------------------


def compute_rolling_entropy(
    returns: pd.Series,
    window: int = 30,
    n_bins: int = 20,
) -> pd.Series:
    """Compute Shannon Entropy over a rolling window of returns.

    Slides a window of ``window`` trading days across the return series
    and computes :func:`compute_shannon_entropy` for each position.

    Parameters
    ----------
    returns : pd.Series
        Log-return series with a DatetimeIndex.
    window : int, optional
        Rolling window size in trading days (default 30).
    n_bins : int, optional
        Number of histogram bins per window (default 20, smaller than
        static because each window has fewer data points).

    Returns
    -------
    pd.Series
        Time series of normalised entropy values.  NaN for periods
        before enough data accumulates (``min_periods = window // 2``).
    """
    min_periods = window // 2
    entropy_values = pd.Series(np.nan, index=returns.index, dtype=float)

    for i in range(len(returns)):
        start = max(0, i - window + 1)
        chunk = returns.iloc[start : i + 1]

        if len(chunk) < min_periods:
            continue

        entropy_values.iloc[i] = compute_shannon_entropy(chunk.values, n_bins)

    return entropy_values


# ---------------------------------------------------------------------------
# 3. Market State Classification
# ---------------------------------------------------------------------------


def classify_market_state(
    current_entropy: float,
    baseline_entropy: float,
    stress_threshold: float | None = None,
    black_swan_threshold: float | None = None,
) -> str:
    """Classify the current market state based on entropy deviation.

    Compares the current entropy value against a baseline (calm-period)
    reference.  The excess entropy determines the classification.

    Parameters
    ----------
    current_entropy : float
        Entropy value at the current time window.
    baseline_entropy : float
        Mean entropy during a calm reference period.
    stress_threshold : float | None, optional
        Excess entropy above baseline that triggers ``"STRESS"``.
        Read from ``config.yaml`` if ``None`` / omitted.
    black_swan_threshold : float | None, optional
        Excess entropy above baseline that triggers ``"BLACK_SWAN"``.
        Read from ``config.yaml`` if ``None`` / omitted.

    Returns
    -------
    str
        One of ``"NORMAL"``, ``"STRESS"``, or ``"BLACK_SWAN"``.
    """
    # Try to read thresholds from config, falling back to func defaults
    cfg_stress, cfg_black_swan = _load_config_thresholds()
    stress_threshold = stress_threshold if stress_threshold is not None else cfg_stress
    black_swan_threshold = black_swan_threshold if black_swan_threshold is not None else cfg_black_swan

    excess = current_entropy - baseline_entropy

    if excess >= black_swan_threshold:
        return "BLACK_SWAN"
    if excess >= stress_threshold:
        return "STRESS"
    return "NORMAL"


# ---------------------------------------------------------------------------
# 4. Entropy Acceleration
# ---------------------------------------------------------------------------


def compute_entropy_acceleration(entropy_series: pd.Series) -> pd.Series:
    """Compute how rapidly entropy is changing over time.

    A sudden *acceleration* in entropy is a stronger Black Swan signal
    than sustained high entropy (which may simply reflect a volatile
    but stable regime).

    Parameters
    ----------
    entropy_series : pd.Series
        Output from :func:`compute_rolling_entropy`.

    Returns
    -------
    pd.Series
        Rolling 5-day mean of absolute day-over-day entropy changes.
        Same index as input.
    """
    daily_diff = entropy_series.diff()
    acceleration = daily_diff.abs().rolling(window=5, min_periods=1).mean()
    return acceleration


# ---------------------------------------------------------------------------
# 5. Full Entropy Analysis Pipeline
# ---------------------------------------------------------------------------


def _load_ohlcv(csv_path: str | Path) -> pd.DataFrame:
    """Load a yfinance-formatted OHLCV CSV and return a flat DataFrame."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = [col[0] for col in df.columns]
    return df


def run_entropy_analysis(
    ticker: str,
    csv_path: str,
    calm_start: str,
    calm_end: str,
    event_start: str,
    event_end: str,
) -> dict:
    """Run the full Shannon Entropy analysis pipeline on a single ticker.

    Loads OHLCV data, computes rolling entropy, classifies each day in
    the event window, and generates a three-panel diagnostic plot saved
    to ``data/simulation_output/``.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. ``"SPY"``).
    csv_path : str
        Path to the yfinance CSV (e.g. ``"data/Indices/SPY.csv"``).
    calm_start, calm_end : str
        ``"YYYY-MM-DD"`` date range for the calm baseline period.
    event_start, event_end : str
        ``"YYYY-MM-DD"`` date range for the crisis period under study.

    Returns
    -------
    dict
        Keys:
        - ``baseline_entropy`` : float — mean entropy during calm period
        - ``event_entropy``    : float — mean entropy during event period
        - ``entropy_delta``    : float — event minus baseline
        - ``peak_entropy``     : float — max entropy during event period
        - ``peak_entropy_date``: str   — date of peak entropy
        - ``black_swan_days``  : int   — days classified BLACK_SWAN
        - ``stress_days``      : int   — days classified STRESS
        - ``classification_series`` : pd.Series — per-day classifications
        - ``plot_path``        : str   — path to the saved PNG
    """
    # --- Load and compute log returns ---
    df = _load_ohlcv(csv_path)
    close = df["Close"].dropna().astype(float)
    log_returns = np.log(close / close.shift(1)).dropna()
    log_returns = log_returns.loc[~log_returns.index.duplicated(keep="first")]

    # --- Rolling entropy ---
    rolling_ent = compute_rolling_entropy(log_returns, window=30, n_bins=20)

    # --- Baseline entropy (calm period) ---
    calm_mask = (rolling_ent.index >= calm_start) & (rolling_ent.index <= calm_end)
    calm_entropy = rolling_ent.loc[calm_mask].dropna()
    baseline_entropy = float(calm_entropy.mean()) if len(calm_entropy) > 0 else 0.0

    # --- Event entropy ---
    event_mask = (rolling_ent.index >= event_start) & (rolling_ent.index <= event_end)
    event_entropy_series = rolling_ent.loc[event_mask].dropna()
    event_entropy = (
        float(event_entropy_series.mean()) if len(event_entropy_series) > 0 else 0.0
    )

    # --- Classify each day in event period ---
    cfg_stress, cfg_bswan = _load_config_thresholds()
    classifications = event_entropy_series.apply(
        lambda h: classify_market_state(
            h, baseline_entropy, 
            stress_threshold=cfg_stress, 
            black_swan_threshold=cfg_bswan
        )
    )

    black_swan_days = int((classifications == "BLACK_SWAN").sum())
    stress_days = int((classifications == "STRESS").sum())

    # --- Peak entropy ---
    if len(event_entropy_series) > 0:
        peak_idx = event_entropy_series.idxmax()
        peak_entropy = float(event_entropy_series.loc[peak_idx])
        peak_entropy_date = str(peak_idx.date())
    else:
        peak_entropy = 0.0
        peak_entropy_date = ""

    # --- Entropy acceleration ---
    acceleration = compute_entropy_acceleration(rolling_ent)

    # --- Plot ---
    plot_path = _plot_entropy_analysis(
        ticker=ticker,
        close=close,
        rolling_ent=rolling_ent,
        acceleration=acceleration,
        baseline_entropy=baseline_entropy,
        event_start=event_start,
        event_end=event_end,
        stress_threshold=cfg_stress,
        black_swan_threshold=cfg_bswan,
    )

    return {
        "baseline_entropy": round(baseline_entropy, 6),
        "event_entropy": round(event_entropy, 6),
        "entropy_delta": round(event_entropy - baseline_entropy, 6),
        "peak_entropy": round(peak_entropy, 6),
        "peak_entropy_date": peak_entropy_date,
        "black_swan_days": black_swan_days,
        "stress_days": stress_days,
        "classification_series": classifications,
        "plot_path": str(plot_path),
    }


# ---------------------------------------------------------------------------
# Plotting (dark-mode, consistent with simulation.py)
# ---------------------------------------------------------------------------


def _plot_entropy_analysis(
    ticker: str,
    close: pd.Series,
    rolling_ent: pd.Series,
    acceleration: pd.Series,
    baseline_entropy: float,
    event_start: str,
    event_end: str,
    stress_threshold: float = 0.15,
    black_swan_threshold: float = 0.30,
) -> Path:
    """Generate a three-panel entropy diagnostic plot in dark mode."""

    # --- Dark-mode palette (matches simulation.py) ---
    COLOR_BG = "#0d1117"
    COLOR_GRID = "#21262d"
    COLOR_TEXT = "#c9d1d9"
    COLOR_PRICE = "#58a6ff"
    COLOR_ENTROPY = "#f0883e"
    COLOR_ACCEL = "#a371f7"
    COLOR_BASELINE = "#8b949e"
    COLOR_STRESS = "#d29922"
    COLOR_BSWAN = "#f85149"
    COLOR_EVENT_BG = "#f8514920"

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, dpi=120
    )
    fig.patch.set_facecolor(COLOR_BG)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(COLOR_BG)
        ax.tick_params(colors=COLOR_TEXT, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_GRID)
        ax.spines["bottom"].set_color(COLOR_GRID)
        ax.grid(True, alpha=0.15, color=COLOR_GRID, linestyle="--")

    ev_start = pd.Timestamp(event_start)
    ev_end = pd.Timestamp(event_end)

    stress_line = baseline_entropy + stress_threshold
    bswan_line = baseline_entropy + black_swan_threshold

    # --- Subplot 1: Price ---
    ax1.plot(close.index, close.values, color=COLOR_PRICE, linewidth=1.0)
    ax1.axvspan(ev_start, ev_end, alpha=0.12, color=COLOR_BSWAN)
    ax1.set_ylabel("Close Price", fontsize=10, color=COLOR_TEXT, labelpad=8)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )
    ax1.set_title(
        f"Shannon Entropy Analysis — {ticker}",
        fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=10,
    )

    # --- Subplot 2: Rolling Entropy ---
    ax2.plot(
        rolling_ent.index, rolling_ent.values,
        color=COLOR_ENTROPY, linewidth=1.0, label="Rolling Entropy (30d)",
    )
    ax2.axhline(
        baseline_entropy, color=COLOR_BASELINE, linestyle="--",
        linewidth=1.0, label=f"Baseline ({baseline_entropy:.3f})",
    )
    ax2.axhline(
        stress_line, color=COLOR_STRESS, linestyle="--",
        linewidth=0.8, label=f"Stress threshold ({stress_line:.3f})",
    )
    ax2.axhline(
        bswan_line, color=COLOR_BSWAN, linestyle="--",
        linewidth=0.8, label=f"Black Swan threshold ({bswan_line:.3f})",
    )
    ax2.axvspan(ev_start, ev_end, alpha=0.12, color=COLOR_BSWAN)
    ax2.set_ylabel("Entropy [0‒1]", fontsize=10, color=COLOR_TEXT, labelpad=8)
    ax2.legend(
        loc="upper left", fontsize=7, facecolor=COLOR_BG,
        edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT,
    )

    # --- Subplot 3: Entropy Acceleration ---
    ax3.plot(
        acceleration.index, acceleration.values,
        color=COLOR_ACCEL, linewidth=1.0, label="Entropy Acceleration (5d)",
    )
    ax3.axvspan(ev_start, ev_end, alpha=0.12, color=COLOR_BSWAN)
    ax3.set_ylabel("Acceleration", fontsize=10, color=COLOR_TEXT, labelpad=8)
    ax3.set_xlabel("Date", fontsize=10, color=COLOR_TEXT, labelpad=8)
    ax3.legend(
        loc="upper left", fontsize=7, facecolor=COLOR_BG,
        edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT,
    )

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"entropy_{ticker}_{event_end}.png"
    fig.savefig(
        out_file, dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    logger.info("Entropy plot saved → %s", out_file)

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)

    return out_file
