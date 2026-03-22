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
import antropy

# Fixed normalization reference for Sample Entropy.
# All sample entropy values are normalized against log(252)
# regardless of actual window length to ensure comparability
# across windows of different sizes.
_SAMPLE_ENTROPY_NORM_CONSTANT = np.log(252)  # ≈ 5.529

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
# 1. Core Entropy Computations
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


def compute_permutation_entropy(
    returns: pd.Series | np.ndarray,
    order: int = 3,
    delay: int = 1,
) -> float:
    """Compute normalised Permutation Entropy of a return sequence.

    Measures the diversity of ordinal patterns in return sequences. It
    splits the returns into overlapping subsequences of length `order`,
    records the rank ordering (e.g. up-up, down-up), and computes the 
    Shannon entropy of these ordinal patterns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Log-return series.
    order : int, optional
        Length of subsequences (default 3). Captures up-up, up-down, 
        down-up, down-down patterns — the minimum meaningful structure.
    delay : int, optional
        Time delay between elements (default 1).

    Returns
    -------
    float
        Normalised permutation entropy in [0, 1].

    Notes
    -----
    Consumed by BSSC Layer 3 -> Convergence. During directional crashes,
    ordinal patterns become monotonically decreasing, reducing entropy.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    
    if len(arr) < order:
        return 0.0
        
    try:
        perm_en = antropy.perm_entropy(arr, order=order, delay=delay, normalize=True)
    except Exception:
        return 0.0
        
    if np.isnan(perm_en) or np.isinf(perm_en):
        return 0.0
        
    return float(np.clip(perm_en, 0.0, 1.0))


def compute_sample_entropy(
    returns: pd.Series | np.ndarray,
    order: int = 2,
    metric: str = "chebyshev",
    tolerance: float = 0.005,
) -> float:
    """Compute normalised Sample Entropy of a return sequence.

    Measures the negative log probability that sequences similar in `order` 
    points remain similar at `order+1` points. Detects breakdown in self-similarity.
    High SampEn = patterns stop repeating = structural breakdown.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Log-return series.
    order : int, optional
        Template length `m` (default 2).
    metric : str, optional
        Distance metric (default 'chebyshev').
    tolerance : float, optional
        Absolute similarity distance threshold (default 0.005). Must be fixed
        rather than dynamic (antropy default scales by std) so crashes don't
        artificially widen the acceptance band.

    Returns
    -------
    float
        Normalised sample entropy in [0, 1].
        
    Notes
    -----
    Has an INVERSE relationship with Shannon entropy during crashes:
    Shannon drops during directional crash, while SampEn rises.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    
    if n <= order:
        return 0.0
        
    try:
        samp_en = antropy.sample_entropy(arr, order=order, metric=metric, tolerance=tolerance)
    except Exception:
        samp_en = np.nan
        
    if np.isnan(samp_en) or np.isinf(samp_en):
        return 1.0  # Max disorder
        
    # Normalise by fixed reference window (252 trading days = 1 year).
    # Using np.log(n) where n = current window length makes values
    # incomparable across windows of different sizes — short windows
    # produce artificially inflated values because the denominator
    # is smaller. A fixed reference constant ensures all windows
    # are normalized to the same scale regardless of length.
    # Reference: log(252) = 5.529 (one trading year baseline)
    return float(np.clip(
        samp_en / _SAMPLE_ENTROPY_NORM_CONSTANT, 0.0, 1.0
    ))


def compute_tsallis_entropy(
    returns: pd.Series | np.ndarray,
    q: float = 0.5,
    n_bins: int = 50,
) -> float:
    """Compute normalised Tsallis Entropy of a return distribution.

    A generalised entropy that amplifies the contribution of rare events
    when `q < 1`. Highly sensitive to fat tails indicative of Black Swans.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Log-return series.
    q : float, optional
        Entropic index (default 0.5 for black swan sensitivity).
    n_bins : int, optional
        Number of bins for discretization (default 50).

    Returns
    -------
    float
        Normalised Tsallis entropy in [0, 1].
    """
    if np.isclose(q, 1.0):
        return compute_shannon_entropy(returns, n_bins=n_bins)
        
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return 0.0

    n_unique = len(np.unique(arr))
    if n_unique <= 1:
        return 0.0

    effective_bins = min(n_bins, n_unique)
    counts, _ = np.histogram(arr, bins=effective_bins)
    total = counts.sum()

    if total == 0:
        return 0.0

    probs = counts / total
    probs = probs[probs > 0]
    
    # Tsallis formula: S_q = (1 - sum(p^q)) / (q - 1)
    s_q = (1.0 - np.sum(np.power(probs, q))) / (q - 1.0)
    
    # Max possible Tsallis entropy: S_q_max = (1 - n_bins^(1-q)) / (q - 1)
    s_q_max = (1.0 - np.power(effective_bins, 1.0 - q)) / (q - 1.0)
    
    if s_q_max == 0.0 or np.isnan(s_q_max):
        return 0.0
        
    return float(np.clip(s_q / s_q_max, 0.0, 1.0))




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

def compute_rolling_multi_entropy(
    returns: pd.Series,
    window: int = 30,
) -> pd.DataFrame:
    """Compute all four entropy types over a rolling window.

    Rolls Shannon, Permutation, Sample, and Tsallis entropy across
    the return series.

    Parameters
    ----------
    returns : pd.Series
        Log-return series with a DatetimeIndex.
    window : int, optional
        Rolling window size in trading days (default 30).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['shannon', 'permutation', 'sample', 'tsallis'].
    """
    min_periods = window // 2
    
    cols = ["shannon", "permutation", "sample", "tsallis"]
    df = pd.DataFrame(np.nan, index=returns.index, columns=cols, dtype=float)

    for i in range(len(returns)):
        start = max(0, i - window + 1)
        chunk = returns.iloc[start : i + 1]

        if len(chunk) < min_periods:
            continue

        vals = chunk.values
        df.iloc[i, 0] = compute_shannon_entropy(vals, n_bins=20)
        df.iloc[i, 1] = compute_permutation_entropy(vals)
        df.iloc[i, 2] = compute_sample_entropy(vals)
        df.iloc[i, 3] = compute_tsallis_entropy(vals, n_bins=20)

    return df


# ---------------------------------------------------------------------------
# 3. Market State Classification
# ---------------------------------------------------------------------------


def classify_market_state(
    current_entropy: float,
    baseline_entropy: float,
    stress_threshold: float | None = None,
    black_swan_threshold: float | None = None,
    confirmation_entropy: float | None = None,
    confirmation_baseline: float | None = None,
    vol_regime: dict | None = None,
    perm_alarm: dict | None = None,
) -> str:
    """Classify the current market state based on entropy deviation.

    Compares the current entropy value against a baseline (calm-period)
    reference. The primary signal verifies against a confirmation 
    signal to reduce false positives.

    Parameters
    ----------
    current_entropy : float
        Primary signal (Sample Entropy) value at the current time window.
    baseline_entropy : float
        Primary signal (Sample Entropy) baseline from a calm reference period.
    stress_threshold : float | None, optional
        Excess entropy above baseline that triggers ``"STRESS"``.
        Read from ``config.yaml`` if ``None`` / omitted.
    black_swan_threshold : float | None, optional
        Excess entropy above baseline that triggers ``"BLACK_SWAN"``.
        Read from ``config.yaml`` if ``None`` / omitted.
    confirmation_entropy : float | None, optional
        Confirmation signal (Permutation Entropy).
    confirmation_baseline : float | None, optional
        Confirmation signal (Permutation Entropy) baseline from calm period.
    vol_regime: dict | None, optional
        Two-signal output dict containing current_state from vol persistence.
    perm_alarm: dict | None, optional
        Two-signal output dict containing alarm_active boolean.

    Returns
    -------
    str
        One of ``"NORMAL"``, ``"WATCH"``, ``"STRESS"``, or ``"BLACK_SWAN"``.

    Notes
    -----
    Primary signal is now Sample Entropy (selected empirically — see 
    final_test_results.md Entry 001). Confirmation signal is Permutation Entropy.
    Two-signal mode requires both to agree for BLACK_SWAN.
    Single-signal mode (confirmation_entropy=None) preserves original 
    behavior for backward compatibility.
    Reference: data/simulation_output/entropy_method_selection.json
    """
    # NEW TWO-SIGNAL MODE
    # If vol_regime and perm_alarm are provided,
    # use the validated two-signal architecture.
    # This is the primary detection path.
    if vol_regime is not None:
        vol_state    = vol_regime.get("current_state", "NORMAL")
        alarm_active = perm_alarm.get("alarm_active", False) \
                       if perm_alarm is not None else False

        if vol_state == "BLACK_SWAN":
            return "BLACK_SWAN"
        elif vol_state == "STRESS":
            return "STRESS"
        elif alarm_active:
            return "WATCH"
        else:
            return "NORMAL"

    # LEGACY MODE (backward compatible)
    # Falls through to existing logic if vol_regime
    # is not provided.
    
    # Try to read thresholds from config, falling back to func defaults
    cfg_stress, cfg_black_swan = _load_config_thresholds()
    stress_threshold = stress_threshold if stress_threshold is not None else cfg_stress
    black_swan_threshold = black_swan_threshold if black_swan_threshold is not None else cfg_black_swan

    # Primary signal (Sample Entropy)
    excess = current_entropy - baseline_entropy
    
    if confirmation_entropy is not None and confirmation_baseline is not None:
        # Two-signal mode: require confirmation for BLACK_SWAN
        # Permutation entropy drops consistently during crashes
        # Confirmed if permutation is below its baseline (directional check)
        permutation_confirmed = (
            confirmation_entropy < confirmation_baseline
        )
        
        if excess >= black_swan_threshold and permutation_confirmed:
            return "BLACK_SWAN"
        elif excess >= black_swan_threshold and not permutation_confirmed:
            return "STRESS"  # Primary fires but confirmation absent
        elif excess >= stress_threshold:
            return "STRESS"
        else:
            return "NORMAL"
    
    else:
        # Single-signal mode: original behavior preserved exactly
        # This ensures backward compatibility
        if excess >= black_swan_threshold:
            return "BLACK_SWAN"
        elif excess >= stress_threshold:
            return "STRESS"
        else:
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


def compute_volatility_regime(
    returns: pd.Series,
    baseline_mean_abs: float,
    stress_multiplier: float = 1.5,
    bswan_multiplier: float = 3.0,
    stress_persistence_days: int = 10,
    bswan_persistence_days: int = 5,
    rolling_window: int = 10,
) -> dict:
    """
    Classify market regime using volatility ratio
    with persistence gating.

    Empirically validated in TS-002 and persistence
    diagnostic. Replaces entropy as primary signal
    for sustained crash detection.

    Volatility ratio does not adapt to crashes —
    it always compares to the fixed historical baseline.

    Parameters
    ----------
    returns : pd.Series
        Daily return series with DatetimeIndex.
    baseline_mean_abs : float
        Mean absolute daily return from calm period.
        TS-002 winner: 0.00714 (B Split baseline).
    stress_multiplier : float
        Vol ratio threshold for STRESS. Default 1.5x.
    bswan_multiplier : float
        Vol ratio threshold for BLACK_SWAN. Default 3.0x.
    stress_persistence_days : int
        Consecutive days above stress threshold
        required before confirming STRESS. Default 10.
        Validated: filters 8-day vaccine rally streak.
    bswan_persistence_days : int
        Consecutive days above black swan threshold
        required before confirming BLACK_SWAN. Default 5.
        Validated: fires on COVID 23-day streak.
    rolling_window : int
        Rolling window for volatility computation. Default 10.

    Returns
    -------
    dict
        Keys:
          current_state: str
            NORMAL / STRESS / BLACK_SWAN
          vol_ratio: float
            Current rolling vol / baseline
          stress_streak: int
            Current consecutive days above stress threshold
          bswan_streak: int
            Current consecutive days above bswan threshold
          rolling_vol_series: pd.Series
            Full rolling volatility ratio time series
          stress_breach_series: pd.Series
            Boolean series — above stress threshold
          bswan_breach_series: pd.Series
            Boolean series — above bswan threshold
    """
    rolling_abs = returns.abs().rolling(rolling_window).mean()
    vol_ratio_series = rolling_abs / baseline_mean_abs

    stress_breach = (vol_ratio_series > stress_multiplier)
    bswan_breach  = (vol_ratio_series > bswan_multiplier)

    def current_streak(series):
        values = series.dropna().values
        if len(values) == 0:
            return 0
        streak = 0
        for v in reversed(values):
            if v:
                streak += 1
            else:
                break
        return streak

    stress_streak = current_streak(stress_breach)
    bswan_streak  = current_streak(bswan_breach)

    if bswan_streak >= bswan_persistence_days:
        current_state = "BLACK_SWAN"
    elif stress_streak >= stress_persistence_days:
        current_state = "STRESS"
    else:
        current_state = "NORMAL"

    current_vol_ratio = float(
        vol_ratio_series.dropna().iloc[-1]
        if len(vol_ratio_series.dropna()) > 0
        else 0.0
    )

    return {
        "current_state": current_state,
        "vol_ratio": current_vol_ratio,
        "stress_streak": stress_streak,
        "bswan_streak": bswan_streak,
        "rolling_vol_series": vol_ratio_series,
        "stress_breach_series": stress_breach,
        "bswan_breach_series": bswan_breach
    }

def compute_permutation_alarm(
    returns: pd.Series,
    baseline_perm_entropy: float,
    alarm_drop_threshold: float = 0.05,
    rolling_window: int = 20,
) -> dict:
    """
    Compute permutation entropy early alarm signal.

    Permutation entropy detects when market return
    sequences become directionally monotonic —
    a sign that the market is entering a one-directional
    trend before volatility magnitude becomes extreme.

    Validated in persistence diagnostic:
      COVID:   alarm fires 56 days before vol ratio
      Q4 2018: alarm fires 11 days before vol ratio
      Vaccine: alarm never fires (0 false alarms)

    This is the ALARM not the CONFIRMATION.
    It fires with no persistence gate — rather safe
    than sorry. Vol ratio confirmation required for
    STRESS or BLACK_SWAN classification.

    Parameters
    ----------
    returns : pd.Series
        Daily return series with DatetimeIndex.
    baseline_perm_entropy : float
        Mean permutation entropy during calm period.
        Computed from 2018-Q1 data.
    alarm_drop_threshold : float
        How far below baseline triggers alarm. Default 0.05.
        Validated: 0.05 drop fires on COVID and Q4 2018,
        silent on vaccine rally.
    rolling_window : int
        Rolling window for permutation entropy. Default 20.

    Returns
    -------
    dict
        Keys:
          alarm_active: bool
            True if current perm entropy below threshold
          current_perm_entropy: float
            Most recent rolling permutation entropy value
          alarm_threshold: float
            baseline_perm_entropy - alarm_drop_threshold
          perm_entropy_series: pd.Series
            Full rolling permutation entropy time series
          alarm_series: pd.Series
            Boolean series — alarm active each day
    """
    result = []
    dates = []
    for i in range(rolling_window - 1, len(returns)):
        chunk = returns.iloc[
            i - rolling_window + 1:i + 1
        ].values
        try:
            pe = antropy.perm_entropy(
                chunk, order=3, delay=1, normalize=True
            )
            if np.isnan(pe) or np.isinf(pe):
                pe = float(baseline_perm_entropy)
        except Exception:
            pe = float(baseline_perm_entropy)
        result.append(pe)
        dates.append(returns.index[i])
    perm_series = pd.Series(result, index=dates)

    alarm_threshold = baseline_perm_entropy - alarm_drop_threshold
    alarm_series = perm_series < alarm_threshold

    if len(perm_series) > 0:
        current_perm = float(perm_series.iloc[-1])
        alarm_active = bool(alarm_series.iloc[-1])
    else:
        current_perm = float(baseline_perm_entropy)
        alarm_active = False

    return {
        "alarm_active": alarm_active,
        "current_perm_entropy": current_perm,
        "alarm_threshold": alarm_threshold,
        "perm_entropy_series": perm_series,
        "alarm_series": alarm_series
    }

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
    calm_entropy = rolling_ent.loc[calm_start:calm_end].dropna()

    print(f"[DEBUG] Requested calm period ({calm_start} to {calm_end}) has {len(calm_entropy)} days of data.")
    
    if len(calm_entropy) > 0:
        print(f"[DEBUG] Raw entropy values (first 5): {calm_entropy.head().values}")
        baseline_entropy = float(calm_entropy.mean())
        print(f"[DEBUG] Computed mean baseline: {baseline_entropy:.6f}")
        
        baseline_period_used = "requested"
        baseline_start_date = str(calm_entropy.index[0].date())
        baseline_end_date = str(calm_entropy.index[-1].date())
    else:
        print("[WARNING] Calm period slice is empty. Falling back to the first 252 available trading days.")
        calm_entropy = rolling_ent.dropna().iloc[:252]
        baseline_entropy = float(calm_entropy.mean()) if len(calm_entropy) > 0 else 0.0
        print(f"[DEBUG] Fallback mean baseline: {baseline_entropy:.6f}")
        
        baseline_period_used = "fallback_first_252_days"
        baseline_start_date = str(calm_entropy.index[0].date()) if len(calm_entropy) > 0 else calm_start
        baseline_end_date = str(calm_entropy.index[-1].date()) if len(calm_entropy) > 0 else calm_end

    # --- Event entropy ---
    event_entropy_series = rolling_ent.loc[event_start:event_end].dropna()
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
        "baseline_period_used": baseline_period_used,
        "baseline_start_date": baseline_start_date,
        "baseline_end_date": baseline_end_date,
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

# ---------------------------------------------------------------------------
# 6. Multi-Entropy Empirical Evaluation
# ---------------------------------------------------------------------------

def plot_multi_entropy_comparison(
    ticker: str,
    csv_path: str,
    event_start: str,
    event_end: str,
    calm_start: str | None = None,
    calm_end: str | None = None,
) -> dict:
    """Generate a 5-panel comparison plot for all four entropy types.

    Parameters
    ----------
    ticker : str
    csv_path : str
    event_start : str
    event_end : str
    calm_start : str | None, optional
    calm_end : str | None, optional

    Returns
    -------
    dict
        plot_path, entropy_df, correlation_matrix
    """
    df = _load_ohlcv(csv_path)
    close = df["Close"].dropna().astype(float)
    log_returns = np.log(close / close.shift(1)).dropna()
    log_returns = log_returns.loc[~log_returns.index.duplicated(keep="first")]

    rolling_df = compute_rolling_multi_entropy(log_returns, window=30)
    
    # Baseline logic
    if calm_start and calm_end:
        calm_df = rolling_df.loc[calm_start:calm_end].dropna()
        if len(calm_df) == 0:
            calm_df = rolling_df.dropna().iloc[:252]
    else:
        calm_df = rolling_df.dropna().iloc[:252]
        
    baselines = calm_df.mean() if len(calm_df) > 0 else pd.Series(0.0, index=rolling_df.columns)
    
    # Event logic for correlation
    event_df = rolling_df.loc[event_start:event_end].dropna()
    corr_matrix = event_df.corr() if len(event_df) > 0 else pd.DataFrame()
    
    # Plotting
    COLOR_BG = "#0d0d0d"
    COLOR_GRID = "#21262d"
    COLOR_TEXT = "#c9d1d9"
    COLOR_PRICE = "#58a6ff"
    COLOR_SHAN = "#f0883e"
    COLOR_PERM = "#3fb950"
    COLOR_SAMP = "#2f81f7"
    COLOR_TSAL = "#d29922"
    COLOR_EVENT_BG = "#f8514920"
    COLOR_BSWAN = "#f85149"
    COLOR_BASELINE = "#8b949e"
    
    fig, axes = plt.subplots(5, 1, figsize=(18, 22), sharex=True, dpi=150)
    fig.patch.set_facecolor(COLOR_BG)
    
    for ax in axes:
        ax.set_facecolor(COLOR_BG)
        ax.tick_params(colors=COLOR_TEXT, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_GRID)
        ax.spines["bottom"].set_color(COLOR_GRID)
        ax.grid(True, alpha=0.15, color=COLOR_GRID, linestyle="--")
        
    ev_start = pd.Timestamp(event_start)
    ev_end = pd.Timestamp(event_end)
    
    # Subplot 1: Price
    ax = axes[0]
    ax.plot(close.index, close.values, color=COLOR_PRICE, linewidth=1.2)
    ax.axvspan(ev_start, ev_end, alpha=0.12, color=COLOR_BSWAN)
    ax.set_ylabel("Close Price", fontsize=11, color=COLOR_TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(f"Multi-Entropy Analysis — {ticker}", fontsize=15, fontweight="bold", color=COLOR_TEXT, pad=12)
    
    # Subplot 2: All Entropies
    ax = axes[1]
    ax.plot(rolling_df.index, rolling_df["shannon"], color=COLOR_SHAN, label="Shannon", linewidth=1.2)
    ax.plot(rolling_df.index, rolling_df["permutation"], color=COLOR_PERM, label="Permutation", linewidth=1.2)
    ax.plot(rolling_df.index, rolling_df["sample"], color=COLOR_SAMP, label="Sample", linewidth=1.2)
    ax.plot(rolling_df.index, rolling_df["tsallis"], color=COLOR_TSAL, label="Tsallis", linewidth=1.2)
    ax.axvspan(ev_start, ev_end, alpha=0.12, color=COLOR_BSWAN)
    ax.set_ylabel("Normalized Entropy [0-1]", fontsize=11, color=COLOR_TEXT)
    ax.legend(loc="upper left", facecolor=COLOR_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    
    # Subplot 3: Shannon alone
    ax = axes[2]
    sh_base = baselines["shannon"]
    ax.plot(rolling_df.index, rolling_df["shannon"], color=COLOR_SHAN, linewidth=1.2, label="Shannon Entropy")
    ax.axhline(sh_base, color=COLOR_BASELINE, linestyle="--", label="Baseline")
    ax.axhline(sh_base + 0.15, color="#d29922", linestyle="--", label="Stress +0.15")
    ax.axhline(sh_base + 0.30, color=COLOR_BSWAN, linestyle="--", label="Black Swan +0.30")
    ax.axvspan(ev_start, ev_end, alpha=0.12, color=COLOR_BSWAN)
    ax.set_ylabel("Shannon Entropy [0-1]", fontsize=11, color=COLOR_TEXT)
    ax.legend(loc="upper left", facecolor=COLOR_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    
    # Subplot 4: Sample alone
    ax = axes[3]
    sa_base = baselines["sample"]
    ax.plot(rolling_df.index, rolling_df["sample"], color=COLOR_SAMP, linewidth=1.2, label="Sample Entropy")
    ax.axhline(sa_base, color=COLOR_BASELINE, linestyle="--", label="Baseline")
    ax.axhline(sa_base + 0.15, color="#d29922", linestyle="--", label="Stress +0.15")
    ax.axhline(sa_base + 0.30, color=COLOR_BSWAN, linestyle="--", label="Black Swan +0.30")
    ax.axvspan(ev_start, ev_end, alpha=0.12, color=COLOR_BSWAN)
    ax.set_ylabel("Sample Entropy [normalized]", fontsize=11, color=COLOR_TEXT)
    ax.set_title("Sample Entropy (rises during directional crash)", fontsize=11, color=COLOR_TEXT)
    ax.legend(loc="upper left", facecolor=COLOR_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    
    # Subplot 5: Tsallis alone
    ax = axes[4]
    ts_base = baselines["tsallis"]
    ax.plot(rolling_df.index, rolling_df["tsallis"], color=COLOR_TSAL, linewidth=1.2, label="Tsallis Entropy")
    ax.axhline(ts_base, color=COLOR_BASELINE, linestyle="--", label="Baseline")
    ax.axhline(ts_base + 0.15, color="#d29922", linestyle="--", label="Stress +0.15")
    ax.axhline(ts_base + 0.30, color=COLOR_BSWAN, linestyle="--", label="Black Swan +0.30")
    ax.axvspan(ev_start, ev_end, alpha=0.12, color=COLOR_BSWAN)
    ax.set_ylabel("Tsallis Entropy [normalized]", fontsize=11, color=COLOR_TEXT)
    ax.set_title("Tsallis Entropy (amplifies fat tail events)", fontsize=11, color=COLOR_TEXT)
    ax.legend(loc="upper left", facecolor=COLOR_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30, ha="right")
    
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"multi_entropy_{ticker}_{event_end}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)
        
    return {
        "plot_path": str(out_file),
        "entropy_df": rolling_df,
        "correlation_matrix": corr_matrix
    }

def run_entropy_method_selection(ticker: str, csv_path: str) -> dict:
    import json
    from datetime import datetime
    
    df = _load_ohlcv(csv_path)
    close = df["Close"].dropna().astype(float)
    log_returns = np.log(close / close.shift(1)).dropna()
    log_returns = log_returns.loc[~log_returns.index.duplicated(keep="first")]

    rolling_df = compute_rolling_multi_entropy(log_returns, window=30)
    
    events = {
        "covid_2020": {"start": "2020-02-01", "end": "2020-03-31"},
        "q4_2018_selloff": {"start": "2018-10-01", "end": "2018-12-31"},
        "fed_2022": {"start": "2022-01-01", "end": "2022-06-30"}
    }
    
    methods = ["shannon", "permutation", "sample", "tsallis"]
    
    # 1. False Positive Rate (Calm period = outside 60 days of any event)
    def is_calm(d):
        for ev in events.values():
            e_start = pd.Timestamp(ev["start"]) - pd.Timedelta(days=60)
            e_end = pd.Timestamp(ev["end"]) + pd.Timedelta(days=60)
            if e_start <= d <= e_end:
                return False
        return True
        
    calm_mask = rolling_df.index.map(is_calm)
    calm_df = rolling_df.loc[calm_mask].dropna()
    
    if len(calm_df) == 0:
        calm_df = rolling_df.dropna().iloc[:252]
        
    baselines = calm_df.mean()
    stds = calm_df.std()
    
    calm_months = len(calm_df) / 21.0
    
    fp_rate = {}
    for m in methods:
        breaches = (calm_df[m] > baselines[m] + 0.15).sum()
        fp_rate[m] = breaches / calm_months if calm_months > 0 else 0.0

    # For each method, aggregate Lead Time, Magnitude, and Direction
    lead_times = {m: [] for m in methods}
    magnitudes = {m: [] for m in methods}
    directions = {m: [] for m in methods}
    
    corr_matrices = []
    
    for en, ev in events.items():
        ev_start, ev_end = ev["start"], ev["end"]
        ev_df = rolling_df.loc[ev_start:ev_end]
        ev_close = close.loc[ev_start:ev_end]
        
        if len(ev_df) == 0 or len(ev_close) == 0:
            continue
            
        corr_matrices.append(ev_df.corr())
        min_date = ev_close.idxmin()
        
        for m in methods:
            # Shift magnitude
            mag = (ev_df[m].mean() - baselines[m]) / (stds[m] if stds[m] > 0 else 1.0)
            magnitudes[m].append(mag)
            
            # Direction
            direction = 1 if ev_df[m].mean() > baselines[m] else -1
            directions[m].append(direction)
            
            # Lead time
            pre_min_df = ev_df.loc[:min_date]
            breach_idx = pre_min_df[pre_min_df[m] > baselines[m] + 0.15].index
            
            if len(breach_idx) > 0:
                first_breach = breach_idx[0]
                # count trading days
                days_diff = len(ev_close.loc[first_breach:min_date]) - 1
                lead_times[m].append(max(0, days_diff))
            else:
                lead_times[m].append(0)
                
    # Averages
    avg_lt = {m: np.mean(lead_times[m]) if len(lead_times[m]) > 0 else 0 for m in methods}
    avg_mag = {m: np.abs(np.mean(magnitudes[m])) if len(magnitudes[m]) > 0 else 0 for m in methods}
    
    consistency = {}
    for m in methods:
        if len(directions[m]) == 0:
            consistency[m] = 0.0
            continue
        from collections import Counter
        mc = Counter(directions[m]).most_common(1)[0][1]
        consistency[m] = mc / len(directions[m])
        
    # Normalizing functions
    def normalize_dict(d, invert=False):
        vals = list(d.values())
        if max(vals) == min(vals):
            return {m: 1.0 for m in methods}
        if invert:
            return {m: (max(vals) - d[m]) / (max(vals) - min(vals)) for m in methods}
        return {m: (d[m] - min(vals)) / (max(vals) - min(vals)) for m in methods}
        
    norm_lt = normalize_dict(avg_lt)
    norm_fp = normalize_dict(fp_rate, invert=True)  # lower is better
    norm_mag = normalize_dict(avg_mag)
    
    scores = {}
    for m in methods:
        scores[m] = (0.40 * norm_lt[m]) + (0.30 * norm_fp[m]) + (0.20 * norm_mag[m]) + (0.10 * consistency[m])
        
    sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner = sorted_methods[0][0]
    
    # Event correlations to find runner up
    avg_corr = sum(corr_matrices) / len(corr_matrices) if len(corr_matrices) > 0 else pd.DataFrame()
    runner_up = None
    
    for m, sc in sorted_methods[1:]:
        c = avg_corr.loc[winner, m] if not avg_corr.empty else 0.0
        if c < 0.7:
            runner_up = m
            break
            
    if runner_up is None:
        runner_up = sorted_methods[1][0]
        
    rejected = {}
    for m, sc in sorted_methods:
        if m in (winner, runner_up):
            continue
        rejected[m] = f"Lower composite score ({sc:.3f}), did not meet selection criteria."
        
    print("╔══════════════╦═══════════╦══════════════╦═══════════╦═══════════════╗")
    print("║ Method       ║ Lead Time ║ False Pos/Mo ║ Magnitude ║ Consistency   ║")
    print("╠══════════════╬═══════════╬══════════════╬═══════════╬═══════════════╣")
    for m in methods:
        print(f"║ {m.capitalize():<12} ║ {avg_lt[m]:>6.1f} days ║ {fp_rate[m]:>12.2f} ║ {avg_mag[m]:>9.2f} ║ {consistency[m]:>13.2f} ║")
    print("╚══════════════╩═══════════╩══════════════╩═══════════╩═══════════════╝")
    print()
    print(f"SELECTED PRIMARY METHOD: {winner} (score: {scores[winner]:.3f})")
    print(f"SELECTED CONFIRMATION METHOD: {runner_up} (score: {scores[runner_up]:.3f})")
    print("REASON: Elected empirically via historical event framework weighting lead time heavily.")
    print()
    print("REJECTED METHODS:")
    for m, r in rejected.items():
        print(f"  {m}: {r}")

    out_metrics = {
        m: {
            "lead_time": float(avg_lt[m]),
            "false_pos_rate": float(fp_rate[m]),
            "magnitude": float(avg_mag[m]),
            "consistency": float(consistency[m]),
            "score": float(scores[m])
        } for m in methods
    }
    
    res = {
        "primary_method": winner,
        "confirmation_method": runner_up,
        "primary_score": float(scores[winner]),
        "confirmation_score": float(scores[runner_up]),
        "selection_rationale": "Empirical historical selection across 3 crises prioritizing lead time and low false positive rate. Runner-up selected for correlation < 0.7",
        "rejected_methods": rejected,
        "metrics_table": out_metrics,
        "evaluated_on": ["covid_2020", "q4_2018_selloff", "fed_2022"],
        "timestamp": datetime.now().isoformat()
    }
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "entropy_method_selection.json", "w") as f:
        json.dump(res, f, indent=2)
        
    return res
