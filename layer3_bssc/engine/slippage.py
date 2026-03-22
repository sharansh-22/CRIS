"""
slippage.py — Pipeline Stage 3: Execution Cost & Slippage Model

Quantifies how much real money a trader loses when executing during
different market regimes. Uses Implementation Shortfall (IS) instead
of VWAP to avoid self-contamination, combined with an Almgren-Chriss
square-root market impact model corrected by entropy-derived regime
multipliers.

Pipeline Position
-----------------
    simulation.py  →  entropy.py  →  slippage.py
    (price paths)     (regime)       (execution cost)

Mathematical Background
-----------------------
Implementation Shortfall:
    IS (bps) = |avg_fill - decision_price| / decision_price × 10,000

Almgren-Chriss Market Impact:
    base_impact = eta * sigma_daily * np.sqrt(order_frac) * 10_000

    Regime effects are captured natively via path volatility (sigma_daily).
    The entropy-derived regime multiplier is applied only to the spread cost.

Regime Multipliers (from entropy.classify_market_state):
    NORMAL    → 1.0×  (standard liquidity; square-root law holds)
    STRESS    → 1.5×  (spread widening; partial book thinning)
    BLACK_SWAN → 2.5× (one-sided flow; convex impact regime)

Usage
-----
    python -m layer3_bssc.engine.slippage              # default SPY
    python -m layer3_bssc.engine.slippage --ticker NSEI
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
if os.environ.get("CI") or (
    sys.platform.startswith("linux")
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layer3_bssc.engine.simulation import (
    calibrate_from_data,
    simulate_gbm,
    simulate_jumps,
)
from layer3_bssc.engine.entropy import (
    compute_rolling_entropy,
    classify_market_state,
)

# ---------------------------------------------------------------------------
# DEFERRED COMPONENT — PERMUTATION ENTROPY EXECUTION SPEED
# Original design specified two entropy inputs:
#   Sample Entropy (via market_state) → magnitude multiplier
#     IMPLEMENTED: entropy_to_slippage_multiplier()
#   Permutation Entropy → execution speed recommendation
#     DEFERRED: not yet implemented
#
# Planned implementation:
#   High Permutation Entropy → market is directional
#     → recommend faster execution (timing risk dominates)
#   Low Permutation Entropy  → market is chaotic not directional
#     → recommend moderate execution speed
#
# Combined signal:
#   High Sample + Low Permutation  → max multiplier + fast execution
#   High Sample + High Permutation → high multiplier + moderate speed
#
# Trigger for implementation: after Layer 2 MMAD is complete
# Permutation Entropy will have additional validation data
# from the microstructure regime comparison.
# Implement as compute_execution_speed_recommendation()
# taking permutation_entropy_score as input.
#
# CALIBRATION NOTE: eta=0.3 is at the upper bound of
# typical Almgren-Chriss calibrations. Recalibrate
# against empirical SPY execution costs during
# backtesting phase. Literature range: 0.1-0.3.
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "simulation_output"

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

# Regime multipliers: entropy state → slippage scaling factor
_REGIME_MULTIPLIERS = {
    "NORMAL": 1.0,
    "STRESS": 1.5,
    "BLACK_SWAN": 2.5,
}


# ---------------------------------------------------------------------------
# 1. Implementation Shortfall
# ---------------------------------------------------------------------------


def compute_implementation_shortfall(
    decision_price: float,
    fill_prices: np.ndarray,
    side: str = "sell",
) -> float:
    """Calculate the gap between decision price and average fill in bps.

    Parameters
    ----------
    decision_price : float
        Price at the moment the trade signal fires.
    fill_prices : np.ndarray
        Array of prices at which the order was actually filled
        (simulating partial fills across multiple time steps).
    side : str
        ``"sell"`` (liquidation during crash) or ``"buy"``.

    Returns
    -------
    float
        Implementation Shortfall in basis points (bps).
        Positive value = cost to the trader.
    """
    if decision_price <= 0 or len(fill_prices) == 0:
        return 0.0

    avg_fill = float(np.mean(fill_prices))

    if side == "sell":
        # Selling: we lose money if we fill below the decision price
        is_bps = ((decision_price - avg_fill) / decision_price) * 10_000
    else:
        # Buying: we lose money if we fill above the decision price
        is_bps = ((avg_fill - decision_price) / decision_price) * 10_000

    return max(0.0, float(is_bps))


# ---------------------------------------------------------------------------
# 2. Almgren-Chriss Market Impact
# ---------------------------------------------------------------------------


def compute_market_impact(
    sigma_daily: float,
    order_frac: float = 0.01,
    eta: float = 0.3,
) -> float:
    """Compute market impact using the Almgren-Chriss square-root model.

    Parameters
    ----------
    sigma_daily : float
        Daily volatility of the asset.
    order_frac : float
        Order size as fraction of ADV (Q / ADV). Default 1%.
    eta : float
        Impact coefficient (calibrated, typically 0.1–0.5).

    Returns
    -------
    float
        Market impact in basis points.
    """
    # Base square-root law: η × σ × √(Q/ADV)
    # (no multiplier here — path vol carries this)
    base_impact = eta * sigma_daily * np.sqrt(order_frac) * 10_000

    return float(base_impact)


def compute_spread_cost(
    base_spread_bps: float = 2.0,
    regime_multiplier: float = 1.0,
) -> float:
    """Compute execution cost from bid-ask spread widening.

    Parameters
    ----------
    base_spread_bps : float
        Normal bid-ask spread in basis points.
    regime_multiplier : float
        Entropy-derived scaling factor (1.0 / 1.5 / 2.5).

    Returns
    -------
    float
        Spread cost in basis points.
    """
    # Multiplier applies here — simulation.py does not model spread
    return float(base_spread_bps * regime_multiplier)


# ---------------------------------------------------------------------------
# 3. Entropy → Slippage Multiplier
# ---------------------------------------------------------------------------


def entropy_to_slippage_multiplier(market_state: str) -> float:
    """Map a market state from classify_market_state() to a multiplier.

    Parameters
    ----------
    market_state : str
        One of ``"NORMAL"``, ``"STRESS"``, ``"BLACK_SWAN"``.

    Returns
    -------
    float
        Regime multiplier (1.0 / 1.5 / 2.5).
    """
    return _REGIME_MULTIPLIERS.get(market_state, 1.0)


# ---------------------------------------------------------------------------
# 4. Monte Carlo Slippage Simulation
# ---------------------------------------------------------------------------


def _simulate_single_path_slippage(
    price_path: np.ndarray,
    sigma_daily: float,
    order_frac: float = 0.01,
    eta: float = 0.3,
    execution_window: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run slippage computation on a single simulated price path.

    Finds the maximum drawdown peak in the path and executes the
    liquidation order starting from that point. This models the
    realistic scenario: traders are forced to sell *during* the
    worst decline, not at random calm periods.

    Returns
    -------
    dict
        Keys: is_bps, impact_bps, total_bps, market_state, multiplier
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(price_path)
    if n < 60:
        return {
            "is_bps": 0.0, "impact_bps": 0.0, "spread_bps": 0.0, "total_bps": 0.0,
            "market_state": "NORMAL", "multiplier": 1.0,
        }

    # Compute log returns and rolling entropy for regime classification
    log_returns = np.diff(np.log(price_path))
    returns_series = pd.Series(log_returns, dtype=float)
    rolling_ent = compute_rolling_entropy(returns_series, window=30, n_bins=20)

    # EXECUTION ANCHORING — MAX DRAWDOWN PEAK
    # Assumption: execution is anchored to the maximum
    # drawdown peak of each simulated path.
    # Financial justification: forced liquidation during
    # a crisis concentrates at the worst moments —
    # margin calls arrive when losses are largest,
    # redemption requests peak during maximum fear,
    # risk limits are breached at peak drawdown.
    # This represents the realistic worst-case execution
    # scenario for a fund under stress, not a voluntarily
    # timed trade.
    # Known conservatism: a voluntarily timed trader would
    # experience lower average slippage. This conservatism
    # is intentional — CRIS is a risk assessment system,
    # not an execution optimizer. Conservative estimates
    # protect against underestimating crisis execution costs.
    # Trigger for revisiting: if backtesting shows actual
    # crisis IS consistently below modeled IS by more than
    # 40%, revisit anchoring methodology.
    cummax = np.maximum.accumulate(price_path)
    drawdowns = (cummax - price_path) / cummax
    # The drawdown peak is the point just before the trough deepens most
    # Find the trough (max drawdown point)
    trough_idx = int(np.argmax(drawdowns))
    # The execution starts at the peak before this trough
    # Walk back to find where the cummax was set
    peak_idx = int(np.argmax(price_path[:trough_idx + 1])) if trough_idx > 0 else 0
    # Ensure we have enough entropy history (at least 30 days)
    exec_start = max(peak_idx, 35)
    exec_start = min(exec_start, n - execution_window - 2)
    exec_end = min(exec_start + execution_window, n - 1)

    decision_price = float(price_path[exec_start])
    fill_prices = price_path[exec_start + 1 : exec_end + 1]

    if len(fill_prices) == 0:
        fill_prices = np.array([decision_price])

    # Classify regime at execution point using rolling entropy
    valid_ent = rolling_ent.dropna()
    if len(valid_ent) == 0:
        baseline_entropy = 0.5
        current_entropy = 0.5
    else:
        # First quarter of the path = calm baseline
        baseline_entropy = float(valid_ent.iloc[: len(valid_ent) // 4].mean())
        # Entropy at execution point
        ent_idx = min(exec_start, len(valid_ent) - 1)
        current_entropy = float(valid_ent.iloc[ent_idx])

    market_state = classify_market_state(current_entropy, baseline_entropy)
    multiplier = entropy_to_slippage_multiplier(market_state)

    # Compute IS — captures the price decline during the execution window
    is_bps = compute_implementation_shortfall(decision_price, fill_prices, side="sell")

    # Compute market impact (no multiplier here — path vol carries this)
    impact_bps = compute_market_impact(sigma_daily, order_frac, eta)

    # Compute spread cost (multiplier applies here)
    base_spread_bps = 2.0
    spread_bps = compute_spread_cost(base_spread_bps, multiplier)

    total_bps = is_bps + impact_bps + spread_bps

    return {
        "is_bps": round(is_bps, 4),
        "impact_bps": round(impact_bps, 4),
        "spread_bps": round(spread_bps, 4),
        "total_bps": round(total_bps, 4),
        "market_state": market_state,
        "multiplier": multiplier,
    }


def run_monte_carlo_slippage(
    S0: float,
    mu: float,
    sigma: float,
    n_paths: int = 2000,
    T: float = 1.0,
    dt: float = 1.0 / 252,
    order_frac: float = 0.01,
    eta: float = 0.3,
    execution_window: int = 5,
    mode: str = "gbm",
    lambda_j: float = 2.0,
    mu_j: float = -0.15,
    sigma_j: float = 0.10,
    base_seed: int = 42,
) -> dict:
    """Run Monte Carlo slippage simulation across many price paths.

    Parameters
    ----------
    S0 : float          Initial price from calibration.
    mu : float          Annualised drift.
    sigma : float       Annualised volatility.
    n_paths : int       Number of Monte Carlo paths (default 2000).
    T : float           Horizon in years.
    dt : float          Time-step (1/252 = daily).
    order_frac : float  Order size as fraction of ADV.
    eta : float         Impact coefficient.
    execution_window : int  Days over which the order is filled.
    mode : str          ``"gbm"`` or ``"jump_diffusion"``.
    lambda_j, mu_j, sigma_j : float  Jump parameters (JD mode).
    base_seed : int     Base random seed for reproducibility.

    Returns
    -------
    dict
        Distribution statistics: mean, median, p95, p99, max, plus
        per-path details and regime breakdown.
    """
    sigma_daily = sigma / np.sqrt(TRADING_DAYS_PER_YEAR)

    all_results = []

    for i in range(n_paths):
        seed = base_seed + i

        if mode == "gbm":
            _, price_path = simulate_gbm(S0, mu, sigma, T, dt, seed=seed)
        else:
            _, price_path, _ = simulate_jumps(
                S0, mu, sigma, lambda_j, mu_j, sigma_j, T, dt, seed=seed
            )

        rng = np.random.default_rng(seed + 10_000)
        result = _simulate_single_path_slippage(
            price_path, sigma_daily, order_frac, eta,
            execution_window, rng,
        )
        all_results.append(result)

    # Aggregate statistics
    total_bps_arr = np.array([r["total_bps"] for r in all_results])
    is_bps_arr = np.array([r["is_bps"] for r in all_results])
    impact_bps_arr = np.array([r["impact_bps"] for r in all_results])
    spread_bps_arr = np.array([r["spread_bps"] for r in all_results])

    # Regime breakdown
    states = [r["market_state"] for r in all_results]
    regime_counts = {
        "NORMAL": states.count("NORMAL"),
        "STRESS": states.count("STRESS"),
        "BLACK_SWAN": states.count("BLACK_SWAN"),
    }

    return {
        "mode": mode,
        "n_paths": n_paths,
        "total_slippage": {
            "mean": round(float(np.mean(total_bps_arr)), 4),
            "median": round(float(np.median(total_bps_arr)), 4),
            "p95": round(float(np.percentile(total_bps_arr, 95)), 4),
            "p99": round(float(np.percentile(total_bps_arr, 99)), 4),
            "max": round(float(np.max(total_bps_arr)), 4),
            "std": round(float(np.std(total_bps_arr)), 4),
        },
        "is_component": {
            "mean": round(float(np.mean(is_bps_arr)), 4),
            "p95": round(float(np.percentile(is_bps_arr, 95)), 4),
        },
        "impact_component": {
            "mean": round(float(np.mean(impact_bps_arr)), 4),
            "p95": round(float(np.percentile(impact_bps_arr, 95)), 4),
        },
        "spread_component": {
            "mean": round(float(np.mean(spread_bps_arr)), 4),
            "p95": round(float(np.percentile(spread_bps_arr, 95)), 4),
        },
        "regime_breakdown": regime_counts,
        "per_path_details": all_results,
    }


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------


def plot_slippage_comparison(
    gbm_results: dict,
    jd_results: dict,
    ticker: str,
) -> Path:
    """Generate a side-by-side comparison plot: GBM vs JD slippage."""

    COLOR_BG = "#0d1117"
    COLOR_GRID = "#21262d"
    COLOR_TEXT = "#c9d1d9"
    COLOR_GBM = "#58a6ff"
    COLOR_JD = "#f0883e"
    COLOR_BSWAN = "#f85149"

    gbm_totals = np.array([r["total_bps"] for r in gbm_results["per_path_details"]])
    jd_totals = np.array([r["total_bps"] for r in jd_results["per_path_details"]])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)

    for ax in axes:
        ax.set_facecolor(COLOR_BG)
        ax.tick_params(colors=COLOR_TEXT, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_GRID)
        ax.spines["bottom"].set_color(COLOR_GRID)
        ax.grid(True, alpha=0.15, color=COLOR_GRID, linestyle="--")

    # --- Panel 1: Histogram overlay ---
    ax = axes[0]
    ax.hist(gbm_totals, bins=40, alpha=0.7, color=COLOR_GBM, label="GBM (Normal)")
    ax.hist(jd_totals, bins=40, alpha=0.7, color=COLOR_JD, label="Jump-Diffusion")
    ax.set_xlabel("Total Slippage (bps)", color=COLOR_TEXT, fontsize=10)
    ax.set_ylabel("Frequency", color=COLOR_TEXT, fontsize=10)
    ax.set_title("Slippage Distribution", color=COLOR_TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=COLOR_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, fontsize=9)

    # --- Panel 2: Box plot ---
    ax = axes[1]
    bp = ax.boxplot(
        [gbm_totals, jd_totals],
        labels=["GBM", "Jump-Diffusion"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor(COLOR_GBM + "80")
    bp["boxes"][1].set_facecolor(COLOR_JD + "80")
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color(COLOR_TEXT)
    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor=COLOR_BSWAN, markersize=3, alpha=0.5)
    ax.set_ylabel("Total Slippage (bps)", color=COLOR_TEXT, fontsize=10)
    ax.set_title("Distribution Comparison", color=COLOR_TEXT, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", colors=COLOR_TEXT)

    # --- Panel 3: Regime breakdown ---
    ax = axes[2]
    regimes = ["NORMAL", "STRESS", "BLACK_SWAN"]
    gbm_counts = [gbm_results["regime_breakdown"][r] for r in regimes]
    jd_counts = [jd_results["regime_breakdown"][r] for r in regimes]
    x = np.arange(len(regimes))
    width = 0.35
    ax.bar(x - width / 2, gbm_counts, width, color=COLOR_GBM, alpha=0.8, label="GBM")
    ax.bar(x + width / 2, jd_counts, width, color=COLOR_JD, alpha=0.8, label="JD")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, color=COLOR_TEXT)
    ax.set_ylabel("Path Count", color=COLOR_TEXT, fontsize=10)
    ax.set_title("Regime Classification", color=COLOR_TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=COLOR_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, fontsize=9)

    fig.suptitle(
        f"Slippage Monte Carlo — {ticker}\n"
        f"GBM mean: {gbm_results['total_slippage']['mean']:.1f} bps  |  "
        f"JD mean: {jd_results['total_slippage']['mean']:.1f} bps  |  "
        f"Ratio: {jd_results['total_slippage']['mean'] / max(gbm_results['total_slippage']['mean'], 0.01):.2f}×",
        fontsize=13, fontweight="bold", color=COLOR_TEXT, y=1.02,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"slippage_montecarlo_{ticker}.png"
    fig.savefig(
        out_file, dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)

    return out_file


# ---------------------------------------------------------------------------
# 6. Main — WandB Instrumented Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full slippage Monte Carlo pipeline with WandB logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Slippage Monte Carlo — Pipeline Stage 3",
    )
    parser.add_argument(
        "--ticker", default="SPY", type=str.upper,
        help="Ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--n-paths", default=2000, type=int,
        help="Paths per mode (default: 2000, total = 4000)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Skip WandB logging (offline mode)",
    )
    args = parser.parse_args()

    ticker = args.ticker
    n_paths = args.n_paths

    # --- Resolve data file ---
    ticker_map = {
        "SPY": DATA_DIR / "Indices" / "SPY.csv",
        "NSEI": DATA_DIR / "Indices" / "NSEI.csv",
    }
    csv_path = ticker_map.get(ticker)
    if csv_path is None or not csv_path.exists():
        for name, path in ticker_map.items():
            if path.exists():
                ticker = name
                csv_path = path
                logger.warning("Falling back to %s", name)
                break
        else:
            logger.error("No CSV data found — run ingest_data.py first.")
            sys.exit(1)

    # --- Calibrate ---
    S0, mu, sigma = calibrate_from_data(csv_path)

    # --- Simulation parameters ---
    T = 1.0
    dt = 1.0 / 252
    lambda_j = 2.0
    mu_j = -0.15
    sigma_j = 0.10
    order_frac = 0.01
    eta = 0.3
    execution_window = 5

    config = {
        "ticker": ticker,
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "T": T,
        "dt": dt,
        "n_paths_per_mode": n_paths,
        "lambda_j": lambda_j,
        "mu_j": mu_j,
        "sigma_j": sigma_j,
        "order_frac": order_frac,
        "eta": eta,
        "execution_window": execution_window,
        "regime_multipliers": _REGIME_MULTIPLIERS,
    }

    # --- WandB init ---
    use_wandb = not args.no_wandb
    run = None
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project="CRIS",
                job_type="validation",
                name="TS-002-Slippage",
                config=config,
                tags=["post-fix", "2000-paths-validation"],
            )
        except Exception as e:
            logger.warning("WandB init failed (%s), continuing offline.", e)
            use_wandb = False

    print(f"\n{'='*60}")
    print(f"  SLIPPAGE MONTE CARLO — {ticker}")
    print(f"  {n_paths} GBM paths + {n_paths} Jump-Diffusion paths")
    print(f"{'='*60}\n")

    # --- Run GBM Monte Carlo ---
    print("⏳ Running GBM (Normal Market) simulation...")
    gbm_results = run_monte_carlo_slippage(
        S0, mu, sigma, n_paths=n_paths, T=T, dt=dt,
        order_frac=order_frac, eta=eta,
        execution_window=execution_window,
        mode="gbm", base_seed=42,
    )
    print(f"   ✓ GBM mean slippage: {gbm_results['total_slippage']['mean']:.2f} bps")

    # --- Run Jump-Diffusion Monte Carlo ---
    print("⏳ Running Jump-Diffusion (Black Swan) simulation...")
    jd_results = run_monte_carlo_slippage(
        S0, mu, sigma, n_paths=n_paths, T=T, dt=dt,
        order_frac=order_frac, eta=eta,
        execution_window=execution_window,
        mode="jump_diffusion",
        lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j,
        base_seed=10_042,
    )
    print(f"   ✓ JD  mean slippage: {jd_results['total_slippage']['mean']:.2f} bps")

    # --- Validation gate ---
    ratio = jd_results["total_slippage"]["mean"] / max(
        gbm_results["total_slippage"]["mean"], 0.01
    )
    passed = ratio > 1.5

    print(f"\n{'─'*60}")
    print(f"  VALIDATION: JD/GBM ratio = {ratio:.2f}×")
    print(f"  Required: > 1.5×")
    print(f"  Result: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"{'─'*60}")

    # --- Summary table ---
    print(f"\n{'─'*60}")
    print(f"  {'Metric':<25} {'GBM':>12} {'Jump-Diff':>12}")
    print(f"{'─'*60}")
    for key in ["mean", "median", "p95", "p99", "max", "std"]:
        g = gbm_results["total_slippage"][key]
        j = jd_results["total_slippage"][key]
        print(f"  {key.upper():<25} {g:>10.2f}bp {j:>10.2f}bp")
    print(f"{'─'*60}")
    print(f"  {'REGIME':<25} {'GBM':>12} {'Jump-Diff':>12}")
    for regime in ["NORMAL", "STRESS", "BLACK_SWAN"]:
        g = gbm_results["regime_breakdown"][regime]
        j = jd_results["regime_breakdown"][regime]
        print(f"  {regime:<25} {g:>12d} {j:>12d}")
    print(f"{'─'*60}\n")

    # --- Plot ---
    plot_path = plot_slippage_comparison(gbm_results, jd_results, ticker)
    print(f"📊 Plot saved → {plot_path}")

    # --- WandB logging (all dynamic, zero hardcoded) ---
    if use_wandb and run is not None:
        import wandb

        # Log scalar metrics
        wandb.log({
            "gbm_mean_slippage_bps": gbm_results["total_slippage"]["mean"],
            "gbm_p95_slippage_bps": gbm_results["total_slippage"]["p95"],
            "gbm_p99_slippage_bps": gbm_results["total_slippage"]["p99"],
            "gbm_max_slippage_bps": gbm_results["total_slippage"]["max"],
            "jd_mean_slippage_bps": jd_results["total_slippage"]["mean"],
            "jd_p95_slippage_bps": jd_results["total_slippage"]["p95"],
            "jd_p99_slippage_bps": jd_results["total_slippage"]["p99"],
            "jd_max_slippage_bps": jd_results["total_slippage"]["max"],
            "jd_gbm_ratio": ratio,
            "validation_passed": passed,
        })

        # Log comparison table
        columns = ["Mode", "Mean", "Median", "P95", "P99", "Max", "Std"]
        table_data = []
        for res in [gbm_results, jd_results]:
            s = res["total_slippage"]
            table_data.append([
                res["mode"], s["mean"], s["median"],
                s["p95"], s["p99"], s["max"], s["std"],
            ])
        wandb.log({"slippage_summary": wandb.Table(columns=columns, data=table_data)})

        # Log regime breakdown table
        regime_cols = ["Mode", "NORMAL", "STRESS", "BLACK_SWAN"]
        regime_data = []
        for res in [gbm_results, jd_results]:
            rb = res["regime_breakdown"]
            regime_data.append([res["mode"], rb["NORMAL"], rb["STRESS"], rb["BLACK_SWAN"]])
        wandb.log({"regime_breakdown": wandb.Table(columns=regime_cols, data=regime_data)})

        # Log the plot
        wandb.log({"slippage_comparison_plot": wandb.Image(str(plot_path))})

        wandb.finish()
        print(f"☁️  WandB run logged → {run.url}")

    print("\n✅ Slippage pipeline complete.\n")


if __name__ == "__main__":
    main()
