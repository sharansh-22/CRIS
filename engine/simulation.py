"""
simulation.py — Phase 2: Merton Jump-Diffusion Stress Test

Calibrates baseline volatility from historical OHLCV data, then generates
side-by-side price paths comparing "Normal" (GBM) vs "Black Swan"
(Jump-Diffusion) market regimes.

Mathematical Background
-----------------------
Geometric Brownian Motion (GBM):
    S(t+dt) = S(t) · exp[(μ − σ²/2)·dt + σ·√dt·Z],   Z ~ N(0,1)

Merton Jump-Diffusion extends GBM with a compound Poisson process:
    S(t+dt) = S_gbm(t+dt) · exp(J · N_jump)
    where  N_jump ~ Poisson(λ·dt)   and   J ~ N(μ_j, σ_j²)

The jumps inject *non-Gaussian* shocks, producing fat tails (excess
kurtosis) and negative skewness — the statistical fingerprints of
"Black Swan" events that standard Normal models completely miss.

Usage
-----
    python -m engine.simulation                # uses SPY by default
    python -m engine.simulation --ticker NSEI  # use NSEI instead
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib

# Use Agg backend when no display is available (headless / CI)
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "simulation_output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Trading days per year (standard for US markets)
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# 1. Calibration — Estimate μ and σ from Historical Data
# ---------------------------------------------------------------------------


def calibrate_from_data(csv_path: str | Path) -> tuple[float, float, float]:
    """Estimate annualised drift and volatility from a yfinance CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to a yfinance-formatted CSV (multi-level headers).

    Returns
    -------
    S0 : float
        Last observed closing price (simulation start point).
    mu : float
        Annualised drift (mean of daily log-returns × 252).
    sigma : float
        Annualised volatility (std of daily log-returns × √252).
    """
    csv_path = Path(csv_path)
    logger.info("Calibrating from %s …", csv_path.name)

    # yfinance CSVs have a multi-level header: row 0 = field, row 1 = ticker
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)

    # Flatten multi-level columns — take only the first level
    close_col = [c for c in df.columns if c[0] == "Close"]
    if not close_col:
        raise ValueError(f"No 'Close' column found in {csv_path.name}")

    close = df[close_col[0]].dropna().astype(float)

    # Daily log-returns: r_t = ln(S_t / S_{t-1})
    log_returns = np.log(close / close.shift(1)).dropna()

    S0 = float(close.iloc[-1])
    mu = float(log_returns.mean()) * TRADING_DAYS_PER_YEAR
    sigma = float(log_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

    logger.info("  S₀ = %.2f  |  μ = %.4f  |  σ = %.4f", S0, mu, sigma)
    return S0, mu, sigma


# ---------------------------------------------------------------------------
# 2. Geometric Brownian Motion (GBM) — "Normal" Market
# ---------------------------------------------------------------------------


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float = 1.0,
    dt: float = 1.0 / 252,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a single Geometric Brownian Motion price path.

    Uses the exact log-normal discretization:
        S(t+dt) = S(t) · exp[(μ − σ²/2)·dt + σ·√dt·Z]

    Parameters
    ----------
    S0 : float     Initial price.
    mu : float     Annualised drift.
    sigma : float  Annualised volatility.
    T : float      Horizon in years (default 1).
    dt : float     Time-step in years (default 1/252 = one trading day).
    seed : int     Random seed for reproducibility.

    Returns
    -------
    t : ndarray    Time array (in trading days).
    S : ndarray    Simulated price path.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)

    t = np.arange(n_steps + 1)
    S = np.zeros(n_steps + 1)
    S[0] = S0

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    for i in range(1, n_steps + 1):
        Z = rng.standard_normal()
        S[i] = S[i - 1] * np.exp(drift + diffusion * Z)

    return t, S


# ---------------------------------------------------------------------------
# 3. Merton Jump-Diffusion — "Black Swan" Market
# ---------------------------------------------------------------------------


def simulate_jumps(
    S0: float,
    mu: float,
    sigma: float,
    lambda_j: float = 2.0,
    mu_j: float = -0.15,
    sigma_j: float = 0.10,
    T: float = 1.0,
    dt: float = 1.0 / 252,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Simulate a Merton Jump-Diffusion price path.

    Extends GBM with a compound Poisson jump component:
        At each step, a jump occurs with probability  λ·dt.
        If a jump fires, the price is shocked by  exp(J),
        where  J ~ N(μ_j, σ_j²).

    Parameters
    ----------
    S0 : float       Initial price.
    mu : float       Annualised drift (from calibration).
    sigma : float    Annualised baseline volatility.
    lambda_j : float Jump intensity — expected number of jumps per year.
    mu_j : float     Mean jump size in log-space (negative = crash).
    sigma_j : float  Jump-size volatility in log-space.
    T : float        Horizon in years.
    dt : float       Time-step in years (1/252 = daily).
    seed : int       Random seed for reproducibility.

    Returns
    -------
    t : ndarray          Time array (trading days).
    S : ndarray          Simulated price path.
    jump_indices : list  Indices where jumps occurred (for annotation).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)

    t = np.arange(n_steps + 1)
    S = np.zeros(n_steps + 1)
    S[0] = S0

    # Compensated drift: subtract the expected jump contribution so
    # the drift remains comparable to GBM under no-arbitrage.
    jump_compensator = lambda_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    drift = (mu - 0.5 * sigma**2 - jump_compensator) * dt
    diffusion = sigma * np.sqrt(dt)

    jump_indices: list[int] = []

    for i in range(1, n_steps + 1):
        # --- Diffusion component (same as GBM) ---
        Z = rng.standard_normal()
        diffusion_factor = np.exp(drift + diffusion * Z)

        # --- Jump component (compound Poisson) ---
        n_jumps = rng.poisson(lambda_j * dt)
        jump_factor = 1.0
        if n_jumps > 0:
            J = rng.normal(mu_j, sigma_j, size=n_jumps).sum()
            jump_factor = np.exp(J)
            jump_indices.append(i)

        S[i] = S[i - 1] * diffusion_factor * jump_factor

    logger.info(
        "  Jump-Diffusion: %d jumps occurred over %d steps",
        len(jump_indices),
        n_steps,
    )
    return t, S, jump_indices


# ---------------------------------------------------------------------------
# 4. Visualisation — Education-Ready Comparison Plot
# ---------------------------------------------------------------------------


def plot_comparison(
    t: np.ndarray,
    gbm_path: np.ndarray,
    jd_path: np.ndarray,
    jump_indices: list[int],
    ticker_name: str,
    S0: float,
    mu: float,
    sigma: float,
    output_dir: Path | None = None,
) -> None:
    """Generate a side-by-side comparison plot: GBM vs Jump-Diffusion.

    Parameters
    ----------
    t : ndarray          Shared time axis (trading days).
    gbm_path : ndarray   GBM price path.
    jd_path : ndarray    Jump-Diffusion price path.
    jump_indices : list   Indices where jumps occurred.
    ticker_name : str     Ticker label for the title.
    S0, mu, sigma : float Calibration parameters (for subtitle).
    output_dir : Path     Directory to save the PNG (created if needed).
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 6.5), sharey=True, dpi=120
    )
    fig.patch.set_facecolor("#0d1117")

    # --- Colour palette (dark-mode, premium feel) ---
    COLOR_BG = "#0d1117"
    COLOR_GRID = "#21262d"
    COLOR_TEXT = "#c9d1d9"
    COLOR_GBM = "#58a6ff"       # cool blue
    COLOR_JD = "#f0883e"        # warm amber
    COLOR_JUMP = "#f85149"      # vivid red for jump markers

    for ax in (ax1, ax2):
        ax.set_facecolor(COLOR_BG)
        ax.tick_params(colors=COLOR_TEXT, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_GRID)
        ax.spines["bottom"].set_color(COLOR_GRID)
        ax.grid(True, alpha=0.15, color=COLOR_GRID, linestyle="--")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
        )

    # --- Plot A: GBM ("Normal") ---
    ax1.plot(t, gbm_path, color=COLOR_GBM, linewidth=1.3, alpha=0.9)
    ax1.fill_between(t, gbm_path.min() * 0.98, gbm_path, alpha=0.06,
                      color=COLOR_GBM)
    ax1.set_title(
        'Plot A — "Normal" Market (GBM)',
        fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12,
    )
    ax1.set_xlabel("Trading Days", fontsize=11, color=COLOR_TEXT, labelpad=8)
    ax1.set_ylabel("Simulated Price", fontsize=11, color=COLOR_TEXT, labelpad=8)

    # --- Plot B: Jump-Diffusion ("Black Swan") ---
    ax2.plot(
        t, jd_path, color=COLOR_JD, linewidth=1.3, alpha=0.9,
        label="Price Path",
    )
    ax2.fill_between(t, jd_path.min() * 0.98, jd_path, alpha=0.06,
                      color=COLOR_JD)

    # Annotate jump events
    if jump_indices:
        jump_t = t[jump_indices]
        jump_s = jd_path[jump_indices]
        ax2.scatter(
            jump_t, jump_s,
            color=COLOR_JUMP, s=40, zorder=5, edgecolors="white",
            linewidths=0.5, label="Jump Events",
        )
        # Annotate only the largest |jump| to avoid clutter
        log_returns_at_jumps = np.log(
            jd_path[jump_indices] / jd_path[np.array(jump_indices) - 1]
        )
        worst_idx = int(np.argmin(log_returns_at_jumps))
        ax2.annotate(
            f"  ⚡ {log_returns_at_jumps[worst_idx]*100:+.1f}%",
            xy=(jump_t[worst_idx], jump_s[worst_idx]),
            fontsize=9, fontweight="bold", color=COLOR_JUMP,
            xytext=(10, -20), textcoords="offset points",
            arrowprops=dict(
                arrowstyle="->", color=COLOR_JUMP, lw=1.2,
            ),
        )

    ax2.set_title(
        'Plot B — "Black Swan" (Jump-Diffusion)',
        fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12,
    )
    ax2.set_xlabel("Trading Days", fontsize=11, color=COLOR_TEXT, labelpad=8)
    ax2.legend(
        loc="upper left", fontsize=9, facecolor=COLOR_BG,
        edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT,
    )

    # --- Suptitle with calibration info ---
    fig.suptitle(
        f"Merton Jump-Diffusion Stress Test — {ticker_name}\n"
        f"Calibrated:  S₀ = ${S0:,.2f}   |   μ = {mu:.4f}   |   σ = {sigma:.4f}",
        fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.02,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save output
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"jump_diffusion_{ticker_name}.png"
    fig.savefig(
        out_file, dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    logger.info("Plot saved → %s", out_file)

    # Only call plt.show() when an interactive backend is available
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Statistical Summary — Fat Tails vs Normal
# ---------------------------------------------------------------------------


def print_return_statistics(
    gbm_path: np.ndarray, jd_path: np.ndarray
) -> None:
    """Print a comparative table of return statistics.

    Demonstrates how jumps create *fat tails* (excess kurtosis) and
    asymmetry (negative skewness) that GBM fundamentally cannot produce.
    """
    def _stats(prices: np.ndarray) -> dict:
        rets = np.diff(np.log(prices))
        return {
            "Mean (daily)": f"{rets.mean():.6f}",
            "Std  (daily)": f"{rets.std():.6f}",
            "Skewness":     f"{stats.skew(rets):.4f}",
            "Kurtosis":     f"{stats.kurtosis(rets):.4f}",
            "Min return":   f"{rets.min():.4f}",
            "Max return":   f"{rets.max():.4f}",
        }

    gbm_s = _stats(gbm_path)
    jd_s = _stats(jd_path)

    header = f"\n{'Statistic':<20} {'GBM (Normal)':>14}  {'Jump-Diffusion':>14}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print("  📊  Return Distribution Comparison")
    print(sep)
    print(header)
    print(sep)
    for key in gbm_s:
        print(f"  {key:<18} {gbm_s[key]:>14}  {jd_s[key]:>14}")
    print(sep)
    print(
        "  ➜  Kurtosis > 0 means FATTER TAILS than a Normal distribution.\n"
        "  ➜  Negative skewness means more extreme LEFT-tail (crash) events.\n"
        "  ➜  These are the statistical fingerprints of Black Swan risk.\n"
    )


# ---------------------------------------------------------------------------
# 6. Main — Standalone Execution
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full calibration → simulation → visualisation pipeline."""

    # --- Resolve data file ---
    ticker_map = {
        "SPY": DATA_DIR / "Indices" / "SPY.csv",
        "NSEI": DATA_DIR / "Indices" / "NSEI.csv",
    }

    # Use CLI arg if provided:  python -m engine.simulation --ticker NSEI
    ticker_name = "SPY"
    if "--ticker" in sys.argv:
        idx = sys.argv.index("--ticker")
        if idx + 1 < len(sys.argv):
            ticker_name = sys.argv[idx + 1].upper()

    csv_path = ticker_map.get(ticker_name)
    if csv_path is None or not csv_path.exists():
        # Fallback chain
        for name, path in ticker_map.items():
            if path.exists():
                ticker_name = name
                csv_path = path
                logger.warning(
                    "Requested ticker not found, falling back to %s", name
                )
                break
        else:
            logger.error("No CSV data found in %s — run ingest_data.py first.", DATA_DIR)
            sys.exit(1)

    # --- 1. Calibrate from historical data ---
    S0, mu, sigma = calibrate_from_data(csv_path)

    # --- 2. Simulation parameters ---
    T = 1.0            # 1 year
    dt = 1.0 / 252     # daily steps
    lambda_j = 2.0     # ~2 jumps per year
    mu_j = -0.15       # mean jump = −15 % (crash bias)
    sigma_j = 0.10     # jump-size volatility

    print(f"\n🔬  Simulation Parameters")
    print(f"   Horizon       : {T} year ({int(T/dt)} trading days)")
    print(f"   Jump intensity: λ = {lambda_j} jumps/year")
    print(f"   Jump mean     : μ_j = {mu_j} (log-space)")
    print(f"   Jump vol      : σ_j = {sigma_j}")
    print()

    # --- 3. Run simulations ---
    t, gbm_path = simulate_gbm(S0, mu, sigma, T, dt, seed=42)
    t, jd_path, jump_indices = simulate_jumps(
        S0, mu, sigma, lambda_j, mu_j, sigma_j, T, dt, seed=42
    )

    # --- 4. Visualise ---
    plot_comparison(t, gbm_path, jd_path, jump_indices, ticker_name, S0, mu, sigma)

    # --- 5. Statistical comparison ---
    print_return_statistics(gbm_path, jd_path)


if __name__ == "__main__":
    main()
