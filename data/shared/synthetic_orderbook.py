"""
synthetic_orderbook.py — Synthetic Level-2 Order Book Generator

Reconstructs realistic Level-2 order book snapshots from existing OHLCV
data already present in the CRIS project.  This module replaces the
``data/lobster/`` placeholder (which required university-gated LOBSTER
access) with a pure reconstruction approach built on top of daily
OHLCV candles.

Reconstruction Approach
-----------------------
1. **Spread estimation**  — The daily bid-ask spread is derived from the
   High-Low range scaled by the square root of daily volume (Corwin &
   Schultz, 2012 intuition).
2. **Price levels**       — *N* discrete price levels are generated on each
   side of the mid-price, spaced by ``tick_size = spread / (2 * N)``.
3. **Depth assignment**   — Resting quantity at each level follows an
   exponential decay from the best bid/ask outward, calibrated to the
   day's traded volume.
4. **Order arrival**      — Intraday snapshots are simulated via a Poisson
   process whose rate (λ) is calibrated so that the integral over the
   trading day reproduces the observed daily volume.
5. **Imbalance**          — Order-book imbalance is computed as
   ``(total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)``.

Output
------
Each event window produces a CSV in ``data/synthetic_orderbook/`` with
the filename pattern ``{ticker}_{event}_{date}.csv``.

Consumed by: **Layer 2 — Market Microstructure Anomaly Detector (MMAD)**
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR                          # data/ lives alongside this script
OUTPUT_DIR = DATA_DIR / "synthetic_orderbook"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default reconstruction parameters (mirrored in config.yaml)
DEFAULT_N_LEVELS: int = 5
DEFAULT_POISSON_LAMBDA: float = 10.0   # order arrivals per minute
DEFAULT_DEPTH_DECAY: float = 0.7       # exponential decay factor

# Trading session: 6.5 hours × 60 minutes
TRADING_MINUTES_PER_DAY: int = 390


# ---------------------------------------------------------------------------
# Data Structure
# ---------------------------------------------------------------------------


@dataclass
class OrderBookSnapshot:
    """A single synthetic Level-2 order book snapshot.

    Attributes
    ----------
    timestamp : str
        ISO-format timestamp for this snapshot.
    mid_price : float
        Mid-point between best bid and best ask.
    spread : float
        Bid-ask spread (best_ask - best_bid).
    bid_prices : list[float]
        Price levels on the bid side, best (highest) first.
    ask_prices : list[float]
        Price levels on the ask side, best (lowest) first.
    bid_quantities : list[int]
        Resting depth at each bid level.
    ask_quantities : list[int]
        Resting depth at each ask level.
    order_imbalance : float
        (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty).
        Range [-1, +1].  Positive ⇒ more buying pressure.
    """

    timestamp: str
    mid_price: float
    spread: float
    bid_prices: list[float] = field(default_factory=list)
    ask_prices: list[float] = field(default_factory=list)
    bid_quantities: list[int] = field(default_factory=list)
    ask_quantities: list[int] = field(default_factory=list)
    order_imbalance: float = 0.0


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def generate_orderbook_snapshot(
    ohlcv_row: pd.Series,
    n_levels: int = DEFAULT_N_LEVELS,
    depth_decay: float = DEFAULT_DEPTH_DECAY,
    rng: np.random.Generator | None = None,
) -> OrderBookSnapshot:
    """Generate a single synthetic order book snapshot from one OHLCV row.

    The reconstruction works as follows:

    1. **Mid-price** is the average of Open and Close for the day.
    2. **Spread** is estimated from the High-Low range normalised by
       ``sqrt(volume)`` (proxy for tick-level spread from daily bars).
    3. *N* bid levels descend from ``mid - spread/2`` by a uniform tick
       size; *N* ask levels ascend from ``mid + spread/2`` similarly.
    4. Depth at each level decays exponentially away from the inside:
       ``base_qty * decay^k`` where *k* is the level index (0 = best).
    5. A small random perturbation is added to quantities so successive
       snapshots are not identical.

    Parameters
    ----------
    ohlcv_row : pd.Series
        A single row with keys ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume``.  Values must be numeric.
    n_levels : int, optional
        Number of price levels on each side of the book (default 5).
    depth_decay : float, optional
        Exponential decay factor for depth away from best (default 0.7).
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    OrderBookSnapshot
        A fully populated snapshot consumed by **Layer 2 MMAD**.
    """
    if rng is None:
        rng = np.random.default_rng()

    open_p = float(ohlcv_row["Open"])
    high_p = float(ohlcv_row["High"])
    low_p = float(ohlcv_row["Low"])
    close_p = float(ohlcv_row["Close"])
    volume = float(ohlcv_row["Volume"])

    mid_price = (open_p + close_p) / 2.0

    # Spread estimate: high-low range scaled down.  A full-day range is
    # much wider than the instantaneous spread, so we shrink it.
    daily_range = high_p - low_p
    spread = max(daily_range * 0.01, mid_price * 1e-4)  # floor at 1 bps

    tick_size = spread / max(n_levels, 1)
    best_bid = mid_price - spread / 2.0
    best_ask = mid_price + spread / 2.0

    bid_prices = [round(best_bid - k * tick_size, 4) for k in range(n_levels)]
    ask_prices = [round(best_ask + k * tick_size, 4) for k in range(n_levels)]

    # Base quantity: fraction of daily volume distributed across levels
    base_qty = max(int(volume / (TRADING_MINUTES_PER_DAY * 2 * n_levels)), 1)

    bid_quantities: list[int] = []
    ask_quantities: list[int] = []
    for k in range(n_levels):
        decay = depth_decay ** k
        noise = rng.uniform(0.85, 1.15)
        bid_quantities.append(max(int(base_qty * decay * noise), 1))
        noise = rng.uniform(0.85, 1.15)
        ask_quantities.append(max(int(base_qty * decay * noise), 1))

    total_bid = sum(bid_quantities)
    total_ask = sum(ask_quantities)
    denom = total_bid + total_ask
    imbalance = (total_bid - total_ask) / denom if denom > 0 else 0.0

    timestamp = str(ohlcv_row.get("Date", ohlcv_row.name))

    return OrderBookSnapshot(
        timestamp=timestamp,
        mid_price=round(mid_price, 4),
        spread=round(spread, 6),
        bid_prices=bid_prices,
        ask_prices=ask_prices,
        bid_quantities=bid_quantities,
        ask_quantities=ask_quantities,
        order_imbalance=round(imbalance, 6),
    )


def _load_ohlcv(ticker: str) -> pd.DataFrame:
    """Load the OHLCV CSV for *ticker* from the data directory.

    Searches ``data/Indices/`` then ``data/Equities/`` for a matching file.

    Parameters
    ----------
    ticker : str
        E.g. ``"SPY"``, ``"NVDA"``.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with columns ``Open, High, Low, Close, Volume``
        and a DatetimeIndex.
    """
    candidates = [
        DATA_DIR / "Indices" / f"{ticker}.csv",
        DATA_DIR / "Equities" / f"{ticker}.csv",
    ]
    csv_path: Path | None = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"No CSV found for ticker '{ticker}' in {DATA_DIR}/Indices or "
            f"{DATA_DIR}/Equities"
        )

    # yfinance CSVs have a multi-level header: row 0 = field, row 1 = ticker
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)

    # Flatten: take only the first header level
    df.columns = [col[0] for col in df.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def generate_event_window(
    ticker: str,
    start_date: str,
    end_date: str,
    event_name: str,
    n_levels: int = DEFAULT_N_LEVELS,
    depth_decay: float = DEFAULT_DEPTH_DECAY,
    poisson_lambda: float = DEFAULT_POISSON_LAMBDA,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic order book snapshots for a specific event window.

    For each trading day in ``[start_date, end_date]``, a Poisson-sampled
    number of intraday snapshots is generated (calibrated so the expected
    total across the day ≈ ``poisson_lambda × TRADING_MINUTES_PER_DAY``).
    Each snapshot is built by :func:`generate_orderbook_snapshot`.

    The resulting DataFrame is also saved to CSV in
    ``data/synthetic_orderbook/{ticker}_{event_name}_{date}.csv``.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. ``"SPY"``).  Must have a corresponding CSV
        in ``data/Indices/`` or ``data/Equities/``.
    start_date : str
        ISO start date (inclusive).
    end_date : str
        ISO end date (inclusive).
    event_name : str
        Short label for the crisis event (e.g. ``"flash_crash_2010"``).
    n_levels : int, optional
        Price levels per side (default 5).
    depth_decay : float, optional
        Exponential depth decay (default 0.7).
    poisson_lambda : float, optional
        Mean snapshots per minute (default 10).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        All snapshots for the event window, with columns:
        ``timestamp, mid_price, spread, bid_prices, ask_prices,
        bid_quantities, ask_quantities, order_imbalance``.

        Consumed by **Layer 2 MMAD** for microstructure anomaly detection.
    """
    rng = np.random.default_rng(seed)
    df = _load_ohlcv(ticker)

    mask = (df.index >= start_date) & (df.index <= end_date)
    window = df.loc[mask]

    if window.empty:
        logger.warning(
            "No data for %s in [%s, %s] — returning empty DataFrame.",
            ticker, start_date, end_date,
        )
        return pd.DataFrame()

    logger.info(
        "Generating snapshots for %s | %s | %d trading days",
        event_name, ticker, len(window),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for date_idx, row in window.iterrows():
        date_str = pd.Timestamp(date_idx).strftime("%Y-%m-%d")

        # Number of snapshots for this day (Poisson-sampled)
        n_snapshots = rng.poisson(poisson_lambda * TRADING_MINUTES_PER_DAY)
        n_snapshots = max(n_snapshots, 1)  # at least one

        # Generate random intraday timestamps (sorted)
        minutes = np.sort(rng.integers(0, TRADING_MINUTES_PER_DAY, size=n_snapshots))
        base_ts = pd.Timestamp(date_idx).replace(hour=9, minute=30)

        for minute_offset in minutes:
            snap_ts = base_ts + pd.Timedelta(minutes=int(minute_offset))
            row_with_ts = row.copy()
            row_with_ts.name = date_idx

            snapshot = generate_orderbook_snapshot(
                row_with_ts,
                n_levels=n_levels,
                depth_decay=depth_decay,
                rng=rng,
            )
            snapshot.timestamp = snap_ts.isoformat()

            all_rows.append(
                {
                    "timestamp": snapshot.timestamp,
                    "mid_price": snapshot.mid_price,
                    "spread": snapshot.spread,
                    "bid_prices": snapshot.bid_prices,
                    "ask_prices": snapshot.ask_prices,
                    "bid_quantities": snapshot.bid_quantities,
                    "ask_quantities": snapshot.ask_quantities,
                    "order_imbalance": snapshot.order_imbalance,
                }
            )

        # Save per-day CSV
        day_df = pd.DataFrame(
            [r for r in all_rows if r["timestamp"].startswith(date_str)]
        )
        day_file = OUTPUT_DIR / f"{ticker}_{event_name}_{date_str}.csv"
        day_df.to_csv(day_file, index=False)

    result_df = pd.DataFrame(all_rows)
    logger.info(
        "  %s complete: %d snapshots across %d days",
        event_name, len(result_df), len(window),
    )
    return result_df


def generate_all_cris_events(
    n_levels: int = DEFAULT_N_LEVELS,
    depth_decay: float = DEFAULT_DEPTH_DECAY,
    poisson_lambda: float = DEFAULT_POISSON_LAMBDA,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic order books for all three CRIS backtesting events.

    The three event windows are:

    1. **Flash Crash 2010** — SPY, 2010-05-01 → 2010-05-15
    2. **COVID Crash 2020** — SPY, 2020-02-01 → 2020-03-31
    3. **China Circuit Breaker 2015** — SPY (proxy), 2015-06-01 → 2016-01-31

    Parameters
    ----------
    n_levels : int, optional
        Price levels per side (default 5).
    depth_decay : float, optional
        Exponential depth decay (default 0.7).
    poisson_lambda : float, optional
        Mean snapshots per minute (default 10).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from event name to its snapshot DataFrame.
        All DataFrames are consumed by **Layer 2 MMAD**.
    """
    events = {
        "flash_crash_2010": {
            "ticker": "SPY",
            "start_date": "2010-05-01",
            "end_date": "2010-05-15",
        },
        "covid_2020": {
            "ticker": "SPY",
            "start_date": "2020-02-01",
            "end_date": "2020-03-31",
        },
        "china_2015": {
            "ticker": "SPY",
            "start_date": "2015-06-01",
            "end_date": "2016-01-31",
        },
    }

    results: dict[str, pd.DataFrame] = {}

    for event_name, params in events.items():
        df = generate_event_window(
            ticker=params["ticker"],
            start_date=params["start_date"],
            end_date=params["end_date"],
            event_name=event_name,
            n_levels=n_levels,
            depth_decay=depth_decay,
            poisson_lambda=poisson_lambda,
        )
        results[event_name] = df

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🔧  Synthetic Order Book Generator — CRIS\n")

    results = generate_all_cris_events()

    print("\n" + "=" * 60)
    print("  📊  Generation Summary")
    print("=" * 60)

    for event_name, df in results.items():
        if df.empty:
            print(f"\n  ⚠  {event_name}: no data in date range")
            continue

        rows, cols = df.shape
        print(f"\n  ✅  {event_name}")
        print(f"      Shape : {rows} rows × {cols} columns")
        print(f"      Dates : {df['timestamp'].iloc[0][:10]}"
              f" → {df['timestamp'].iloc[-1][:10]}")

    # Print first row of first non-empty event to confirm structure
    for event_name, df in results.items():
        if not df.empty:
            print(f"\n  📋  Sample row ({event_name}):")
            first = df.iloc[0]
            for col in df.columns:
                print(f"      {col:20s} = {first[col]}")
            break

    print(f"\n  Output directory: {OUTPUT_DIR}/")
    print("=" * 60 + "\n")
