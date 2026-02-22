"""
ingest_data.py — BSSC Historical Market Data Ingestion

Downloads daily OHLCV data for a curated set of tickers using
yfinance and persists each as a CSV inside asset-class sub-folders
under the project's data/ directory.

Directory layout after a run:
    data/
    ├── Indices/   (SPY, NSEI, VIX)
    ├── Equities/  (NVDA, TSLA, HDFCBANK_NS)
    └── Macro/     (GLD, TNX)

Usage:
    python data/ingest_data.py
"""

import logging
import sys
from pathlib import Path

import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolve project root and data directory relative to *this* script,
# so the output location is deterministic regardless of the caller's cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # CSVs live alongside this script in data/

START_DATE = "2018-01-01"
END_DATE = "2026-02-22"

TICKERS: dict[str, list[str]] = {
    "Indices": ["SPY", "^NSEI", "^VIX"],
    "Equities": ["NVDA", "TSLA", "HDFCBANK.NS"],
    "Macro": ["GLD", "^TNX"],
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_filename(ticker: str) -> str:
    """Convert a ticker symbol into a filesystem-safe CSV filename.

    Examples
    --------
    >>> _safe_filename("^VIX")
    'VIX.csv'
    >>> _safe_filename("HDFCBANK.NS")
    'HDFCBANK_NS.csv'
    """
    return ticker.replace("^", "").replace(".", "_") + ".csv"


def download_ticker(ticker: str, start: str, end: str, dest_dir: Path) -> None:
    """Download daily OHLCV data for *ticker* and save as CSV.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. ``"SPY"``, ``"^NSEI"``).
    start : str
        ISO-format start date.
    end : str
        ISO-format end date (exclusive in yfinance).
    dest_dir : Path
        Directory in which to write the CSV.
    """
    try:
        logger.info("Downloading %s …", ticker)
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            logger.warning("No data returned for %s — skipping.", ticker)
            return

        out_path = dest_dir / _safe_filename(ticker)
        df.to_csv(out_path)
        logger.info(
            "Downloaded %s: %d rows  →  %s",
            ticker,
            len(df),
            out_path.relative_to(SCRIPT_DIR.parent),
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to download %s: %s", ticker, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry-point: ensure output directories exist, then fetch all tickers."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Data directory: %s", DATA_DIR)

    all_tickers = [t for group in TICKERS.values() for t in group]
    logger.info(
        "Fetching %d tickers (%s → %s) …",
        len(all_tickers),
        START_DATE,
        END_DATE,
    )

    for category, symbols in TICKERS.items():
        category_dir = DATA_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        logger.info("— Category: %s  →  %s/", category, category)
        for ticker in symbols:
            download_ticker(ticker, START_DATE, END_DATE, category_dir)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    sys.exit(main() or 0)
