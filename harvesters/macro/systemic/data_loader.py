"""
data_loader.py — Safe walk-forward data loading for systemic market structure harvesters.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

# Dynamic project root discovery
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "environment.yml").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

logger = logging.getLogger("CRIS.systemic.data_loader")

SECTOR_ETFS = ["XLY", "XLI", "XLB", "XLF", "XLE", "XLU", "XLP", "XLV", "XLRE"]

def ensure_sector_etfs_downloaded():
    """Ensure the sector ETFs are downloaded and formatted for fetch_historical_safe_data."""
    data_dir = PROJECT_ROOT / "data" / "macro"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    missing_tickers = [t for t in SECTOR_ETFS if not (data_dir / f"{t}.csv").exists()]
    if not missing_tickers:
        return
        
    logger.info(f"Downloading missing sector ETFs for Market Structure: {missing_tickers}")
    for ticker in missing_tickers:
        file_path = data_dir / f"{ticker}.csv"
        try:
            # Download a wide historical window to support all historical simulation dates
            df = yf.download(ticker, start="2005-01-01", auto_adjust=True, progress=False)
            if df.empty:
                logger.warning(f"Download returned empty DataFrame for {ticker}")
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df.reset_index()
            
            # Keep only the columns we need: Date, Close, High, Low, Open, Volume
            df = df[["Date", "Close", "High", "Low", "Open", "Volume"]].dropna()
            
            # Write custom headers (to match the skiprows=3 expected by fetch_historical_safe_data)
            with open(file_path, "w") as f:
                f.write("Price,Close,High,Low,Open,Volume\n")
                f.write(f"Ticker,{ticker},{ticker},{ticker},{ticker},{ticker}\n")
                f.write("Date,,,,,\n")
                for idx, row in df.iterrows():
                    date_str = row["Date"].strftime("%Y-%m-%d")
                    f.write(f"{date_str},{row['Close']},{row['High']},{row['Low']},{row['Open']},{row['Volume']}\n")
                    
            logger.info(f"Successfully downloaded and cached {ticker}.csv")
        except Exception as e:
            logger.error(f"Failed to download/format {ticker}: {str(e)}")

def fetch_sector_historical_safe_data(ticker: str, as_of_date: str) -> pd.DataFrame:
    """Fetch and slice data for a single ticker up to as_of_date."""
    file_path = PROJECT_ROOT / "data" / "macro" / f"{ticker}.csv"
    if not file_path.exists():
        ensure_sector_etfs_downloaded()
        if not file_path.exists():
            raise ValueError(f"Failed to fetch data for {ticker}. File {file_path} not found.")
    
    df = pd.read_csv(file_path, skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[df.index <= pd.to_datetime(as_of_date)]
    return df

def get_available_sector_data(as_of_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve returns and prices dataframes for all active sector ETFs as of as_of_date."""
    ensure_sector_etfs_downloaded()
    
    active_prices = {}
    for ticker in SECTOR_ETFS:
        try:
            df = fetch_sector_historical_safe_data(ticker, as_of_date)
            # We require at least 252 days of history to compute robust indicators (like 50-day SMA, 63-day correlation)
            if len(df) >= 252:
                active_prices[ticker] = df["Close"]
            else:
                logger.debug(f"Excluding sector {ticker} at {as_of_date} due to insufficient history ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Error loading sector {ticker} at {as_of_date}: {str(e)}")
            
    if not active_prices:
        raise ValueError(f"No active sector ETFs found with sufficient history prior to {as_of_date}")
        
    prices_df = pd.DataFrame(active_prices).sort_index()
    returns_df = prices_df.pct_change().dropna()
    return returns_df, prices_df
