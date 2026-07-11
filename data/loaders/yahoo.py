from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from data.contracts import RawDataset
from data.errors import DataSourceError

logger = logging.getLogger(__name__)

def load_yahoo_data(dataset_name: str, tickers: list[str], start_date: str = "2005-01-01", end_date: str | None = None, retries: int = 3) -> RawDataset:
    """Download OHLCV data for multiple tickers using yfinance and return a RawDataset."""
    logger.info(f"Downloading {len(tickers)} tickers from Yahoo Finance for dataset: {dataset_name}")
    
    all_data = []
    source_urls: list[str] = []
    provider_version = getattr(yf, "__version__", "unknown")
    
    for ticker in tickers:
        df = pd.DataFrame()
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
                if not df.empty:
                    break
            except Exception as exc:
                last_error = exc
                logger.warning(f"Attempt {attempt} failed for {ticker}: {exc}")
        if df.empty:
            if last_error is not None:
                logger.error(f"Failed to download {ticker}: {last_error}")
            else:
                logger.warning(f"Download returned empty DataFrame for {ticker}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.reset_index()

        if "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)

        df["Ticker"] = ticker
        df["__row_id"] = [f"{ticker}:{idx}" for idx in range(len(df))]
        all_data.append(df)
        source_urls.append(f"https://finance.yahoo.com/quote/{ticker}/history")
            
    if not all_data:
        raise DataSourceError(f"failed to download any Yahoo data for {dataset_name}")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return RawDataset(
        dataset_name=dataset_name,
        data=combined_df,
        fetch_timestamp=datetime.utcnow(),
        provider="yahoo",
        provider_version=provider_version,
        source_url=", ".join(source_urls),
        source_metadata={"tickers": tickers, "start_date": start_date, "end_date": end_date},
        source_row_count=len(combined_df),
        source_column_count=len(combined_df.columns),
    )
