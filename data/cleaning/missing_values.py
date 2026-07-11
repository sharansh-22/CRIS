import pandas as pd
import logging

from data.errors import DatasetContractError

logger = logging.getLogger(__name__)

def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Handle missing values based on the dataset-specific strategy."""
    if df.empty:
        return df

    if strategy == "forward_fill":
        df = df.ffill()
    elif strategy == "drop":
        df = df.dropna()
    elif strategy == "mean_fill":
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == "none":
        pass
    else:
        raise DatasetContractError(f"unknown missing value strategy: {strategy}")
        
    return df
