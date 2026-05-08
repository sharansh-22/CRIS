"""
Data Ingestion and Cleaning for Credit Risk Pipeline
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.credit_config import RAW_DATA_PATH, LEAKAGE_COLS, TARGET_COL, GOOD_STATUS, BAD_STATUS, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(nrows=None):
    """Load LendingClub data from compressed CSV."""
    logger.info(f"Loading data from {RAW_DATA_PATH}...")
    try:
        # Load only necessary columns to save memory if possible, 
        # but for cleaning we might need to see many columns first.
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False, nrows=nrows)
        logger.info(f"Loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean the dataset: handle missing values, leakage, and target construction."""
    logger.info("Starting data cleaning...")
    
    # 1. Filter by target status
    initial_count = len(df)
    df = df[df[TARGET_COL].isin(GOOD_STATUS + BAD_STATUS)].copy()
    logger.info(f"Filtered to {len(df)} rows (from {initial_count}) with statuses: {GOOD_STATUS + BAD_STATUS}")
    
    # 2. Construct Target Label
    df['target'] = df[TARGET_COL].apply(lambda x: 1 if x in BAD_STATUS else 0)
    
    # 3. Remove Leakage Columns
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    # Also drop columns with > 50% missing values
    missing_pct = df.isnull().mean()
    high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
    cols_to_drop = list(set(cols_to_drop + high_missing_cols + [TARGET_COL]))
    
    df_clean = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} columns (leakage + high missing + target source).")
    
    # 4. Temporal Parsing
    if 'issue_d' in df_clean.columns:
        df_clean['issue_d'] = pd.to_datetime(df_clean['issue_d'], format='%b-%Y')
        logger.info("Parsed 'issue_d' to datetime.")
    
    # 5. Handle remaining missing values (simple imputation for now)
    # Categorical: fill with 'Unknown'
    # Numerical: fill with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    df_clean[cat_cols] = df_clean[cat_cols].fillna('Unknown')
    
    logger.info("Imputed missing values.")
    
    return df_clean, cols_to_drop

def generate_quality_report(df, dropped_cols):
    """Generate a markdown report on data quality."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "data_quality_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Data Quality Report - Credit Risk Phase 1\n\n")
        f.write(f"**Total Records Processed:** {len(df)}\n")
        f.write(f"**Target Distribution:**\n")
        dist = df['target'].value_counts(normalize=True)
        f.write(f"- Good Loans (0): {dist.get(0, 0):.2%}\n")
        f.write(f"- Bad Loans (1): {dist.get(1, 0):.2%}\n\n")
        
        f.write("## Removed Features (Leakage Prevention)\n")
        f.write("The following features were removed because they contain post-issuance information or are identifiers:\n\n")
        for col in sorted(dropped_cols):
            f.write(f"- {col}\n")
            
        f.write("\n## Missing Value Strategy\n")
        f.write("- Columns with > 50% missing values were dropped.\n")
        f.write("- Remaining numerical columns were imputed with median.\n")
        f.write("- Remaining categorical columns were imputed with 'Unknown'.\n")
        
    logger.info(f"Data quality report generated at {report_path}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df_clean, dropped = clean_data(df)
        generate_quality_report(df_clean, dropped)
        # Save intermediate cleaned data
        df_clean.to_parquet(OUTPUT_DIR / "cleaned_data.parquet", index=False)
        logger.info("Saved cleaned data to parquet.")
