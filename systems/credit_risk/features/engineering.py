"""
Feature Engineering for Credit Risk Pipeline
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.credit_config import OUTPUT_DIR, SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def engineer_features(df):
    """Transform raw cleaned data into model-ready features."""
    logger.info("Starting feature engineering...")
    
    # 1. Emp Length Conversion
    def clean_emp_length(x):
        if x == 'Unknown': return 0
        x = str(x).replace(' years', '').replace(' year', '')
        if '< 1' in x: return 0.5
        if '10+' in x: return 10
        try:
            return float(x)
        except:
            return 0
    
    if 'emp_length' in df.columns:
        df['emp_length_num'] = df['emp_length'].apply(clean_emp_length)
        df = df.drop(columns=['emp_length'])
    
    # 2. Credit History Length
    if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
        df['cr_hist_years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
        df['cr_hist_years'] = df['cr_hist_years'].fillna(df['cr_hist_years'].median())
        df = df.drop(columns=['earliest_cr_line'])
    
    # 3. Term Conversion
    if 'term' in df.columns:
        df['term_months'] = df['term'].apply(lambda x: 36 if '36' in str(x) else 60)
        df = df.drop(columns=['term'])
        
    # 4. Encoding Categorical Features
    # Use simple frequency encoding or dummy encoding for high-cardinality?
    # For now, let's use Label Encoding for simplicity in this baseline, or One-Hot for small sets.
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'emp_title' in cat_cols: cat_cols.remove('emp_title') # Too high cardinality
    if 'issue_d' in df.columns: # Keep issue_d for temporal split, then drop
        pass
    
    logger.info(f"Encoding categorical columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Drop emp_title for now as it's too noisy
    if 'emp_title' in df.columns:
        df = df.drop(columns=['emp_title'])
        
    return df

def generate_feature_report(df):
    """Document feature engineering results."""
    report_path = OUTPUT_DIR / "feature_engineering_report.md"
    with open(report_path, "w") as f:
        f.write("# Feature Engineering Report\n\n")
        f.write("## Transformations\n")
        f.write("- **emp_length**: Converted from string (e.g., '10+ years') to numeric (0-10).\n")
        f.write("- **earliest_cr_line**: Converted to `cr_hist_years` relative to `issue_d`.\n")
        f.write("- **term**: Converted to numeric months (36 or 60).\n")
        f.write("- **Categorical Variables**: One-hot encoded (grade, home_ownership, verification_status, purpose, application_type).\n")
        f.write("- **emp_title**: Dropped due to high cardinality.\n\n")
        f.write(f"## Final Feature Set\n")
        f.write(f"Total Features: {len(df.columns) - 2} (excluding target and issue_d)\n")
        f.write("Top 10 columns by name:\n\n")
        for col in sorted(df.columns)[:10]:
            f.write(f"- {col}\n")
            
    logger.info(f"Feature report generated at {report_path}")

if __name__ == "__main__":
    data_path = OUTPUT_DIR / "cleaned_data.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
        df_feat = engineer_features(df)
        generate_feature_report(df_feat)
        df_feat.to_parquet(OUTPUT_DIR / "engineered_data.parquet", index=False)
        logger.info("Saved engineered data to parquet.")
    else:
        logger.error("Cleaned data not found.")
