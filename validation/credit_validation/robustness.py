"""
Temporal Robustness and Stress Testing for Credit Risk Pipeline
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.credit_config import OUTPUT_DIR, MODEL_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_by_year(model, df, model_name, is_scaled=False, scaler=None):
    """Compute AUC for each year to detect performance drift."""
    logger.info(f"Evaluating temporal robustness for {model_name}...")
    
    years = sorted(df['issue_d'].dt.year.unique())
    results = []
    
    for year in years:
        year_df = df[df['issue_d'].dt.year == year]
        if len(year_df) < 100: continue
        
        X = year_df.drop(columns=['target', 'issue_d'])
        y = year_df['target']
        
        if is_scaled:
            X = scaler.transform(X)
            
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        results.append({'Year': year, 'AUC': auc})
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load full engineered data (before split)
    data_path = OUTPUT_DIR / "engineered_data.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        
        lgb = joblib.load(MODEL_DIR / "lightgbm.joblib")
        robustness_df = evaluate_by_year(lgb, df, "LightGBM")
        
        robustness_df.to_csv(OUTPUT_DIR / "temporal_robustness.csv", index=False)
        logger.info("Temporal robustness evaluation complete.")
    else:
        logger.error("Engineered data not found.")
