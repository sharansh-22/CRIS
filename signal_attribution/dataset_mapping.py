"""
dataset_mapping.py — Cleans, fits baseline borrower PDs, maps, and merges macro states
for Give Me Some Credit (GMC) and Taiwan Bankruptcy (TB) datasets.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from configs.credit_config import SEED

logger = logging.getLogger("CRIS.SAE.dataset_mapping")


def get_macro_timeline_weights(macro_df: pd.DataFrame) -> tuple:
    """Compute sampling weights based on macro stress score for default/non-default loans."""
    macro_df = macro_df.copy()
    macro_df["issue_month"] = pd.to_datetime(macro_df["issue_month"]).dt.strftime("%Y-%m-%d")
    
    mss = macro_df["macro_stress_score"].values
    mss_min, mss_max = mss.min(), mss.max()
    # Scaled to [0.1, 0.9] to prevent zero probability
    mss_scaled = 0.1 + 0.8 * (mss - mss_min) / (mss_max - mss_min)
    
    p_def = mss_scaled / mss_scaled.sum()
    p_non = (1.0 - mss_scaled) / (1.0 - mss_scaled).sum()
    
    months = macro_df["issue_month"].values
    return months, p_def, p_non


def load_gmc_mapped(project_root: Path) -> pd.DataFrame:
    """Load, fit borrower PD, map issue_month, and merge with macro states for Give Me Some Credit."""
    logger.info("Mapping and merging Give Me Some Credit dataset...")
    
    gmc_path = project_root / "data" / "credit_risk" / "give_me_some_credit.csv"
    macro_path = project_root / "outputs" / "credit_risk" / "phase2_layer3_macro_states.csv"
    
    gmc = pd.read_csv(gmc_path)
    macro = pd.read_csv(macro_path)
    macro["issue_month"] = pd.to_datetime(macro["issue_month"]).dt.strftime("%Y-%m-%d")
    
    # Clean GMC NaNs
    gmc["MonthlyIncome"] = gmc["MonthlyIncome"].fillna(gmc["MonthlyIncome"].median())
    gmc["NumberOfDependents"] = gmc["NumberOfDependents"].fillna(0)
    
    # Train borrower PD LightGBM
    features = [c for c in gmc.columns if c != "SeriousDlqin2yrs"]
    X = gmc[features]
    y = gmc["SeriousDlqin2yrs"]
    
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=SEED, verbosity=-1)
    model.fit(X, y)
    
    gmc["borrower_pd"] = model.predict_proba(X)[:, 1]
    gmc["target"] = y
    
    # Assign issue_month based on macro stress score sampling weights
    months, p_def, p_non = get_macro_timeline_weights(macro)
    rng = np.random.RandomState(SEED)
    
    gmc_months = []
    for t in gmc["target"].values:
        if t == 1:
            m = rng.choice(months, p=p_def)
        else:
            m = rng.choice(months, p=p_non)
        gmc_months.append(m)
        
    gmc["issue_month"] = gmc_months
    gmc_merged = gmc.merge(macro, on="issue_month", how="left")
    gmc_merged["year"] = pd.to_datetime(gmc_merged["issue_month"]).dt.year
    
    logger.info(f"GMC mapped shape: {gmc_merged.shape}")
    return gmc_merged


def load_tb_mapped(project_root: Path) -> pd.DataFrame:
    """Load, fit borrower PD, map issue_month, and merge with macro states for Taiwan Bankruptcy."""
    logger.info("Mapping and merging Taiwan Bankruptcy dataset...")
    
    tb_path = project_root / "data" / "credit_risk" / "taiwan_bankruptcy.csv"
    macro_path = project_root / "outputs" / "credit_risk" / "phase2_layer3_macro_states.csv"
    
    tb = pd.read_csv(tb_path)
    macro = pd.read_csv(macro_path)
    macro["issue_month"] = pd.to_datetime(macro["issue_month"]).dt.strftime("%Y-%m-%d")
    
    # Clean columns (strip whitespace)
    tb.columns = [c.strip() for c in tb.columns]
    
    # Train borrower PD LightGBM
    features = [c for c in tb.columns if c != "Bankrupt?"]
    X = tb[features]
    y = tb["Bankrupt?"]
    
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=SEED, verbosity=-1)
    model.fit(X, y)
    
    tb["borrower_pd"] = model.predict_proba(X)[:, 1]
    tb["target"] = y
    
    # Assign issue_month based on macro stress score sampling weights
    months, p_def, p_non = get_macro_timeline_weights(macro)
    rng = np.random.RandomState(SEED)
    
    tb_months = []
    for t in tb["target"].values:
        if t == 1:
            m = rng.choice(months, p=p_def)
        else:
            m = rng.choice(months, p=p_non)
        tb_months.append(m)
        
    tb["issue_month"] = tb_months
    tb_merged = tb.merge(macro, on="issue_month", how="left")
    tb_merged["year"] = pd.to_datetime(tb_merged["issue_month"]).dt.year
    
    logger.info(f"TB mapped shape: {tb_merged.shape}")
    return tb_merged
