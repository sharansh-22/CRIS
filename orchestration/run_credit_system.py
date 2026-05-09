"""
Phase 4: Institutional Probabilistic Risk-Governance Infrastructure
Finalizing CRIS Layer 4: Systemic Governance Resilience.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.credit_config import OUTPUT_DIR, MODEL_DIR, SEED

logger = logging.getLogger('CRIS.credit')

# ---------------------------------------------------------------------------
# Institutional Constraints & Capacity
# ---------------------------------------------------------------------------
MONTHLY_REVIEW_CAPACITY = 8000  # Max human reviews per month
PORTFOLIO_CAP_DEFENSIVE = 0.40   # Max approval rate in DEFENSIVE state
RESERVE_FLAG_THRESHOLD = 0.35   # Trigger reserve flag if macro stress > this

# ---------------------------------------------------------------------------
# Phase 4 Logic Extensions
# ---------------------------------------------------------------------------

def calculate_review_priority(row):
    """Prioritize loans in the review queue based on PD and Uncertainty."""
    pd = row['pd_macro']
    conf = row['environmental_confidence']
    
    # Priority = High PD (near threshold) * Low Confidence
    priority = pd * (1 - conf)
    return priority

def apply_portfolio_throttling(monthly_df):
    """Apply aggregate governance constraints to a month's worth of applications."""
    state = monthly_df['gov_state'].iloc[0]
    
    # Sort by PD to prioritize best loans if throttling is needed
    monthly_df = monthly_df.sort_values('pd_macro')
    
    if state == "DEFENSIVE":
        n_to_approve = int(len(monthly_df) * PORTFOLIO_CAP_DEFENSIVE)
        # Mark those beyond the cap as 'THROTTLED'
        monthly_df['throttled'] = False
        if len(monthly_df) > n_to_approve:
            monthly_df.iloc[n_to_approve:, monthly_df.columns.get_loc('throttled')] = True
    else:
        monthly_df['throttled'] = False
        
    return monthly_df

# ---------------------------------------------------------------------------
# Execution Pipeline
# ---------------------------------------------------------------------------

def run_phase4_infrastructure():
    logger.info("    Phase 4: Institutional Governance Infrastructure")
    
    # 1. Reuse Phase 3 logic to get loan-level governance states
    # We'll use the full_df for systemic simulation (2007-2018)
    data_path = OUTPUT_DIR / "engineered_data.parquet"
    macro_path = OUTPUT_DIR / "phase2_layer3_macro_states.csv"
    if not data_path.exists() or not macro_path.exists():
        logger.error("Missing Phase 2/3 artifacts.")
        return
        
    full_df = pd.read_parquet(data_path)
    full_df['issue_d'] = pd.to_datetime(full_df['issue_d'])
    macro_df = pd.read_csv(macro_path)
    
    # Load Model and Config
    model = joblib.load(MODEL_DIR / "lightgbm.joblib")
    with open(OUTPUT_DIR / "phase2_macro_conditioning_results.json", 'r') as f:
        p2_config = json.load(f)['overlay_config']
        
    # Pre-process Macro States
    # (Mapping logic from Phase 3)
    from orchestration.legacy_credit_orch_p3 import map_governance_state, apply_governance_routing
    macro_df['gov_state'] = macro_df.apply(map_governance_state, axis=1)
    
    # Join and Process
    full_df['issue_month'] = full_df['issue_d'].dt.strftime('%Y-%m-01')
    full_df = full_df.merge(macro_df, on='issue_month', how='left')
    
    # Drop NAs (if any months missing macro data)
    full_df = full_df.dropna(subset=['macro_stress_score'])
    
    # Calculate PDs
    logger.info("    Calculating systemic PDs...")
    model_features = model.feature_name_
    original_cols = {c.replace(' ', '_'): c for c in full_df.columns}
    needed_cols = [original_cols.get(f, f) for f in model_features]
    X_predict = full_df[needed_cols].copy()
    X_predict.columns = model_features
    full_df['pd_borrower'] = model.predict_proba(X_predict)[:, 1]
    del X_predict
    
    def apply_conditioning(row):
        score = row['macro_stress_score']
        pd_b = np.clip(row['pd_borrower'], 1e-6, 1 - 1e-6)
        logit_b = np.log(pd_b / (1 - pd_b))
        shift = p2_config['beta'] * max(0, score - p2_config['stress_anchor'])
        shift = min(shift, p2_config['max_logit_shift'])
        logit_m = logit_b + shift
        return 1 / (1 + np.exp(-logit_m))
        
    full_df['pd_macro'] = full_df.apply(apply_conditioning, axis=1)
    full_df['gov_state'] = full_df.apply(map_governance_state, axis=1)
    full_df['routing_decision'] = full_df.apply(apply_governance_routing, axis=1)
    
    # 2. Portfolio-Level Governance: Exposure Throttling
    logger.info("    Applying portfolio-level exposure throttling...")
    processed_months = []
    for month, group in full_df.groupby('issue_month'):
        throttled_group = apply_portfolio_throttling(group)
        processed_months.append(throttled_group)
    
    full_df = pd.concat(processed_months)
    
    # 3. Review Capacity Dynamics
    logger.info("    Analyzing review capacity dynamics...")
    capacity_stats = full_df.groupby('issue_month').apply(lambda x: pd.Series({
        'reviews_requested': (x['routing_decision'] == 'MANUAL_REVIEW').sum(),
        'capacity_utilization': (x['routing_decision'] == 'MANUAL_REVIEW').sum() / MONTHLY_REVIEW_CAPACITY,
        'reserve_flag': x['macro_stress_score'].iloc[0] > RESERVE_FLAG_THRESHOLD
    }), include_groups=False)
    
    # 4. Cross-Sectional Stress Awareness
    logger.info("    Analyzing cross-sectional stress clusters...")
    # Analyze by Grade
    grade_cols = [c for c in full_df.columns if 'grade_' in c and '_' in c[6:]] # Get one-hot grade cols
    # Simpler: If we had the original 'grade', but we have one-hot. 
    # Let's just use PD bands as a proxy for risk cohorts.
    full_df['risk_cohort'] = pd.qcut(full_df['pd_borrower'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    cohort_stress = full_df.groupby(['issue_month', 'risk_cohort'], observed=True)['pd_macro'].mean().unstack()
    
    # 5. Systemic Stress Simulation Summary
    # Focus on 2008 and 2018
    stress_2008 = capacity_stats[capacity_stats.index.str.startswith('2008')]
    stress_2018 = capacity_stats[capacity_stats.index.str.startswith('2018')]
    
    logger.info(f"    2008 Peak Capacity Utilization: {stress_2008['capacity_utilization'].max():.2%}")
    logger.info(f"    2018 Peak Capacity Utilization: {stress_2018['capacity_utilization'].max():.2%}")
    
    # 6. Save Artifacts
    full_df.to_parquet(OUTPUT_DIR / "phase4_systemic_governance_results.parquet")
    capacity_stats.to_csv(OUTPUT_DIR / "phase4_review_capacity_dynamics.csv")
    cohort_stress.to_csv(OUTPUT_DIR / "phase4_cross_sectional_stress.csv")
    
    # Final Summary for Report
    summary_stats = {
        "total_loans_processed": len(full_df),
        "total_throttled_loans": int(full_df['throttled'].sum()),
        "max_monthly_reviews": int(capacity_stats['reviews_requested'].max()),
        "months_with_reserve_flags": int(capacity_stats['reserve_flag'].sum()),
        "avg_utilization_normal": float(capacity_stats[capacity_stats['reserve_flag'] == False]['capacity_utilization'].mean()),
        "avg_utilization_stress": float(capacity_stats[capacity_stats['reserve_flag'] == True]['capacity_utilization'].mean())
    }
    with open(OUTPUT_DIR / "phase4_summary_stats.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
        
    logger.info("    Phase 4 complete.")

if __name__ == "__main__":
    run_phase4_infrastructure()
