"""
Experiment 01: Governance Sensitivity Sweep
Objective: Identify the optimal trade-off between loss reduction (resilience) 
and opportunity cost (efficiency) by varying the conditioning beta.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from validation.governance_lab.shared.engine import GovernanceLabEngine
from configs.credit_config import OUTPUT_DIR as CREDIT_OUTPUT

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CRIS.governance_lab.exp01')

# Paths
DATA_PATH = CREDIT_OUTPUT / "engineered_data.parquet"
MACRO_PATH = CREDIT_OUTPUT / "phase2_layer3_macro_states.csv"
CONFIG_PATH = CREDIT_OUTPUT / "phase2_macro_conditioning_results.json"
LAB_DIR = Path(__file__).resolve().parent.parent

def run_sensitivity_sweep():
    logger.info("Initializing Governance Lab Sensitivity Sweep...")
    engine = GovernanceLabEngine(DATA_PATH, MACRO_PATH, CONFIG_PATH)
    
    # Sensitivity Parameter (Beta)
    # 0.0 = Baseline (No macro adjustment)
    # 0.4 = Calibrated (Standard CRIS)
    # 1.0+ = Hyper-defensive
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    results = []
    
    # Fix anchor at 0.25 (Standard anchor)
    anchor = 0.25
    
    for beta in betas:
        logger.info(f"Running simulation for Beta: {beta}...")
        df = engine.run_policy_simulation(beta=beta, stress_anchor=anchor)
        metrics = engine.calculate_experiment_metrics(df)
        metrics['beta'] = beta
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    
    # Save Results
    results_df.to_csv(LAB_DIR / "metrics" / "01_sensitivity_results.csv", index=False)
    logger.info("Metrics saved to validation/governance_lab/metrics/01_sensitivity_results.csv")
    
    # Generate Plots
    generate_plots(results_df)
    generate_report(results_df)

def generate_plots(df):
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Pareto Frontier (Default Rate vs Opportunity Loss)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['opportunity_loss_count'], df['default_rate'], c=df['beta'], cmap='viridis', s=100)
    plt.colorbar(label='Governance Sensitivity (Beta)')
    plt.xlabel('Opportunity Loss (Good Loans Rejected)')
    plt.ylabel('Default Rate (Portfolio Quality)')
    plt.title('Governance Efficiency Frontier: Resilience vs Efficiency')
    
    for i, txt in enumerate(df['beta']):
        plt.annotate(f"β={txt}", (df['opportunity_loss_count'].iloc[i], df['default_rate'].iloc[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.savefig(LAB_DIR / "plots" / "01_efficiency_frontier.png")
    
    # Plot 2: Institutional Utility
    plt.figure(figsize=(10, 6))
    plt.plot(df['beta'], df['net_utility'], marker='o', linewidth=2)
    plt.xlabel('Governance Sensitivity (Beta)')
    plt.ylabel('Net Institutional Utility (Estimated P&L)')
    plt.title('Institutional Utility Curve: Finding the Optimal Defensive Bias')
    plt.axvline(0.4, color='red', linestyle='--', label='Current Calibrated Beta (0.4)')
    plt.legend()
    plt.savefig(LAB_DIR / "plots" / "01_utility_curve.png")
    
    # Plot 3: Approval Rate vs Capital Efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(df['beta'], df['approval_rate'], marker='s', label='Approval Rate')
    plt.plot(df['beta'], df['capital_efficiency'] * 10, marker='^', label='Capital Efficiency (x10)')
    plt.xlabel('Governance Sensitivity (Beta)')
    plt.ylabel('Rate / Efficiency')
    plt.title('Capital Efficiency vs Credit Throughput')
    plt.legend()
    plt.savefig(LAB_DIR / "plots" / "01_efficiency_throughput.png")

def generate_report(df):
    # Identify Optimal
    optimal_utility = df.loc[df['net_utility'].idxmax()]
    baseline = df.loc[df['beta'] == 0.0].iloc[0]
    calibrated = df.loc[df['beta'] == 0.4].iloc[0]
    
    report = f"""# Governance Experiment 01: Sensitivity Sweep Report
    
## 1. Executive Summary
This experiment varied the governance sensitivity coefficient (Beta) to evaluate the impact of macro-conditioning on institutional performance. We measured the trade-off between credit losses (False Negatives) and opportunity costs (False Positives/Rejections of Good Loans).

## 2. Quantitative Results
{df.to_markdown(index=False)}

## 3. Key Observations
* **Utility Peak:** The institutional utility (P&L approximation) peaks at Beta = {optimal_utility['beta']}.
* **Over-Defensiveness Check:** At Beta = 0.4 (Current Calibrated), the approval rate is {calibrated['approval_rate']:.2%}. 
* **Baseline Comparison:** The baseline (Beta=0.0) has a default rate of {baseline['default_rate']:.2%}, while the optimal policy has a default rate of {optimal_utility['default_rate']:.2%}.
* **Opportunity Cost:** Increasing Beta from 0.0 to 1.5 increases opportunity loss from {baseline['opportunity_loss_count']} to {df.iloc[-1]['opportunity_loss_count']} rejections of valid loans.

## 4. Conclusion
The "Over-Defensiveness" bottleneck is quantified here. If the Utility curve slopes downward after Beta=0.4, CRIS is already at or past the point of diminishing returns for resilience.

## 5. Decision Support
Based on these findings, the recommended governance sensitivity for the current regime is Beta={optimal_utility['beta']}.
"""
    with open(LAB_DIR / "reports" / "01_sensitivity_report.md", 'w') as f:
        f.write(report)
    logger.info("Report saved to validation/governance_lab/reports/01_sensitivity_report.md")

if __name__ == "__main__":
    run_sensitivity_sweep()
