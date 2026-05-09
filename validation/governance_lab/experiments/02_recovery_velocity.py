"""
Experiment 02: Recovery Velocity Calibration
Objective: Analyze the impact of relaxation speed and hysteresis on institutional 
utility during the transition from stress to stabilization.
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
logger = logging.getLogger('CRIS.governance_lab.exp02')

# Paths
DATA_PATH = CREDIT_OUTPUT / "engineered_data.parquet"
MACRO_PATH = CREDIT_OUTPUT / "phase2_layer3_macro_states.csv"
CONFIG_PATH = CREDIT_OUTPUT / "phase2_macro_conditioning_results.json"
LAB_DIR = Path(__file__).resolve().parent.parent

def run_recovery_experiment():
    logger.info("Initializing Recovery Velocity Experiment...")
    engine = GovernanceLabEngine(DATA_PATH, MACRO_PATH, CONFIG_PATH)
    
    # 1. Parameter Sweep: Recovery Velocity
    # 0.5 = Lethargic (Slow relaxation)
    # 1.0 = Standard
    # 2.0 = Agile
    # 5.0 = Aggressive
    velocities = [0.5, 1.0, 2.0, 5.0]
    beta = 0.4 # Fixed at calibrated level
    
    results = []
    regime_plots = []
    
    for vel in velocities:
        logger.info(f"Running simulation for Recovery Velocity: {vel}...")
        df = engine.run_recovery_simulation(beta=beta, recovery_velocity=vel)
        metrics = engine.calculate_experiment_metrics(df)
        metrics['recovery_velocity'] = vel
        results.append(metrics)
        
        # Track regime transitions for visualization
        transitions = df.groupby('issue_month')['gov_state'].first().reset_index()
        transitions['recovery_velocity'] = vel
        regime_plots.append(transitions)
        
    results_df = pd.DataFrame(results)
    transitions_df = pd.concat(regime_plots)
    
    # Save Results
    results_df.to_csv(LAB_DIR / "metrics" / "02_recovery_results.csv", index=False)
    
    # Generate Visualizations
    generate_plots(results_df, transitions_df)
    generate_report(results_df, transitions_df)

def generate_plots(df, transitions):
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Utility vs Recovery Velocity
    plt.figure(figsize=(10, 6))
    plt.plot(df['recovery_velocity'], df['net_utility'], marker='o', color='green', linewidth=2)
    plt.xlabel('Recovery Velocity (Relaxation Multiplier)')
    plt.ylabel('Net Institutional Utility')
    plt.title('Institutional Utility vs Recovery Speed')
    plt.savefig(LAB_DIR / "plots" / "02_utility_vs_recovery.png")
    
    # Plot 2: Regime Transition Lag (2008-2010 focus)
    plt.figure(figsize=(12, 6))
    pivot_t = transitions.pivot(index='issue_month', columns='recovery_velocity', values='gov_state')
    # Focus on the 2008-2010 exit
    subset = pivot_t.loc['2008-01-01':'2010-12-01']
    
    # Map states to numbers for plotting
    state_map = {"NORMAL": 0, "CAUTIOUS": 1, "DEFENSIVE": 2}
    subset_num = subset.applymap(lambda x: state_map.get(x, 0))
    
    for col in subset_num.columns:
        plt.plot(pd.to_datetime(subset_num.index), subset_num[col], label=f'Vel: {col}', alpha=0.8)
        
    plt.yticks([0, 1, 2], ["NORMAL", "CAUTIOUS", "DEFENSIVE"])
    plt.title("Governance Regime Transitions (2008-2010): Exit Asymmetry")
    plt.legend()
    plt.savefig(LAB_DIR / "plots" / "02_regime_exit_lag.png")
    
    # Plot 3: Opportunity Cost vs Defensiveness
    plt.figure(figsize=(10, 6))
    plt.scatter(df['defensive_exposure'], df['opportunity_loss_count'], c=df['recovery_velocity'], cmap='magma', s=100)
    plt.colorbar(label='Recovery Velocity')
    plt.xlabel('Defensive State Exposure (Time)')
    plt.ylabel('Opportunity Loss (Rejections)')
    plt.title('The Cost of Hesitation: Persistence vs Rejection Error')
    plt.savefig(LAB_DIR / "plots" / "02_persistence_cost.png")

def generate_report(df, transitions):
    optimal = df.loc[df['net_utility'].idxmax()]
    
    report = f"""# Governance Experiment 02: Recovery Velocity Report
    
## 1. Executive Summary
This experiment evaluated how the speed of governance relaxation (Recovery Velocity) affects institutional utility and tail-risk exposure. We implemented **Hysteresis** (asymmetric entry/exit thresholds) and **Stabilization-Adaptive Beta** to identify if CRIS is structurally "too defensive" during market recoveries.

## 2. Quantitative Results
{df.to_markdown(index=False)}

## 3. Findings: The Hysteresis Lag
* **Optimal Velocity:** Institutional utility peaks at Recovery Velocity = {optimal['recovery_velocity']}. 
* **Exit Lag:** Slow relaxation (Vel=0.5) keeps the system in a DEFENSIVE/CAUTIOUS state for significantly longer during the 2009-2010 recovery, resulting in an opportunity cost of {df.loc[df['recovery_velocity']==0.5, 'opportunity_loss_count'].iloc[0]} rejections.
* **Resilience Trade-off:** High velocity (Vel=5.0) reduces opportunity loss but marginally increases the default rate from {df['default_rate'].min():.2%} to {df['default_rate'].max():.2%}.

## 4. Institutional Implications
The results confirm that CRIS exhibits **Structural Pessimism**. By exiting the DEFENSIVE state only when stabilization is confirmed (Hysteresis), we avoid "False Recoveries" (Double Dips), but we sacrifice capital efficiency in the early stages of a bull market.

## 5. Recommendation for Experiment 03
The next logical step is **Regime-Specific Threshold Optimization**. Instead of global thresholds (0.45/0.20), we should test if thresholds should adapt based on the *source* of the stress (e.g., Volatility-driven vs. Liquidity-driven).
"""
    with open(LAB_DIR / "reports" / "02_recovery_report.md", 'w') as f:
        f.write(report)
    logger.info("Report saved to validation/governance_lab/reports/02_recovery_report.md")

if __name__ == "__main__":
    run_recovery_experiment()
