# CRIS Governance Policy Framework

This report outlines the structural framework and rules of the Cascade Risk Intelligence System (CRIS) Governance Layer.
Rather than injecting environmental signals into the borrower-level credit model (which degrades out-of-sample prediction quality), the Governance Layer dynamically adjusts portfolio parameters based on monthly macro stress regimes.

## Governance Policy Parameters

| Stress Regime | Target Capacity | Risk Appetite (Max PD Threshold) | Operational Goal |
| :--- | :---: | :---: | :--- |
| **Low Stress** (Score < 33rd Pctl) | 60% to 70% | 35% to 40% | Capture volume, loosen standards slightly |
| **Medium Stress** (33rd to 66th Pctl) | 35% to 50% | 18% to 25% | Proactive tightening, moderate risk containment |
| **High Stress** (Score >= 66th Pctl) | 15% to 30% | 8% to 15% | Capital preservation, freeze high-risk cohorts |

## Operational Policies Evaluated

### 1. System A: Credit Risk Only (Baseline)
- **Low Stress**: Capacity = 60%, Max PD = 1.0 (No limit)
- **Medium Stress**: Capacity = 60%, Max PD = 1.0 (No limit)
- **High Stress**: Capacity = 60%, Max PD = 1.0 (No limit)
- *Rationale*: Standard static lending strategy that maintains volume irrespective of macroeconomic environment.

### 2. Scenario 1: Aggressive Governance
- **Low Stress**: Capacity = 70%, Max PD = 0.35
- **Medium Stress**: Capacity = 45%, Max PD = 0.20
- **High Stress**: Capacity = 20%, Max PD = 0.10
- *Rationale*: Maximizes volume in benign periods, aggressive credit freeze during stress.

### 3. Scenario 2: Moderate Governance
- **Low Stress**: Capacity = 60%, Max PD = 0.40
- **Medium Stress**: Capacity = 50%, Max PD = 0.25
- **High Stress**: Capacity = 30%, Max PD = 0.15
- *Rationale*: Balanced approach designed to control risk without shutting down credit supply entirely.

### 4. Scenario 3: Conservative Governance
- **Low Stress**: Capacity = 50%, Max PD = 0.30
- **Medium Stress**: Capacity = 35%, Max PD = 0.18
- **High Stress**: Capacity = 15%, Max PD = 0.08
- *Rationale*: Strict capital preservation, highly sensitive to environmental risk signals.