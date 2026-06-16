# CRIS Signal Attribution Engine — Research Report V1

---

## 1. Final Attribution Distribution

| Rank | Signal | Source | Weight | Correlation | AUC Lift | Temporal Stab. | Regime Stab. |
|------|--------|--------|--------|-------------|----------|----------------|--------------|
| 1 | rebound_failure | Layer3.Decay | 0.1171 | 0.4038 | -0.000002 | 0.45 | 0.76 |
| 2 | uncertainty_pressure | Layer3.Meta | 0.1149 | 0.5152 | -0.000016 | 0.41 | 0.58 |
| 3 | trajectory_fragility | Layer3.Decay | 0.1127 | 0.3959 | +0.000033 | 0.42 | 0.73 |
| 4 | erosion_strength | Layer3.Decay | 0.0868 | 0.3853 | +0.000039 | 0.49 | 0.26 |
| 5 | signal_coherence | Layer3.Meta | 0.0845 | 0.3785 | +0.000052 | 0.49 | 0.23 |
| 6 | structural_instability | Layer3.Slow | 0.0666 | 0.1426 | +0.000002 | 0.78 | 0.05 |
| 7 | correlation_density | MarketStructure | 0.0569 | 0.3635 | +0.001246 | 0.05 | 0.05 |
| 8 | breadth_health | MarketStructure | 0.0474 | 0.0647 | +0.000003 | 0.44 | 0.22 |
| 9 | shock_intensity | Layer3.Fast | 0.0422 | 0.0923 | +0.000001 | 0.48 | 0.05 |
| 10 | breadth_deterioration | MarketStructure | 0.0420 | 0.0418 | +0.000027 | 0.42 | 0.18 |
| 11 | dispersion_pressure | MarketStructure | 0.0399 | 0.1252 | +0.000245 | 0.33 | 0.05 |
| 12 | resilience_deficit | Layer3.Decay | 0.0345 | 0.0698 | +0.000277 | 0.18 | 0.21 |
| 13 | stabilization_strength | Layer3.Meta | 0.0343 | 0.2707 | -0.000239 | 0.05 | 0.05 |
| 14 | market_structure_fragility | MarketStructure | 0.0318 | 0.1378 | +0.000171 | 0.20 | 0.05 |
| 15 | stress_persistence | Layer3.Slow | 0.0231 | 0.1625 | +0.000001 | 0.05 | 0.05 |
| 16 | structural_fragility | Layer3.Slow | 0.0231 | 0.1624 | +0.000000 | 0.05 | 0.05 |
| 17 | instability_velocity | Layer3.Fast | 0.0225 | 0.0155 | +0.000251 | 0.24 | 0.05 |
| 18 | liquidity_disruption | Layer3.Fast | 0.0197 | 0.0665 | +0.000000 | 0.16 | 0.05 |

**Σ weights = 1.0000**

## 2. Attribution Entropy Analysis

- **Shannon Entropy**: 3.9360 bits
- **Maximum Entropy**: 4.1699 bits (uniform over 18 signals)
- **Normalized Entropy**: 0.9439
- **Top-3 Concentration**: 34.5%
- **Top-5 Concentration**: 51.6%
- **Interpretation**: HIGHLY DISTRIBUTED: Information is spread broadly across many signals. No single signal dominates. The environmental state is multi-dimensional.

## 3. Attribution Through Time

### 2007–2010
- Loans: 18,065 | Defaults: 2,373 | Default Rate: 13.14%
  - liquidity_disruption: 0.1530
  - instability_velocity: 0.1102
  - uncertainty_pressure: 0.1044
  - stabilization_strength: 0.0987
  - shock_intensity: 0.0789

### 2010–2013
- Loans: 221,428 | Defaults: 34,452 | Default Rate: 15.56%
  - resilience_deficit: 0.0827
  - breadth_deterioration: 0.0796
  - structural_instability: 0.0778
  - instability_velocity: 0.0776
  - breadth_health: 0.0760

### 2013–2016
- Loans: 1,026,558 | Defaults: 206,242 | Default Rate: 20.09%
  - uncertainty_pressure: 0.1834
  - dispersion_pressure: 0.1792
  - correlation_density: 0.1668
  - rebound_failure: 0.0924
  - trajectory_fragility: 0.0672

### 2016–2018
- Loans: 518,744 | Defaults: 116,295 | Default Rate: 22.42%
  - erosion_strength: 0.1223
  - trajectory_fragility: 0.1213
  - signal_coherence: 0.1209
  - resilience_deficit: 0.1084
  - rebound_failure: 0.1034

## 4. Walk-Forward Validation

- ✓ **state_date_ordering**: PASS (0 violations)
- ✓ **shock_intensity_completeness**: PASS (0.0% missing)
- ✓ **liquidity_disruption_completeness**: PASS (0.0% missing)
- ✓ **instability_velocity_completeness**: PASS (0.0% missing)
- ✓ **structural_instability_completeness**: PASS (0.0% missing)
- ✓ **stress_persistence_completeness**: PASS (0.0% missing)
- ✓ **structural_fragility_completeness**: PASS (0.0% missing)
- ✓ **erosion_strength_completeness**: PASS (0.0% missing)
- ✓ **rebound_failure_completeness**: PASS (0.0% missing)
- ✓ **resilience_deficit_completeness**: PASS (0.0% missing)
- ✓ **trajectory_fragility_completeness**: PASS (0.0% missing)
- ✓ **stabilization_strength_completeness**: PASS (0.0% missing)
- ✓ **uncertainty_pressure_completeness**: PASS (0.0% missing)
- ✓ **signal_coherence_completeness**: PASS (0.0% missing)
- ✓ **breadth_health_completeness**: PASS (0.0% missing)
- ✓ **breadth_deterioration_completeness**: PASS (0.0% missing)
- ✓ **market_structure_fragility_completeness**: PASS (0.0% missing)
- ✓ **dispersion_pressure_completeness**: PASS (0.0% missing)
- ✓ **correlation_density_completeness**: PASS (0.0% missing)
- ✓ **temporal_monotonicity**: PASS
- ✓ **unique_months**: PASS (139 distinct months)
- ✓ **shock_intensity_leakage_check**: PASS (|corr|=0.003)
- ✓ **liquidity_disruption_leakage_check**: PASS (|corr|=0.000)
- ✓ **instability_velocity_leakage_check**: PASS (|corr|=0.015)
- ✓ **structural_instability_leakage_check**: PASS (|corr|=0.010)
- ✓ **stress_persistence_leakage_check**: PASS (|corr|=0.006)
- ✓ **structural_fragility_leakage_check**: PASS (|corr|=0.006)
- ✓ **erosion_strength_leakage_check**: PASS (|corr|=0.015)
- ✓ **rebound_failure_leakage_check**: PASS (|corr|=0.009)
- ✓ **resilience_deficit_leakage_check**: PASS (|corr|=0.014)
- ✓ **trajectory_fragility_leakage_check**: PASS (|corr|=0.015)
- ✓ **stabilization_strength_leakage_check**: PASS (|corr|=0.008)
- ✓ **uncertainty_pressure_leakage_check**: PASS (|corr|=0.012)
- ✓ **signal_coherence_leakage_check**: PASS (|corr|=0.016)
- ✓ **breadth_health_leakage_check**: PASS (|corr|=0.000)
- ✓ **breadth_deterioration_leakage_check**: PASS (|corr|=0.006)
- ✓ **market_structure_fragility_leakage_check**: PASS (|corr|=0.014)
- ✓ **dispersion_pressure_leakage_check**: PASS (|corr|=0.016)
- ✓ **correlation_density_leakage_check**: PASS (|corr|=0.039)
- ✓ **OVERALL**: PASS — All checks green

**Overall Status**: `PASS — All checks green`

## 5. Dataset Summary

- Total Loans: 1,345,350
- Total Defaults: 268,599
- Overall Default Rate: 19.96%
