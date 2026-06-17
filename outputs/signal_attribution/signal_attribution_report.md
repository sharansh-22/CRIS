# CRIS Signal Attribution Engine — Research Report V1

---

## 1. Final Attribution Distribution

| Rank | Signal | Source | Weight | Correlation | AUC Lift | Temporal Stab. | Regime Stab. |
|------|--------|--------|--------|-------------|----------|----------------|--------------|
| 1 | uncertainty_pressure | Layer3.Meta | 0.2421 | 0.5278 | +0.000857 | 0.47 | 0.96 |
| 2 | stabilization_strength | Layer3.Meta | 0.1460 | 0.5294 | +0.000035 | 0.15 | 0.44 |
| 3 | trajectory_fragility | Layer3.Decay | 0.1398 | 0.3957 | +0.000029 | 0.40 | 0.36 |
| 4 | erosion_strength | Layer3.Decay | 0.1252 | 0.3862 | +0.000039 | 0.45 | 0.17 |
| 5 | signal_coherence | Layer3.Meta | 0.0997 | 0.3099 | +0.000082 | 0.22 | 0.27 |
| 6 | structural_fragility | Layer3.Slow | 0.0968 | 0.1781 | +0.000002 | 0.64 | 0.05 |
| 7 | shock_intensity | Layer3.Fast | 0.0630 | 0.2195 | -0.000003 | 0.22 | 0.05 |
| 8 | structural_instability | Layer3.Slow | 0.0563 | 0.2111 | +0.000002 | 0.16 | 0.07 |
| 9 | liquidity_disruption | Layer3.Fast | 0.0311 | 0.0665 | +0.000000 | 0.16 | 0.05 |

**Σ weights = 1.0000**

## 2. Attribution Entropy Analysis

- **Shannon Entropy**: 2.9713 bits
- **Maximum Entropy**: 3.1699 bits (uniform over 9 signals)
- **Normalized Entropy**: 0.9373
- **Top-3 Concentration**: 52.8%
- **Top-5 Concentration**: 75.3%
- **Interpretation**: HIGHLY DISTRIBUTED: Information is spread broadly across many signals. No single signal dominates. The environmental state is multi-dimensional.

## 3. Attribution Through Time

### 2007–2010
- Loans: 18,065 | Defaults: 2,373 | Default Rate: 13.14%
  - liquidity_disruption: 0.2669
  - uncertainty_pressure: 0.1851
  - shock_intensity: 0.1732
  - structural_instability: 0.1703
  - structural_fragility: 0.0743

### 2010–2013
- Loans: 221,428 | Defaults: 34,452 | Default Rate: 15.56%
  - stabilization_strength: 0.3657
  - uncertainty_pressure: 0.1876
  - liquidity_disruption: 0.1071
  - erosion_strength: 0.0906
  - trajectory_fragility: 0.0782

### 2013–2016
- Loans: 1,026,558 | Defaults: 206,242 | Default Rate: 20.09%
  - uncertainty_pressure: 0.2167
  - trajectory_fragility: 0.1806
  - structural_fragility: 0.1544
  - erosion_strength: 0.1343
  - signal_coherence: 0.1246

### 2016–2018
- Loans: 518,744 | Defaults: 116,295 | Default Rate: 22.42%
  - stabilization_strength: 0.1775
  - erosion_strength: 0.1698
  - trajectory_fragility: 0.1697
  - signal_coherence: 0.1574
  - shock_intensity: 0.1191

## 4. Walk-Forward Validation

- ✓ **state_date_ordering**: PASS (0 violations)
- ✓ **shock_intensity_completeness**: PASS (0.0% missing)
- ✓ **liquidity_disruption_completeness**: PASS (0.0% missing)
- ✓ **structural_instability_completeness**: PASS (0.0% missing)
- ✓ **structural_fragility_completeness**: PASS (0.0% missing)
- ✓ **erosion_strength_completeness**: PASS (0.0% missing)
- ✓ **trajectory_fragility_completeness**: PASS (0.0% missing)
- ✓ **stabilization_strength_completeness**: PASS (0.0% missing)
- ✓ **uncertainty_pressure_completeness**: PASS (0.0% missing)
- ✓ **signal_coherence_completeness**: PASS (0.0% missing)
- ✓ **temporal_monotonicity**: PASS
- ✓ **unique_months**: PASS (139 distinct months)
- ✓ **shock_intensity_leakage_check**: PASS (|corr|=0.002)
- ✓ **liquidity_disruption_leakage_check**: PASS (|corr|=0.000)
- ✓ **structural_instability_leakage_check**: PASS (|corr|=0.005)
- ✓ **structural_fragility_leakage_check**: PASS (|corr|=0.007)
- ✓ **erosion_strength_leakage_check**: PASS (|corr|=0.015)
- ✓ **trajectory_fragility_leakage_check**: PASS (|corr|=0.014)
- ✓ **stabilization_strength_leakage_check**: PASS (|corr|=0.024)
- ✓ **uncertainty_pressure_leakage_check**: PASS (|corr|=0.037)
- ✓ **signal_coherence_leakage_check**: PASS (|corr|=0.012)
- ✓ **OVERALL**: PASS — All checks green

**Overall Status**: `PASS — All checks green`

## 5. Dataset Summary

- Total Loans: 1,345,350
- Total Defaults: 268,599
- Overall Default Rate: 19.96%
