# CRIS Signal Inventory

This inventory documents every environmental risk signal constructed within the CRIS repository, verifying its data coverage, frequency, source, and leakage risks.

## Signal Metadata and Audit Registry

| Signal | Family | Time Frequency | Missing Values | Coverage (2007-2018) | First Date | Leakage Risk / Mitigation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `shock_intensity` | Layer3.Fast | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking to loan issue month. |
| `liquidity_disruption` | Layer3.Fast | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `instability_velocity` | Layer3.Fast | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `structural_instability` | Layer3.Slow | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `stress_persistence` | Layer3.Slow | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `structural_fragility` | Layer3.Slow | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `erosion_strength` | Layer3.Decay | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `rebound_failure` | Layer3.Decay | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `resilience_deficit` | Layer3.Decay | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `trajectory_fragility` | Layer3.Decay | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `stabilization_strength` | Layer3.Meta | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `uncertainty_pressure` | Layer3.Meta | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `signal_coherence` | Layer3.Meta | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `breadth_health` | MarketStructure | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `breadth_deterioration` | MarketStructure | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `market_structure_fragility` | MarketStructure | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `dispersion_pressure` | MarketStructure | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |
| `correlation_density` | MarketStructure | Monthly | 0.0% | 100% | 2007-06-01 | **None**. Aligned backward-looking. |

## Leakage Control Protocol Verification
- **Temporal Join Safe**: Standardized time-series merge keys (`issue_month`) enforce that the macro state assigned to loan $i$ is strictly the most recent state computed prior to the loan's origination date. No future lookahead or future price/default trends are visible to the model during training or evaluation.
- **No Post-Origination Signals**: The macro and market structure indicators are derived from public market price indices (e.g. S&P 500, Sector ETFs) and contain no borrower outcomes, borrower payment performance, or LendingClub portfolio metrics. This ensures zero target leakage.