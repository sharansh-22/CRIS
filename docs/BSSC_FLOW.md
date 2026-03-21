# BSSC — Flow Structure
## Layer 3: Black Swan and Slippage Computer

## Pipeline
```
INPUT: ticker, csv_path, config
         ↓
[simulation.py]
  Calibrate from historical data
  Generate GBM paths (normal market)
  Generate Jump-Diffusion paths (crisis)
  Output: SimulationResult
         ↓
[entropy.py]
  Compute Sample Entropy (primary)
  Compute Permutation Entropy (confirmation)
  Classify market state: NORMAL/STRESS/BLACK_SWAN
  Output: EntropyResult
         ↓
[slippage.py]
  Almgren-Chriss market impact
  Entropy-conditioned regime multiplier
  Monte Carlo IS distribution (2000 paths)
  Output: SlippageResult
         ↓
[models.py]
  Pydantic validation of all three outputs
  Assemble Layer3Report
  Derive overall_risk_level
  Derive recommended_action: HOLD/REDUCE/LIQUIDATE
         ↓
[detector.py]
  Orchestrate full pipeline
  Run consistency checks
  Output: Layer3Report
         ↓
[report.py]
  Atomic JSON → data/simulation_output/
  Markdown → reports/
  Output: human + machine readable reports
         ↓
OUTPUT: Layer3Report → Convergence Engine
```

## Known Issue
Entropy threshold direction under investigation.
See BSSC_ADR.md Change 2 and Change 3.
