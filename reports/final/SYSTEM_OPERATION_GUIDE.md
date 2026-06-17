# CRIS System Operation Guide

This document is the official operational runbook for the **Cascade Risk Intelligence System (CRIS)**. It provides step-by-step instructions for institutional deployment, full-system execution, and execution of the Governance Lab validation suites.

---

## A. SYSTEM SETUP

The CRIS architecture is designed to be cleanly reproducible on an institutional Linux/CUDA environment. There are no hidden dependencies or assumed system states.

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd CRIS
   ```

2. **Environment Initialization**
   CRIS is governed by a strict `environment.yml` contract. Do not install packages ad-hoc.
   ```bash
   make setup
   # OR manually:
   # conda env create -f environment.yml
   ```

3. **Activation**
   ```bash
   conda activate CRIS
   ```

---

## B. FULL SYSTEM EXECUTION

To run the complete CRIS ecosystem (including Macro Intelligence Harvesting, Credit Governance Conditioning, and Walk-Forward Validation) from end-to-end:

```bash
make run_full_cris
```

This master orchestrator (`orchestration/run_full_cris.py`):
1. Verifies infrastructure integrity.
2. Checks for Phase 2 Macro Signals. If missing, it dynamically runs the Phase 2 Harvester pipeline.
3. Executes the Phase 4 Institutional Governance routing and portfolio throttling logic.
4. Runs the final validation benchmark against a standalone borrower-centric baseline.
5. Saves all logs and artifacts to the `outputs/` directory.

---

## C. GOVERNANCE LAB USAGE

The **Governance Lab** is the research and experimentation layer of CRIS.

To run all Governance Lab experiments (01–09) sequentially and reproduce the research findings:
```bash
make run_experiments
```
This suite automatically tests:
- Governance Sensitivity Sweeps
- Recovery Velocity Calibrations
- Source-Aware Conditioning
- Temporal Cohesion

Reports and plots are saved to `validation/governance_lab/reports/` and `artifacts/`.

---

## D. REPLAY & EXPLAINABILITY (GEL + GRI)

CRIS includes the **Governance Explainability Layer (GEL)** and **Governance Replay Infrastructure (GRI)**. These are tested and demonstrated in Experiment 06 & 07.

To independently execute the replay infrastructure and audit causal traces:
```bash
conda run -n CRIS python validation/governance_lab/experiments/07_governance_replay.py
```
This generates a step-by-step deterministic replay report showing exactly why the systemic governance posture shifted on any given date.

---

## E. OPERATIONAL SIMULATION (OSHIL & GESC)

To test the system against real-world institutional friction (latency, manual overrides, trust decay):
```bash
conda run -n CRIS python validation/governance_lab/experiments/08_operational_realism.py
```

To test the elastic smoothing of the continuous governance transition response (eliminating policy whiplash):
```bash
conda run -n CRIS python validation/governance_lab/experiments/09_governance_elasticity.py
```

---

## F. STRESS CERTIFICATION (IVSC)

To execute the final **Institutional Validation & Stress Certification (IVSC)** audit. This adversarial generation tests the fragility surface and ensures the operational parameters are stable:
```bash
make run_certifications
```
*(This command runs Phase 2 Robustness Audit, Institutional Economic Impact Audit, and Experiment 10 IVSC).*

---

## G. TROUBLESHOOTING

- **Import Errors / Wiring Failures**: Run `make audit`. This will boot up a strict import analyzer mapping over every Python module to trace hidden cyclic or broken dependencies.
- **Memory Errors (OOM)**: The master pipeline uses efficient column selection rather than deep-copying 2-million row dataframes. If OOM errors persist, ensure your machine has >16GB RAM for the `run_full_cris` master orchestration.
- **Missing Paths**: The repository utilizes dynamic project root discovery (`PROJECT_ROOT = Path(__file__).resolve()`). If paths fail, ensure you are running from inside the `CRIS` directory and that `environment.yml` exists at the root.
- **Missing Market Data**: The orchestrator auto-fetches `SPY` cache data via `yfinance`. Ensure you have internet access on the first execution.

---

## H. REPRODUCIBILITY PRINCIPLES

- **Deterministic Execution**: All models and simulators are seeded via `configs.credit_config.SEED`.
- **Temporal Integrity**: The orchestration layer enforces strict point-in-time synchronization. Market states from Layer 3 are backward-joined (`merge_asof(direction='backward')`) against loan issuance dates to prevent lookahead bias.
- **Leakage Prevention**: At no point are raw market signals joined to raw borrower features for monolithic training. They are kept separated by design and interact only in the conditioning/governance layer.
