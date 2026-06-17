# CRIS Research Program Timeline & Findings Map

| Research Phase | Objective | Methodology | Key Findings | Audit Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | Codebase Integrity | Static code audits and pipeline checks | No pipeline breaks, basic code operates correctly | **PASS** (Scaffolding exists) |
| **Phase 0.5** | Repository Audit | Dead code identification and standardization | Identified unused scripts and redundant evaluation logic | **PASS** (Repository cleaned) |
| **Phase 1** | Model Challenge | Train LightGBM, XGBoost, and LogReg benchmarks | LightGBM selected as champion model (ROC-AUC = 0.70687) | **PASS** (Valid champion chosen) |
| **Phase 1.5** | Economic Validation | Link default prediction to portfolio metrics | Baseline 60% capacity model is highly profitable | **PASS** (Valid economic framework) |
| **Phase 2A** | default Concentration | Compare XGBoost and LightGBM default capture | Both models produce clean risk ladders; defaults concentrated in D9-D10 | **PASS** (Risk ladders validated) |
| **Phase 2B** | borrower Profiling | SHAP and feature profiling of borrower cohorts | Identifies typical low-risk vs high-risk profiles | **PASS** (Interpretability validated) |
| **Phase 2C** | borrower-Only Audit | Test model power when LendingClub indicators are dropped | Borrower-only characteristics retain 98% of predictive power | **PASS** (Intrinsic risk verified) |
| **Phase 3** | CRIS Impact Study | Direct integration of all CRIS signals into LightGBM | Out-of-sample performance degrades (ROC-AUC drops by -0.00627) | **FAIL** (Direct integration degrades model) |
| **Phase 3.1** | Signal Reduction | Test subsets of high-value CRIS signals | Performance degrades monotonically; no optimal subset exists | **FAIL** (Signal overload/noise verified) |
| **Phase 4** | Governance Layer | Use macro stress score to dynamically adjust limits | Reduces default losses in stress but reduces volume and NPV | **MIXED** (Risk reduction at cost of yield) |
