# Documentation Audit & Consolidation Report

This document audits all markdown files in the repository during the Phase 5 release validation, classifying each into KEEP, MERGE, ARCHIVE, or DELETE according to quantitative model risk guidelines.

---

## 1. Active Repository Documentation (KEEP)

Only two active README files are maintained at the repository root and platform levels:

| File Path | Description | Classification |
| :--- | :--- | :---: |
| **`README.md`** | Main repository entry point, telling the story of CRIS research and final results. | **KEEP** |
| **`systems/credit_risk/README.md`** | Detailed user guide and model specification for the validated credit risk platform. | **KEEP** |

---

## 2. Final Audited Reports Relocated to `reports/final/` (ARCHIVE)

All final validation reports and ledgers from the model risk committee reviews have been relocated to the `reports/final/` folder to clean the repository root:

| File Path (New Location) | Purpose | Classification |
| :--- | :--- | :---: |
| **`reports/final/CLAIM_VALIDATION_MATRIX.md`** | Hypotheses validation checklist for CRIS. | **ARCHIVE** |
| **`reports/final/CRIS_ECONOMIC_AUDIT.md`** | Audited financial metrics and correction log. | **ARCHIVE** |
| **`reports/final/CRIS_EVIDENCE_LEDGER.md`** | Evidence ledger mapping support vs. contradiction. | **ARCHIVE** |
| **`reports/final/CRIS_FINAL_VERDICT_REPORT.md`** | Model risk committee verdict report. | **ARCHIVE** |
| **`reports/final/CRIS_STATISTICAL_AUDIT.md`** | Bootstrap significance test audit. | **ARCHIVE** |
| **`reports/final/FINAL_PREDICTIVE_VERDICT.md`** | Direct prediction test results. | **ARCHIVE** |
| **`reports/final/FINAL_SIGNAL_VERDICT.md`** | Signal reduction test results. | **ARCHIVE** |
| **`reports/final/FINAL_GOVERNANCE_ATTRIBUTION_REPORT.md`** | Systems A vs. B vs. C comparison. | **ARCHIVE** |
| **`reports/final/REPOSITORY_READINESS_REPORT.md`** | Portfolio release checklist. | **ARCHIVE** |
| **`reports/final/FINAL_REPOSITORY_AUDIT.md`** | Structure, link, and command verification audit. | **ARCHIVE** |
| **`reports/final/CRIS_RELEASE_READINESS.md`** | Archive of general model release rules. | **ARCHIVE** |
| **`reports/final/PROJECT_PUBLICATION_CHECKLIST.md`** | Portfolio checklist. | **ARCHIVE** |
| **`reports/final/CRIS_IMPACT_STUDY_FINAL_REPORT.md`** | Main report for Phase 3. | **ARCHIVE** |
| **`reports/final/CRIS_PHASE3_1_SIGNAL_REDUCTION_REPORT.md`** | Main report for Phase 3.1. | **ARCHIVE** |
| **`reports/final/CRIS_GOVERNANCE_IMPACT_REPORT.md`** | Main report for Phase 4. | **ARCHIVE** |
| **`reports/final/SYSTEM_OPERATION_GUIDE.md`** | Developer setup manual. | **ARCHIVE** |
| **`reports/final/BRANCHING.md`** | Git branching guide. | **ARCHIVE** |
| **`reports/final/README_INVENTORY.md`** | Pre-consolidation documentation checklist. | **ARCHIVE** |
| **`reports/final/RESEARCH_CONSISTENCY_AUDIT.md`** | Cross-phase consistency audit. | **ARCHIVE** |

---

## 3. Merged and Deleted Files

All duplicate, redundant, or outdated markdown documents have been cleaned up to prevent claim drift:

- **`credit_risk/README.md`**: **DELETED** (Merged into `systems/credit_risk/README.md`).
- **`README_CRIS_SYSTEM.md`**: **DELETED** (Merged and archived in `docs/archive/README_CRIS_SYSTEM.md`).
- **`README_FINAL_REVIEW.md`**: **DELETED** (Archived in `docs/archive/README_FINAL_REVIEW.md`).
- **`README_METRIC_AUDIT.md`**: **DELETED** (Archived in `docs/archive/README_METRIC_AUDIT.md`).
- **`README_REVIEW_CHECKLIST.md`**: **DELETED** (Archived in `docs/archive/README_REVIEW_CHECKLIST.md`).
