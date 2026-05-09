"""
CRIS Phase 2 macro-conditioning experiment.

This script keeps the Phase 1 borrower model fixed and evaluates whether
Layer 3 probabilistic environmental states can improve credit governance
behavior during deteriorating market environments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from configs.credit_config import MODEL_DIR, OUTPUT_DIR
from orchestration.run_macro_harvesters import Layer3State, run_layer3
from systems.credit_risk.overlays.bayesian_overlay import (
    apply_bayesian_pressure_overlay,
    fit_bayesian_pressure_overlay,
)
from systems.credit_risk.overlays.confidence_adjustment import environmental_confidence
from systems.credit_risk.overlays.macro_conditioning import (
    align_market_states_to_loans,
    compute_macro_stress_score,
)


MARKET_CACHE = OUTPUT_DIR / "phase2_spy_market_2005_2018.csv"
STATE_CACHE = OUTPUT_DIR / "phase2_layer3_macro_states.csv"
LOAN_AUDIT_PATH = OUTPUT_DIR / "phase2_loan_level_audit_sample.csv"
HISTORICAL_DIAGNOSTIC_PATH = OUTPUT_DIR / "phase2_historical_diagnostic_by_year.csv"
RESULTS_PATH = OUTPUT_DIR / "phase2_macro_conditioning_results.json"
REPORT_PATH = OUTPUT_DIR / "final_phase2_report.md"


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def load_or_download_market_history(force: bool = False) -> pd.DataFrame:
    """Load cached SPY history or download the historical window needed here."""

    if MARKET_CACHE.exists() and not force:
        market = pd.read_csv(MARKET_CACHE, parse_dates=["Date"])
    else:
        market = yf.download(
            "SPY",
            start="2005-01-01",
            end="2019-01-01",
            auto_adjust=True,
            progress=False,
        )
        if market.empty:
            raise RuntimeError("SPY download returned no rows; cannot build Phase 2 market states.")
        market = _flatten_yfinance_columns(market).reset_index()
        market.to_csv(MARKET_CACHE, index=False)

    market["Date"] = pd.to_datetime(market["Date"])
    market = market.sort_values("Date")
    return market[["Date", "Close"]].dropna().reset_index(drop=True)


def generate_monthly_layer3_states(market: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Run Layer 3 at loan-issuance monthly cadence with no future data."""

    if STATE_CACHE.exists() and not force:
        return pd.read_csv(STATE_CACHE, parse_dates=["state_date"])

    market = market.copy()
    market["returns"] = market["Close"].pct_change()
    market = market.dropna().reset_index(drop=True)

    issue_months = pd.date_range("2007-06-01", "2018-12-01", freq="MS")
    state = Layer3State()
    records = []

    for issue_month in issue_months:
        available = market[market["Date"] <= issue_month]
        if len(available) < 60:
            continue

        prices = pd.Series(available["Close"].values, index=available["Date"])
        returns = pd.Series(available["returns"].values, index=available["Date"])
        output, state = run_layer3(
            returns=returns,
            prices=prices,
            ticker="SPY",
            state=state,
            enable_adaptive_calibration=False,
        )

        records.append(
            {
                "issue_month": issue_month,
                "state_date": available["Date"].iloc[-1],
                "uncertainty_pressure": output.meta.uncertainty_pressure,
                "structural_fragility": output.slow.fragility_pressure,
                "liquidity_disruption": output.fast.liquidity_disruption,
                "stabilization_strength": output.meta.stabilization_strength,
                "trajectory_fragility": output.decay.trajectory_fragility,
                "dominant_field": output.meta.dominant_field.value,
                "signal_coherence": output.meta.signal_coherence,
                "shock_intensity": output.fast.shock_intensity,
                "structural_instability": output.slow.structural_instability,
                "erosion_strength": output.decay.erosion_strength,
            }
        )

    states = pd.DataFrame(records)
    states["macro_stress_score"] = compute_macro_stress_score(states)
    states["environmental_confidence"] = environmental_confidence(
        states["uncertainty_pressure"],
        states["macro_stress_score"],
    )
    states.to_csv(STATE_CACHE, index=False)
    return states


def load_phase1_sets() -> dict[str, pd.DataFrame | pd.Series]:
    """Load the saved Phase 1 temporal splits and issue dates."""

    engineered_dates = pd.read_parquet(OUTPUT_DIR / "engineered_data.parquet", columns=["issue_d", "target"])
    engineered_dates["issue_d"] = pd.to_datetime(engineered_dates["issue_d"])

    train_mask = engineered_dates["issue_d"].dt.year <= 2015
    val_mask = engineered_dates["issue_d"].dt.year.between(2016, 2017)
    test_mask = engineered_dates["issue_d"].dt.year >= 2018

    return {
        "X_train": pd.read_parquet(OUTPUT_DIR / "X_train.parquet"),
        "y_train": pd.read_parquet(OUTPUT_DIR / "y_train.parquet").iloc[:, 0],
        "date_train": engineered_dates.loc[train_mask, "issue_d"].reset_index(drop=True),
        "X_val": pd.read_parquet(OUTPUT_DIR / "X_val.parquet"),
        "y_val": pd.read_parquet(OUTPUT_DIR / "y_val.parquet").iloc[:, 0],
        "date_val": engineered_dates.loc[val_mask, "issue_d"].reset_index(drop=True),
        "X_test": pd.read_parquet(OUTPUT_DIR / "X_test.parquet"),
        "y_test": pd.read_parquet(OUTPUT_DIR / "y_test.parquet").iloc[:, 0],
        "date_test": engineered_dates.loc[test_mask, "issue_d"].reset_index(drop=True),
        "date_all": engineered_dates["issue_d"].reset_index(drop=True),
        "target_all": engineered_dates["target"].reset_index(drop=True),
    }


def predict_borrower_pd(data: dict[str, pd.DataFrame | pd.Series]) -> dict[str, np.ndarray]:
    model = joblib.load(MODEL_DIR / "lightgbm.joblib")
    return {
        "train": model.predict_proba(data["X_train"])[:, 1],
        "val": model.predict_proba(data["X_val"])[:, 1],
        "test": model.predict_proba(data["X_test"])[:, 1],
    }


def classification_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def expected_calibration_error(y_true: pd.Series, probability: np.ndarray, bins: int = 10) -> float:
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(probability)})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    total = len(df)
    ece = 0.0
    for _, group in df.groupby("bin", observed=False):
        if group.empty:
            continue
        ece += len(group) / total * abs(group["y"].mean() - group["p"].mean())
    return float(ece)


def metric_bundle(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "ece_10bin": expected_calibration_error(y_true, probability),
    }
    metrics.update(classification_metrics(y_true, probability, threshold))
    return metrics


def governance_metrics(
    y_true: pd.Series,
    base_pd: np.ndarray,
    macro_pd: np.ndarray,
    stress_score: pd.Series,
    env_confidence: pd.Series,
    approval_cutoff: float,
    stress_cutoff: float,
) -> dict[str, float]:
    y = np.asarray(y_true)
    base_approved = base_pd <= approval_cutoff
    macro_approved = macro_pd <= approval_cutoff
    base_fn = base_approved & (y == 1)
    macro_fn = macro_approved & (y == 1)
    high_stress = np.asarray(stress_score) >= stress_cutoff
    intercepted = base_fn & (~macro_approved)

    def _default_rate(mask: np.ndarray) -> float:
        return float(y[mask].mean()) if mask.any() else 0.0

    return {
        "approval_cutoff": float(approval_cutoff),
        "stress_cutoff": float(stress_cutoff),
        "baseline_approval_rate": float(base_approved.mean()),
        "macro_approval_rate": float(macro_approved.mean()),
        "baseline_approved_default_rate": _default_rate(base_approved),
        "macro_approved_default_rate": _default_rate(macro_approved),
        "baseline_false_negatives": int(base_fn.sum()),
        "macro_false_negatives": int(macro_fn.sum()),
        "intercepted_stealth_defaulters": int(intercepted.sum()),
        "baseline_fn_high_stress_share": float((base_fn & high_stress).sum() / max(base_fn.sum(), 1)),
        "macro_fn_high_stress_share": float((macro_fn & high_stress).sum() / max(macro_fn.sum(), 1)),
        "mean_env_confidence_all": float(np.asarray(env_confidence).mean()),
        "mean_env_confidence_false_negatives": float(np.asarray(env_confidence)[macro_fn].mean())
        if macro_fn.any()
        else 0.0,
    }


def segment_metrics(
    y_true: pd.Series,
    base_pd: np.ndarray,
    macro_pd: np.ndarray,
    stress_score: pd.Series,
    stress_cutoff: float,
) -> pd.DataFrame:
    high = np.asarray(stress_score) >= stress_cutoff
    rows = []
    for label, mask in [("high_stress", high), ("lower_stress", ~high)]:
        if mask.sum() < 2 or len(np.unique(np.asarray(y_true)[mask])) < 2:
            continue
        for system, pd_values in [("baseline", base_pd), ("macro_conditioned", macro_pd)]:
            rows.append(
                {
                    "segment": label,
                    "system": system,
                    "n": int(mask.sum()),
                    "default_rate": float(np.asarray(y_true)[mask].mean()),
                    "roc_auc": float(roc_auc_score(np.asarray(y_true)[mask], pd_values[mask])),
                    "pr_auc": float(average_precision_score(np.asarray(y_true)[mask], pd_values[mask])),
                    "brier": float(brier_score_loss(np.asarray(y_true)[mask], pd_values[mask])),
                    "ece_10bin": expected_calibration_error(np.asarray(y_true)[mask], pd_values[mask]),
                }
            )
    return pd.DataFrame(rows)


def monthly_table(
    dates: pd.Series,
    y_true: pd.Series,
    base_pd: np.ndarray,
    macro_pd: np.ndarray,
    macro_context: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "issue_month": pd.to_datetime(dates).dt.to_period("M").astype(str),
            "target": np.asarray(y_true),
            "baseline_pd": base_pd,
            "macro_pd": macro_pd,
            "macro_stress_score": np.asarray(macro_context["macro_stress_score"]),
            "environmental_confidence": np.asarray(macro_context["environmental_confidence"]),
            "dominant_field": np.asarray(macro_context["dominant_field"]),
        }
    )
    grouped = df.groupby("issue_month")
    rows = []
    for month, group in grouped:
        if group["target"].nunique() < 2:
            auc_base = np.nan
            auc_macro = np.nan
        else:
            auc_base = roc_auc_score(group["target"], group["baseline_pd"])
            auc_macro = roc_auc_score(group["target"], group["macro_pd"])
        rows.append(
            {
                "issue_month": month,
                "n": int(len(group)),
                "default_rate": float(group["target"].mean()),
                "macro_stress_score": float(group["macro_stress_score"].mean()),
                "environmental_confidence": float(group["environmental_confidence"].mean()),
                "dominant_field": group["dominant_field"].mode().iloc[0],
                "baseline_brier": float(brier_score_loss(group["target"], group["baseline_pd"])),
                "macro_brier": float(brier_score_loss(group["target"], group["macro_pd"])),
                "baseline_auc": float(auc_base) if not np.isnan(auc_base) else None,
                "macro_auc": float(auc_macro) if not np.isnan(auc_macro) else None,
            }
        )
    return pd.DataFrame(rows)


def yearly_diagnostic_table(
    dates: pd.Series,
    y_true: np.ndarray,
    base_pd: np.ndarray,
    macro_pd: np.ndarray,
    macro_context: pd.DataFrame,
) -> pd.DataFrame:
    """Create a year-level diagnostic table.

    Years through 2015 are in-sample for the saved borrower model and must be
    interpreted as behavioral diagnostics, not validation evidence.
    """

    df = pd.DataFrame(
        {
            "year": pd.to_datetime(dates).dt.year,
            "target": np.asarray(y_true),
            "baseline_pd": base_pd,
            "macro_pd": macro_pd,
            "macro_stress_score": np.asarray(macro_context["macro_stress_score"]),
            "environmental_confidence": np.asarray(macro_context["environmental_confidence"]),
        }
    )
    rows = []
    for year, group in df.groupby("year"):
        if group["target"].nunique() < 2:
            continue
        rows.append(
            {
                "year": int(year),
                "n": int(len(group)),
                "validation_status": "out_of_time" if year >= 2018 else "validation" if year >= 2016 else "in_sample_diagnostic",
                "default_rate": float(group["target"].mean()),
                "macro_stress_score": float(group["macro_stress_score"].mean()),
                "environmental_confidence": float(group["environmental_confidence"].mean()),
                "baseline_auc": float(roc_auc_score(group["target"], group["baseline_pd"])),
                "macro_auc": float(roc_auc_score(group["target"], group["macro_pd"])),
                "baseline_brier": float(brier_score_loss(group["target"], group["baseline_pd"])),
                "macro_brier": float(brier_score_loss(group["target"], group["macro_pd"])),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def write_report(results: dict, report_path: Path = REPORT_PATH) -> None:
    comparative = pd.DataFrame(results["comparative"])
    stress_segments = pd.DataFrame(results["stress_segments"])
    monthly_2018 = pd.DataFrame(results["monthly_2018"])
    yearly_diagnostic = pd.DataFrame(results["historical_diagnostic_by_year"])
    governance = results["governance"]
    overlay = results["overlay_config"]

    test_base = comparative[(comparative["split"] == "2018_test") & (comparative["system"] == "baseline")].iloc[0]
    test_macro = comparative[
        (comparative["split"] == "2018_test") & (comparative["system"] == "macro_conditioned")
    ].iloc[0]

    report = f"""# Phase 2 Final Report: Probabilistic Macro Conditioning Overlay

## 1. Executive Summary
Phase 2 preserved the Phase 1 standalone borrower-risk model as the primary estimator and tested a bounded CRIS Layer 3 macro-conditioning overlay. The experiment asks whether probabilistic market-state interpretation can improve lending robustness during deteriorating environments, not whether CRIS predicts defaults or markets.

On the 2018 out-of-time test set, the macro-conditioned overlay produced ROC-AUC **{test_macro['roc_auc']:.4f}** versus baseline **{test_base['roc_auc']:.4f}**, PR-AUC **{test_macro['pr_auc']:.4f}** versus **{test_base['pr_auc']:.4f}**, and Brier score **{test_macro['brier']:.4f}** versus **{test_base['brier']:.4f}**. This is not an average-case improvement. The overlay made governance more defensive: approval rate declined from **{governance['baseline_approval_rate']:.2%}** to **{governance['macro_approval_rate']:.2%}**, approved-loan default rate moved from **{governance['baseline_approved_default_rate']:.2%}** to **{governance['macro_approved_default_rate']:.2%}**, and **{governance['intercepted_stealth_defaulters']}** baseline-approved defaulters were moved into review by environmental pressure. However, 2018 high-stress calibration worsened, so the evidence supports conditional governance caution rather than a stronger predictive claim.

## 2. Research Motivation
Phase 1 estimated:

`P(Default | Borrower Features)`

Phase 2 investigates:

`P(Default | Borrower Features, Probabilistic Market State)`

The research hypothesis is modest: a borrower-only model may rank applicants reasonably in stable regimes while becoming overconfident when the surrounding system deteriorates. CRIS Layer 3 is used only as an environmental interpretation framework.

## 3. Temporal Synchronization Methodology
LendingClub issue dates are month-level timestamps. For each issue month, the experiment used the latest available SPY market observation on or before the issue date and ran Layer 3 using only market history available up to that date. The join was a backward as-of synchronization; no future market data, loan performance data, or hindsight crisis labels entered the conditioning record.

Artifacts generated:

- `credit_risk/outputs/phase2_spy_market_2005_2018.csv`
- `credit_risk/outputs/phase2_layer3_macro_states.csv`
- `credit_risk/outputs/phase2_macro_conditioning_results.json`
- `credit_risk/outputs/phase2_historical_diagnostic_by_year.csv`

## 4. Layer 3 Environmental-State Design
The allowed Layer 3 outputs were retained as environmental descriptors:

- `uncertainty_pressure`
- `structural_fragility`
- `liquidity_disruption`
- `stabilization_strength`
- `trajectory_fragility`
- `dominant_field`

These were not treated as raw market predictors. A compact audit score, `macro_stress_score`, was derived from permitted descriptors only. It combines uncertainty, structural fragility, liquidity disruption, trajectory fragility, and weak stabilization. Raw SPY returns/prices were never joined to loan records.

## 5. Conditioning Architecture
The borrower probability from Phase 1 LightGBM remained the base PD. The overlay applied a bounded validation-calibrated log-odds pressure shift:

`logit(PD_macro) = logit(PD_borrower) + beta * max(0, macro_stress_score - validation_anchor)`

Fitted overlay parameters:

- Validation stress anchor: **{overlay['stress_anchor']:.4f}**
- Beta: **{overlay['beta']:.4f}**
- Maximum logit shift: **{overlay['max_logit_shift']:.4f}**

The overlay can raise risk under excess environmental pressure, but it cannot lower borrower risk in benign regimes.

## 6. Comparative Baseline Results
{markdown_table(comparative[['split', 'system', 'roc_auc', 'pr_auc', 'brier', 'log_loss', 'ece_10bin', 'f1']])}

## 7. Stress-Period Analysis
Stress periods were defined prospectively from the top quartile of 2018 Layer 3 `macro_stress_score`, not from default outcomes. This keeps the stress segmentation environmental rather than label-driven.

{markdown_table(stress_segments)}

2018 monthly deterioration view:

{markdown_table(monthly_2018[['issue_month', 'n', 'default_rate', 'macro_stress_score', 'environmental_confidence', 'dominant_field', 'baseline_brier', 'macro_brier']])}

Historical structural-break diagnostic:

{markdown_table(yearly_diagnostic[['year', 'validation_status', 'n', 'default_rate', 'macro_stress_score', 'baseline_auc', 'macro_auc', 'baseline_brier', 'macro_brier']])}

The 2007-2015 rows are included to inspect crisis-era behavior and environmental alignment only. They are not out-of-time evidence because the saved Phase 1 borrower model was trained on those years.

## 8. Calibration Analysis
The overlay was tuned on 2016-2017 validation data using Brier score with log-loss tie breaking. On 2018, calibration changed from Brier **{test_base['brier']:.4f}** to **{test_macro['brier']:.4f}** and 10-bin ECE from **{test_base['ece_10bin']:.4f}** to **{test_macro['ece_10bin']:.4f}**.

This is not a universal calibration victory. The overlay is useful only if Layer 3 pressure corresponds to borrower-model underconfidence or overconfidence in a given regime. In late 2018, Layer 3 correctly saw market deterioration, but realized defaults in the closed-loan sample were low for those issue months; the overlay therefore raised risk when the observed credit sample did not validate the increase.

## 9. Failure-Mode Analysis
The overlay is designed to matter most when borrower risk is near the governance boundary and Layer 3 reports elevated uncertainty or deterioration. In this experiment, that design reduced approvals and intercepted some baseline false negatives, but it did not improve 2018 ranking or stress-segment calibration.

Observed failure modes:

- Month-level loan timestamps limit precise daily synchronization.
- SPY is a broad environmental proxy, not a consumer-credit macro panel.
- 2007-2015 diagnostics are in-sample for the saved borrower model and should not be read as out-of-time validation.
- The 2018 test window contains deterioration episodes, but not a full credit crisis.
- A single monotone overlay cannot model every borrower-macro interaction.
- The fitted beta reached the allowed cap, which is a warning sign: the validation window favored stronger defensive pressure than the bounded institutional policy allowed, while the 2018 test window did not reward that pressure on average.

## 10. Institutional Interpretation
Phase 2 does not show that CRIS predicts defaults. It shows that an uncertainty-aware market-state overlay can make lending governance less aggressive when environmental pressure rises. This is aligned with the CRIS philosophy: probabilistic market stress should influence how financial systems behave.

## 11. Practical Limitations
The current overlay uses one market proxy and one bounded coefficient fitted on validation data. It should be treated as a governance-control experiment, not a production credit policy. Future work should test additional pre-specified environmental sources, preserve strict release lags, and validate across lenders or vintages before institutional deployment.

## 12. Future Layer 4 Governance Implications
Layer 4 can use Phase 2 outputs as defensive policy inputs:

- route high-uncertainty approvals to manual review,
- raise capital or reserve buffers during macro deterioration,
- adapt approval thresholds without replacing borrower PDs,
- monitor false-negative concentration in stressed regimes,
- separate model confidence from environmental confidence.

Final institutional rule retained: CRIS interprets probabilistic macro stress; it does not claim causal certainty or market/default prediction.
"""
    report_path.write_text(report)


def run_phase2(force_market: bool = False, force_states: bool = False) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    market = load_or_download_market_history(force=force_market)
    states = generate_monthly_layer3_states(market, force=force_states)
    data = load_phase1_sets()
    borrower_pd = predict_borrower_pd(data)

    aligned_val = align_market_states_to_loans(data["date_val"], states)
    aligned_test = align_market_states_to_loans(data["date_test"], states)
    aligned_train = align_market_states_to_loans(data["date_train"], states)

    overlay_config = fit_bayesian_pressure_overlay(
        borrower_pd["val"],
        data["y_val"],
        aligned_val["macro_stress_score"],
    )
    macro_val = apply_bayesian_pressure_overlay(
        borrower_pd["val"],
        aligned_val["macro_stress_score"],
        overlay_config,
    )
    macro_test = apply_bayesian_pressure_overlay(
        borrower_pd["test"],
        aligned_test["macro_stress_score"],
        overlay_config,
    )
    macro_train = apply_bayesian_pressure_overlay(
        borrower_pd["train"],
        aligned_train["macro_stress_score"],
        overlay_config,
    )

    classification_threshold = float(np.quantile(borrower_pd["val"], 0.80))
    approval_cutoff = classification_threshold
    stress_cutoff = float(np.quantile(aligned_test["macro_stress_score"], 0.75))

    comparative = []
    for split, y, base, macro in [
        ("2016_2017_validation", data["y_val"], borrower_pd["val"], macro_val),
        ("2018_test", data["y_test"], borrower_pd["test"], macro_test),
    ]:
        comparative.append({"split": split, "system": "baseline", **metric_bundle(y, base, classification_threshold)})
        comparative.append(
            {"split": split, "system": "macro_conditioned", **metric_bundle(y, macro, classification_threshold)}
        )

    stress_segments = segment_metrics(
        data["y_test"],
        borrower_pd["test"],
        macro_test,
        aligned_test["macro_stress_score"],
        stress_cutoff,
    )
    monthly_2018 = monthly_table(
        data["date_test"],
        data["y_test"],
        borrower_pd["test"],
        macro_test,
        aligned_test,
    )
    all_dates = pd.concat([data["date_train"], data["date_val"], data["date_test"]], ignore_index=True)
    all_targets = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    all_base = np.concatenate([borrower_pd["train"], borrower_pd["val"], borrower_pd["test"]])
    all_macro = np.concatenate([macro_train, macro_val, macro_test])
    all_context = pd.concat([aligned_train, aligned_val, aligned_test], ignore_index=True)
    historical_diagnostic = yearly_diagnostic_table(
        all_dates,
        all_targets,
        all_base,
        all_macro,
        all_context,
    )
    historical_diagnostic.to_csv(HISTORICAL_DIAGNOSTIC_PATH, index=False)
    governance = governance_metrics(
        data["y_test"],
        borrower_pd["test"],
        macro_test,
        aligned_test["macro_stress_score"],
        aligned_test["environmental_confidence"],
        approval_cutoff,
        stress_cutoff,
    )

    audit_sample = pd.DataFrame(
        {
            "issue_d": np.asarray(data["date_test"]),
            "target": np.asarray(data["y_test"]),
            "baseline_pd": borrower_pd["test"],
            "macro_conditioned_pd": macro_test,
            "macro_stress_score": np.asarray(aligned_test["macro_stress_score"]),
            "environmental_confidence": np.asarray(aligned_test["environmental_confidence"]),
            "dominant_field": np.asarray(aligned_test["dominant_field"]),
        }
    )
    audit_sample.sample(min(5000, len(audit_sample)), random_state=42).to_csv(LOAN_AUDIT_PATH, index=False)

    results = {
        "overlay_config": {
            "stress_anchor": overlay_config.stress_anchor,
            "beta": overlay_config.beta,
            "max_logit_shift": overlay_config.max_logit_shift,
        },
        "classification_threshold": classification_threshold,
        "comparative": comparative,
        "stress_segments": stress_segments.to_dict(orient="records"),
        "monthly_2018": monthly_2018.to_dict(orient="records"),
        "historical_diagnostic_by_year": historical_diagnostic.to_dict(orient="records"),
        "governance": governance,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    write_report(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CRIS Phase 2 macro-conditioning experiment.")
    parser.add_argument("--force-market", action="store_true", help="Re-download market data.")
    parser.add_argument("--force-states", action="store_true", help="Regenerate Layer 3 monthly states.")
    args = parser.parse_args()
    results = run_phase2(force_market=args.force_market, force_states=args.force_states)
    test_rows = [
        row for row in results["comparative"] if row["split"] == "2018_test"
    ]
    print(pd.DataFrame(test_rows).to_string(index=False))
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
