"""
stress_test_layer3.py — Layer 3 BSSC Stress Test

Runs predefined extreme scenarios against the complete
Layer 3 pipeline. Evaluates, logs to WandB, and generates
a permanent markdown report.
"""

import sys
import logging
import argparse
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

# Try to import wandb, but never let its absence or failure stop the test.
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from layer3_bssc.engine.models import Layer3Report
from layer3_bssc.auditor.detector import run_layer3_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PREDEFINED SCENARIOS
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "id": "ST-001",
        "name": "COVID Crash",
        "description": "Macro-driven sharp crash. Primary BSSC validation event.",
        "ticker": "SPY",
        "event_start": "2020-02-01",
        "event_end":   "2020-03-23",
        "calm_start":  "2018-01-01",
        "calm_end":    "2018-12-31",
        "expected_market_states": ["BLACK_SWAN"],
        "expected_actions":       ["LIQUIDATE"],
        "expected_min_breach_days": 15,
        "expected_min_jd_gbm_ratio": 1.5,
        "test_type": "TRUE_POSITIVE",
        "severity": "CRITICAL",
    },
    {
        "id": "ST-002",
        "name": "Q4 2018 Selloff",
        "description": "Fed rate hike driven selloff. Moderate crisis, slower development.",
        "ticker": "SPY",
        "event_start": "2018-10-01",
        "event_end":   "2018-12-31",
        "calm_start":  "2018-01-01",
        "calm_end":    "2018-09-30",
        "expected_market_states": ["STRESS", "BLACK_SWAN"],
        "expected_actions":       ["REDUCE", "LIQUIDATE"],
        "expected_min_breach_days": 8,
        "expected_min_jd_gbm_ratio": 1.5,
        "test_type": "TRUE_POSITIVE",
        "severity": "HIGH",
    },
    {
        "id": "ST-003",
        "name": "Vaccine Rally",
        "description": "Pfizer announcement Nov 2020. Positive shock — must NOT fire BLACK_SWAN. Critical false positive test.",
        "ticker": "SPY",
        "event_start": "2020-11-09",
        "event_end":   "2020-11-20",
        "calm_start":  "2020-01-01",
        "calm_end":    "2020-01-31",
        "expected_market_states": ["NORMAL", "STRESS"],
        "expected_actions":       ["HOLD", "REDUCE"],
        "expected_max_breach_days": 6,
        "expected_min_jd_gbm_ratio": 1.5,
        "test_type": "TRUE_NEGATIVE",
        "severity": "CRITICAL",
    },
    {
        "id": "ST-004",
        "name": "Calm Bull Market 2019",
        "description": "Low volatility bull market H1 2019. Baseline test — system must stay silent.",
        "ticker": "SPY",
        "event_start": "2019-01-01",
        "event_end":   "2019-06-30",
        "calm_start":  "2018-01-01",
        "calm_end":    "2018-12-31",
        "expected_market_states": ["NORMAL"],
        "expected_actions":       ["HOLD"],
        "expected_max_breach_days": 3,
        "expected_min_jd_gbm_ratio": 1.5,
        "test_type": "TRUE_NEGATIVE",
        "severity": "HIGH",
    },
    {
        "id": "ST-005",
        "name": "2022 Fed Bear Market",
        "description": "Policy-driven slow bear market. Tests structural stress detection without a single crash day.",
        "ticker": "SPY",
        "event_start": "2022-01-01",
        "event_end":   "2022-06-30",
        "calm_start":  "2021-01-01",
        "calm_end":    "2021-12-31",
        "expected_market_states": ["STRESS", "BLACK_SWAN"],
        "expected_actions":       ["REDUCE", "LIQUIDATE"],
        "expected_min_breach_days": 8,
        "expected_min_jd_gbm_ratio": 1.5,
        "test_type": "TRUE_POSITIVE",
        "severity": "HIGH",
    },
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _write_text_atomic(content: str, final_path: Path) -> None:
    """Write text atomically using tempfile + os.replace."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=final_path.parent, suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            f.write(content)
        os.replace(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

def evaluate_scenario(scenario: dict, report: Layer3Report) -> dict:
    """Evaluates a Layer3Report against a scenario's expected outcomes."""
    checks = []
    passed = True

    # Check 1 - Market state
    actual_state = report.entropy.market_state
    state_passed = actual_state in scenario["expected_market_states"]
    checks.append({
        "check": "Market State",
        "expected": " or ".join(scenario["expected_market_states"]),
        "actual": actual_state,
        "passed": state_passed,
    })
    if not state_passed: passed = False

    # Check 2 - Recommended action
    actual_action = report.recommended_action
    action_passed = actual_action in scenario["expected_actions"]
    checks.append({
        "check": "Recommended Action",
        "expected": " or ".join(scenario["expected_actions"]),
        "actual": actual_action,
        "passed": action_passed,
    })
    if not action_passed: passed = False

    # Check 3 - Breach duration
    actual_breach = report.entropy.breach_duration_days
    if "expected_min_breach_days" in scenario:
        breach_passed = actual_breach >= scenario["expected_min_breach_days"]
        check_desc = f">= {scenario['expected_min_breach_days']} days"
    elif "expected_max_breach_days" in scenario:
        breach_passed = actual_breach <= scenario["expected_max_breach_days"]
        check_desc = f"<= {scenario['expected_max_breach_days']} days"
    else:
        breach_passed = True
        check_desc = "Any"
        
    checks.append({
        "check": "Breach Duration",
        "expected": check_desc,
        "actual": f"{actual_breach} days",
        "passed": breach_passed,
    })
    if not breach_passed: passed = False

    # Check 4 - JD/GBM ratio
    actual_ratio = report.slippage.jd_gbm_ratio
    ratio_passed = actual_ratio >= scenario["expected_min_jd_gbm_ratio"]
    checks.append({
        "check": "JD/GBM Ratio",
        "expected": f">= {scenario['expected_min_jd_gbm_ratio']}x",
        "actual": f"{actual_ratio:.2f}x",
        "passed": ratio_passed,
    })
    if not ratio_passed: passed = False

    # Check 5 - Pipeline consistency
    warnings = report.validate_pipeline_consistency()
    consistency_passed = len(warnings) == 0
    checks.append({
        "check": "Pipeline Consistency",
        "expected": "No warnings",
        "actual": f"{len(warnings)} warnings",
        "passed": consistency_passed,
    })
    # Warnings do not fail scenario

    return {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "test_type": scenario["test_type"],
        "severity": scenario["severity"],
        "overall_passed": passed,
        "checks": checks,
        "market_state": actual_state,
        "recommended_action": actual_action,
        "breach_duration_days": actual_breach,
        "mean_slippage_bps": report.slippage.mean_bps,
        "p99_slippage_bps": report.slippage.p99_bps,
        "jd_gbm_ratio": actual_ratio,
        "validation_passed": report.slippage.validation_passed,
        "consistency_warnings": warnings,
        "entropy_delta": report.entropy.entropy_delta,
        "peak_entropy_date": report.entropy.peak_entropy_date,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Layer 3 BSSC Stress Test")
    parser.add_argument("--no-wandb", action="store_true", help="Skip WandB logging")
    parser.add_argument("--quick", action="store_true", help="Run only ST-001 and ST-003")
    parser.add_argument("--stability-runs", type=int, default=3, help="Number of stability runs on ST-001")
    args = parser.parse_args()
    
    stability_runs_n = max(2, args.stability_runs)

    project_root = Path(__file__).resolve().parent.parent.parent
    csv_path = project_root / "data" / "Indices" / "SPY.csv"
    
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        logger.error("Run ingest_data.py first to fetch market data.")
        sys.exit(1)

    scenarios_to_run = SCENARIOS
    if args.quick:
        scenarios_to_run = [s for s in SCENARIOS if s["id"] in ("ST-001", "ST-003")]

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            wandb_run = wandb.init(
                project="CRIS",
                name="Layer3-StressTest",
                tags=["layer3", "stress-test", "validation"],
                reinit=True
            )
        except Exception as e:
            logger.warning(f"WandB init failed, continuing without logging: {e}")
            use_wandb = False
            wandb_run = None

    now_str = datetime.now().isoformat()
    
    print("=" * 60)
    print(" CRIS LAYER 3 — STRESS TEST")
    print(f" {len(scenarios_to_run)} scenarios | SPY | {now_str}")
    print("=" * 60)

    results = []
    failed_scenarios = 0
    passed_scenarios = 0
    true_pos_passed = 0
    true_pos_total = 0
    true_neg_passed = 0
    true_neg_total = 0

    st001_p99 = None
    # For reporting scalars to wandb
    covid_state = "UNKNOWN"
    covid_breach = 0
    calc_vaccine_state = "UNKNOWN"
    calc_calm_state = "UNKNOWN"

    for scenario in scenarios_to_run:
        print(f"\n[{scenario['id']}] {scenario['name']} ({scenario['test_type']} — {scenario['severity']})")
        print("  Running full pipeline...")
        
        config = {
            "analysis_window_days": 252,
            "event_start": scenario["event_start"],
            "event_end": scenario["event_end"],
            "baseline_start": scenario["calm_start"],
            "baseline_end": scenario["calm_end"],
            "n_paths": 2000,
            "eta": 0.3, # Used for scaling execution costs
        }

        eval_res = None
        try:
            report = run_layer3_pipeline(
                ticker=scenario["ticker"],
                csv_path=csv_path,
                config=config,
                use_wandb=False,
            )
            eval_res = evaluate_scenario(scenario, report)
        except Exception as e:
            logger.exception(f"Exception during scenario {scenario['id']}: {e}")
            eval_res = {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "test_type": scenario["test_type"],
                "severity": scenario["severity"],
                "overall_passed": False,
                "error": str(e),
                "checks": [],
                "market_state": "ERROR",
                "recommended_action": "ERROR",
                "breach_duration_days": 0,
                "mean_slippage_bps": 0.0,
                "p99_slippage_bps": 0.0,
                "jd_gbm_ratio": 0.0,
                "validation_passed": False,
                "consistency_warnings": [f"Runtime execution failed: {e}"],
                "entropy_delta": 0.0,
                "peak_entropy_date": "",
            }

        results.append(eval_res)
        
        if eval_res["overall_passed"]:
            res_str = "✅ PASSED"
            passed_scenarios += 1
            if scenario["test_type"] == "TRUE_POSITIVE":
                true_pos_passed += 1
            else:
                true_neg_passed += 1
        else:
            if "error" in eval_res:
                res_str = "❌ ERROR"
            else:
                res_str = "❌ FAILED"
            failed_scenarios += 1
            
        if scenario["test_type"] == "TRUE_POSITIVE":
            true_pos_total += 1
        else:
            true_neg_total += 1
            
        if scenario["id"] == "ST-001":
            st001_p99 = eval_res["p99_slippage_bps"]
            covid_state = eval_res["market_state"]
            covid_breach = eval_res["breach_duration_days"]
        elif scenario["id"] == "ST-003":
            calc_vaccine_state = eval_res["market_state"]
        elif scenario["id"] == "ST-004":
            calc_calm_state = eval_res["market_state"]

        if "error" not in eval_res:
            for chk in eval_res["checks"]:
                mark = "✅" if chk["passed"] else ("⚠️" if "Consistency" in chk["check"] else "❌")
                print(f"  {mark} {chk['check']+':':<16} {chk['actual']:<10} (expected: {chk['expected']})")
        else:
             print(f"  ❌ ERROR: {eval_res['error']}")
        
        print(f"  RESULT: {res_str}")
        print("─" * 60)

    # -----------------------------------------------------------------------
    # STABILITY CHECK
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f" STABILITY CHECK — ST-001 COVID ({stability_runs_n} runs)")
    print("=" * 60)
    
    st001_scenario = next((s for s in SCENARIOS if s["id"] == "ST-001"), None)
    stability_p99s = []
    
    if st001_scenario is not None and st001_p99 is not None and ("error" not in next(r for r in results if r["scenario_id"] == "ST-001")):
        # We already have run 1
        stability_p99s.append(st001_p99)
        print(f"  Run 1 P99: {st001_p99:.2f} bps")
        
        config = {
            "analysis_window_days": 252,
            "event_start": st001_scenario["event_start"],
            "event_end": st001_scenario["event_end"],
            "baseline_start": st001_scenario["calm_start"],
            "baseline_end": st001_scenario["calm_end"],
            "n_paths": 2000,
            "eta": 0.3,
        }
        
        for i in range(2, stability_runs_n + 1):
            try:
                report = run_layer3_pipeline(
                    ticker=st001_scenario["ticker"],
                    csv_path=csv_path,
                    config=config,
                    use_wandb=False, 
                )
                p99 = report.slippage.p99_bps
                stability_p99s.append(p99)
                print(f"  Run {i} P99: {p99:.2f} bps")
            except Exception as e:
                logger.error(f"Stability check run {i} failed: {e}")
                
        if len(stability_p99s) == stability_runs_n:
            max_p99 = max(stability_p99s)
            min_p99 = min(stability_p99s)
            mean_p99 = sum(stability_p99s) / len(stability_p99s)
            variance_pct = ((max_p99 - min_p99) / mean_p99) * 100
            if variance_pct < 5:
                stability_status = "STABLE"
            elif variance_pct < 15:
                stability_status = "ACCEPTABLE"
            else:
                stability_status = "UNSTABLE — increase n_paths"
            print(f"  Variance:  {variance_pct:.2f}% — {stability_status}")
        else:
            variance_pct = 0.0
            stability_status = "INCOMPLETE"
    else:
        variance_pct = 0.0
        stability_status = "SKIPPED"
        print("  Skipped (ST-001 failed or missing)")

    # -----------------------------------------------------------------------
    # WANDB LOGGING
    # -----------------------------------------------------------------------
    if use_wandb and wandb_run is not None:
        try:
            wandb.log({
                "stress_test_total_scenarios": len(scenarios_to_run),
                "stress_test_passed": passed_scenarios,
                "stress_test_failed": failed_scenarios,
                "stress_test_pass_rate": passed_scenarios / len(scenarios_to_run) if scenarios_to_run else 0,
                "covid_market_state": covid_state,
                "covid_p99_bps": st001_p99 if st001_p99 is not None else 0.0,
                "covid_breach_days": covid_breach,
                "vaccine_rally_market_state": calc_vaccine_state,
                "calm_2019_market_state": calc_calm_state,
                "stability_variance_pct": variance_pct,
            })
            
            # Scenario table
            res_table = wandb.Table(columns=[
                "ID", "Name", "Type", "Severity", "Overall",
                "Market State", "Action", "Breach Days",
                "Mean Slippage", "P99 Slippage", "JD/GBM Ratio",
                "Validation Gate"
            ])
            for r in results:
                res_table.add_data(
                    r["scenario_id"], r["scenario_name"], r["test_type"], r["severity"],
                    "PASSED" if r["overall_passed"] else ("ERROR" if "error" in r else "FAILED"),
                    r["market_state"], r["recommended_action"], r["breach_duration_days"],
                    r["mean_slippage_bps"], r["p99_slippage_bps"], r["jd_gbm_ratio"],
                    r["validation_passed"]
                )
            wandb.log({"stress_test_results": res_table})
            
            # Stability table
            if stability_p99s:
                stab_table = wandb.Table(columns=["Run", "P99 Slippage", "Variance %"])
                for i, p99 in enumerate(stability_p99s, start=1):
                    stab_table.add_data(i, p99, variance_pct if i == 1 else None)
                wandb.log({"stability_check": stab_table})
                
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")

    # -----------------------------------------------------------------------
    # MARKDOWN REPORT
    # -----------------------------------------------------------------------
    reports_dir = project_root / "reports"
    md_fname = f"layer3_stress_test_results.md"
    md_path = reports_dir / md_fname
    
    md_lines = [
        "# CRIS Layer 3 — Stress Test Results",
        f"**Date:** {now_str}",
        f"**Total Scenarios:** {len(scenarios_to_run)}",
        f"**Passed:** {passed_scenarios} / {len(scenarios_to_run)}",
        f"**Overall Status:** {'✅ ALL PASSED' if failed_scenarios == 0 else f'❌ {failed_scenarios} FAILED'}",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| ID | Scenario | Type | Severity | State | Action | Breach | P99 | Ratio | Result |",
        "|----|----------|------|----------|-------|--------|--------|-----|-------|--------|",
    ]
    
    for r in results:
        res_mark = "✅" if r["overall_passed"] else ("❌" if "error" not in r else "💥 ERROR")
        md_lines.append(
            f"| {r['scenario_id']} | {r['scenario_name']} | {r['test_type']} | {r['severity']} | "
            f"{r['market_state']} | {r['recommended_action']} | {r['breach_duration_days']} | "
            f"{r['p99_slippage_bps']:.1f} | {r['jd_gbm_ratio']:.2f}x | {res_mark} |"
        )

    md_lines.extend(["", "---", "", "## Detailed Results", ""])
    
    for r in results:
        res_mark = "✅" if r["overall_passed"] else ("❌" if "error" not in r else "💥 ERROR")
        s_desc = next((x["description"] for x in scenarios_to_run if x["id"] == r["scenario_id"]), "")
        
        md_lines.extend([
            f"### {r['scenario_id']} — {r['scenario_name']} {res_mark}",
            f"**Test Type:** {r['test_type']}",
            f"**Description:** {s_desc}",
            "",
            "#### Check Results"
        ])
        
        if "error" in r:
            md_lines.extend([
                f"**Runtime Error:** {r['error']}",
                ""
            ])
            continue
            
        md_lines.extend([
            "| Check | Expected | Actual | Result |",
            "|-------|----------|--------|--------|"
        ])
        for c in r["checks"]:
            c_mark = "✅" if c["passed"] else ("⚠️" if "Consistency" in c["check"] else "❌")
            md_lines.append(f"| {c['check']} | {c['expected']} | {c['actual']} | {c_mark} |")
            
        md_lines.extend([
            "",
            "#### Key Metrics",
            f"- Entropy Delta: {r['entropy_delta']:.4f}",
            f"- Peak Entropy Date: {r['peak_entropy_date']}",
            f"- Mean Slippage: {r['mean_slippage_bps']:.2f} bps",
            f"- P99 Slippage: {r['p99_slippage_bps']:.2f} bps",
            ""
        ])

    md_lines.extend(["---", "", "## Stability Check (ST-001 COVID — 3 Runs)", ""])
    if stability_p99s:
        md_lines.extend([
            "| Run | P99 Slippage (bps) |",
            "|-----|-------------------|"
        ])
        for i, p in enumerate(stability_p99s, start=1):
             md_lines.append(f"| {i} | {p:.2f} |")
        md_lines.extend([
             "",
             f"**Variance:** {variance_pct:.2f}% — {stability_status}",
             ""
        ])
    else:
        md_lines.extend(["*Stability check skipped or incomplete.*", ""])

    md_lines.extend(["---", "", "## Failed Scenarios", ""])
    if failed_scenarios == 0:
        md_lines.extend(["✅ No failures detected.", ""])
    else:
        for r in results:
            if not r["overall_passed"]:
                md_lines.extend([
                    f"### {r['scenario_id']} — {r['scenario_name']}",
                ])
                if "error" in r:
                    md_lines.append(f"- **Runtime Error:** {r['error']}")
                else:
                    for c in r["checks"]:
                        if not c["passed"] and "Consistency" not in c["check"]:
                            md_lines.append(f"- **Failed:** {c['check']}")
                            md_lines.append(f"  - Expected: {c['expected']}")
                            md_lines.append(f"  - Actual: {c['actual']}")
                md_lines.append("- *Suggested investigation:* Check models.py defaults, or entropy baseline calibration.")
                md_lines.append("")

    url = wandb_run.get_url() if wandb_run else "None"
    md_lines.extend([
        "---",
        "",
        "## WandB Run",
        url,
        "",
        "---",
        "*Generated by CRIS Layer 3 Stress Test*",
        "*Layer 3 BSSC v1.0*",
        ""
    ])

    _write_text_atomic("\n".join(md_lines), md_path)

    # -----------------------------------------------------------------------
    # FINAL SUMMARY CONSOLE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    print("=" * 60)
    print(f"  Passed:  {passed_scenarios} / {len(scenarios_to_run)}")
    print(f"  Failed:  {failed_scenarios} / {len(scenarios_to_run)}")
    print(f"  Status:  {'✅ ALL PASSED' if failed_scenarios == 0 else '❌ FAILURES DETECTED'}")
    print()
    print(f"  TRUE POSITIVES:  {true_pos_passed} / {true_pos_total} detected correctly")
    print(f"  TRUE NEGATIVES:  {true_neg_passed} / {true_neg_total} correctly suppressed")
    print()
    print("  Reports saved:")
    print(f"    → {md_path.relative_to(project_root) if project_root in md_path.parents else md_path}")
    print(f"    → WandB: {url}")
    print("=" * 60)
    
    if use_wandb and wandb_run is not None:
        try:
            wandb.finish()
        except:
            pass

    sys.exit(0 if failed_scenarios == 0 else 1)

if __name__ == "__main__":
    main()
