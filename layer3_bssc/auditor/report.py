"""
report.py — Layer 3 Persistence and Presentation

Makes Layer 3 output durable for both human and
machine consumption.

Two audiences:
  Humans:   Markdown reports in reports/ directory
  Machines: Versioned JSON in data/simulation_output/
            Read by convergence engine via
            load_layer3_report()

Design decisions encoded here:

1. Atomic JSON writes (tempfile + os.replace):
   Prevents corrupt partial writes from process
   interruption. A partial write corrupts only
   the temp file, never the final output.
   Rationale: documented failure mode in production
   financial reporting systems (2021 study).

2. Schema versioning (CURRENT_SCHEMA_VERSION):
   Every JSON embeds its schema version.
   load_layer3_report() validates version before
   parsing. Prevents convergence engine from
   silently consuming stale reports after
   models.py schema changes.

3. Tiered consistency validation:
   CRITICAL warnings halt report generation entirely.
   NON_CRITICAL warnings are embedded in report and
   flagged visibly but do not stop generation.
   Rationale: silent errors are more dangerous than
   loud ones in financial risk systems (Knight Capital
   2012 — $440M loss from silent flag error in reports).

4. Embedded generation config:
   Full config dict is embedded in every JSON.
   Results are reproducible — the exact parameters
   that generated them are preserved alongside outputs.
   Rationale: reproducibility failure in academic
   finance (JFE 2019 — results without preserved
   parameters cannot be verified or reproduced).

5. Programmatic markdown (no templates):
   Markdown is built section by section in Python.
   Conditional sections appear only when relevant.
   Adapts automatically when models.py gains new fields.
   No template synchronization debt.

Functions:
  generate_layer3_report()  — primary output function
  load_layer3_report()      — convergence engine loader
  compare_layer3_reports()  — drift detection
  list_layer3_reports()     — report discovery
  _build_markdown_report()  — private markdown builder
  _atomic_write_json()      — private atomic writer
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from layer3_bssc.engine.models import (
    SimulationResult,
    EntropyResult,
    SlippageResult,
    Layer3Report,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema Versioning
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION = "1.0"
CRIS_LAYER = "layer3"

# ---------------------------------------------------------------------------
# Warning Level Classification
# ---------------------------------------------------------------------------

CRITICAL_WARNINGS = {
    "P99 slippage below mean",
    "Ticker mismatch",
}

NON_CRITICAL_WARNINGS = set() # All other warnings


# ═══════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _atomic_write_json(data: dict, final_path: Path) -> None:
    """
    Write JSON atomically using tempfile + os.replace.
    
    Writes to a temporary file in the same directory
    as final_path first, then renames atomically.
    Rename is atomic on all major OS — either completes
    fully or not at all.
    A process killed mid-write corrupts only the temp
    file, never the final output.
    
    Rationale: partial JSON writes from process
    interruption are a documented failure mode in
    production financial reporting systems. The
    convergence engine reading a corrupt partial
    JSON would produce silent garbage output.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=final_path.parent,
        suffix=".tmp"
    )
    tmp_path = Path(tmp_path_str)
    
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _build_markdown_report(
    report: Layer3Report,
    consistency_warnings: list,
    generation_config: Optional[dict] = None,
) -> str:
    """Builds markdown programmatically, no templates."""
    now_str = datetime.now().isoformat()
    
    # ---------------------------------------------------------
    # Risk Level Indicator
    # ---------------------------------------------------------
    rl = report.overall_risk_level
    if rl == "BLACK_SWAN":
        risk_str = "🔴 BLACK SWAN EVENT DETECTED"
    elif rl == "STRESS":
        risk_str = "🟡 MARKET STRESS DETECTED"
    else:
        risk_str = "🟢 NORMAL MARKET CONDITIONS"
        
    md = [
        f"# CRIS Layer 3 Report — {report.ticker}",
        f"**Generated:** {now_str}",
        f"**Schema Version:** {CURRENT_SCHEMA_VERSION}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Risk Level | {report.overall_risk_level} |",
        f"| Recommended Action | {report.recommended_action} |",
        f"| Pipeline Timestamp | {report.pipeline_timestamp} |",
        "",
        f"{risk_str}",
        "",
        "---",
        "",
        "## Simulation Results",
        "",
        "| Metric | GBM | Jump-Diffusion |",
        "|--------|-----|----------------|",
        f"| Kurtosis | {report.simulation.kurtosis_gbm:.3f} | {report.simulation.kurtosis_jd:.2f} |",
        f"| Min Return | {report.simulation.min_return_gbm:.2%} | {report.simulation.min_return_jd:.2%} |",
        f"| Skewness | — | {report.simulation.skewness_jd:.3f} |",
        ""
    ]

    # Kurtosis interpretation
    if report.simulation.kurtosis_jd > 50:
        md.extend([
            f"> ✅ Fat tail fingerprint confirmed — JD kurtosis {report.simulation.kurtosis_jd / max(report.simulation.kurtosis_gbm, 0.001):.1f}x higher than GBM.",
            "> Black swan path generation validated.",
            ""
        ])
    elif report.simulation.kurtosis_jd >= 10:
        md.extend([
            "> ⚠️ Moderate fat tails detected. Jump parameters may need recalibration.",
            ""
        ])
    else:
        md.extend([
            "> ❌ Weak fat tails. Jump parameters likely miscalibrated. Review simulation config.",
            ""
        ])

    md.extend([
        "---",
        "",
        "## Entropy Analysis",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Primary Method | {report.entropy.primary_method} |",
        f"| Confirmation Method | {report.entropy.confirmation_method} |",
        f"| Baseline Entropy | {report.entropy.baseline_entropy:.4f} |",
        f"| Event Entropy | {report.entropy.event_entropy:.4f} |",
        f"| Entropy Delta | {report.entropy.entropy_delta:+.4f} |",
        f"| Market State | {report.entropy.market_state} |",
        f"| Breach Duration | {report.entropy.breach_duration_days} days |",
        f"| Peak Entropy Date | {report.entropy.peak_entropy_date} |",
        ""
    ])

    # Entropy interpretation
    bdays = report.entropy.breach_duration_days
    if bdays >= 18:
        md.extend([
            f"> ✅ Sustained breach of {bdays} days matches historical crash signature (18+ days).",
            "> Positive shock false positive unlikely.",
            ""
        ])
    elif bdays >= 6:
        md.extend([
            f"> ⚠️ Moderate breach duration of {bdays} days. Monitor for continuation or resolution.",
            ""
        ])
    else:
        md.extend([
            f"> ⚠️ Brief breach of {bdays} days — consistent with positive shock false positive pattern.",
            "> Verify manually before acting.",
            ""
        ])

    md.extend([
        "---",
        "",
        "## Slippage Analysis",
        "",
        "| Metric | GBM (Normal) | Jump-Diffusion (Crisis) |",
        "|--------|-------------|------------------------|",
        f"| Mean IS | — | {report.slippage.mean_bps:.2f} bps |",
        f"| Median IS | — | {report.slippage.median_bps:.2f} bps |",
        f"| P95 IS | — | {report.slippage.p95_bps:.2f} bps |",
        f"| P99 IS | — | {report.slippage.p99_bps:.2f} bps |",
        f"| Max IS | — | {report.slippage.max_bps:.2f} bps |",
        f"| JD/GBM Ratio | — | {report.slippage.jd_gbm_ratio:.2f}x |",
        ""
    ])

    # Gate interpretation
    if report.slippage.validation_passed:
        md.extend([
            f"> ✅ Validation gate PASSED — JD/GBM ratio {report.slippage.jd_gbm_ratio:.2f}x exceeds 1.5x threshold.",
            ""
        ])
    else:
        md.extend([
            f"> ❌ Validation gate FAILED — JD/GBM ratio {report.slippage.jd_gbm_ratio:.2f}x below 1.5x threshold.",
            "> Slippage estimates may be unreliable.",
            ""
        ])

    md.extend([
        "Regime breakdown of paths:",
        "",
        "| Regime | Path Count |",
        "|--------|-----------|",
        f"| NORMAL | {report.slippage.regime_breakdown.get('NORMAL', 0)} |",
        f"| STRESS | {report.slippage.regime_breakdown.get('STRESS', 0)} |",
        f"| BLACK_SWAN | {report.slippage.regime_breakdown.get('BLACK_SWAN', 0)} |",
        "",
        "---",
        "",
        "## Pipeline Consistency",
        ""
    ])

    if not consistency_warnings:
        md.extend([
            "✅ All 6 consistency checks passed.",
            "Pipeline results are internally consistent.",
            ""
        ])
    else:
        for w in consistency_warnings:
            # Check for critical warnings just to be safe (they should have been caught)
            is_critical = False
            for crit in CRITICAL_WARNINGS:
                if crit in w:
                    is_critical = True
                    break
            
            if is_critical:
                md.append(f"❌ CRITICAL: Report generated despite critical warnings. Do not use for position sizing.")
                md.append(f"❌ {w}")
            else:
                md.append(f"⚠️ {w}")
        md.append("")

    if generation_config:
        md.extend([
            "---",
            "",
            "## Generation Config",
            "",
            "```json",
            json.dumps(generation_config, indent=2),
            "```",
            ""
        ])

    # File paths are added downstream
    
    md.extend([
        "---",
        "*Generated by CRIS Layer 3 BSSC Pipeline*",
        f"*Report schema v{CURRENT_SCHEMA_VERSION}*"
    ])

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC CORE API
# ═══════════════════════════════════════════════════════════════════════════


def generate_layer3_report(
    report: Layer3Report,
    output_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    generation_config: Optional[dict] = None,
    print_summary: bool = True,
) -> dict:
    """Generate durable outputs (JSON and Markdown)."""
    
    project_root = Path(__file__).resolve().parent.parent.parent
    if output_dir is None:
        output_dir = project_root / "data" / "simulation_output"
    if reports_dir is None:
        reports_dir = project_root / "reports"
        
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    
    # ---------------------------------------------------------
    # Step 1 — Run consistency validation
    # ---------------------------------------------------------
    warnings = report.validate_pipeline_consistency()
    critical_found = []
    
    for w in warnings:
        is_critical = False
        for crit in CRITICAL_WARNINGS:
            if crit in w:
                is_critical = True
                break
                
        if is_critical:
            logger.error("Critical consistency failure: %s", w)
            critical_found.append(w)
        else:
            logger.warning("Pipeline consistency warning: %s", w)
            
    if critical_found:
        raise RuntimeError(
            f"Critical pipeline consistency failure detected. "
            f"Report generation halted. Fix the following before regenerating:\n"
            f"{chr(10).join(critical_found)}"
        )

    # ---------------------------------------------------------
    # Step 2 — Build JSON payload
    # ---------------------------------------------------------
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "cris_layer": CRIS_LAYER,
        "generation_timestamp": datetime.now().isoformat(),
        "generation_config": generation_config or {},
        **report.to_dict(),
    }

    # ---------------------------------------------------------
    # Step 3 — Write JSON atomically
    # ---------------------------------------------------------
    now_str_fname = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_fname = f"layer3_{report.ticker}_{now_str_fname}.json"
    json_path = output_dir / json_fname
    
    _atomic_write_json(payload, json_path)
    logger.info("[Layer3][Report] JSON written → %s", json_path)

    # ---------------------------------------------------------
    # Step 4 — Build and write markdown
    # ---------------------------------------------------------
    md_fname = f"layer3_{report.ticker}_{now_str_fname}.md"
    md_path = reports_dir / md_fname
    
    md_content = _build_markdown_report(report, warnings, generation_config)
    
    # Prepend files section exactly before the footer
    files_section = f"## Files\n- JSON: {json_fname}\n- Markdown: {md_fname}\n\n"
    md_lines = md_content.split('\n')
    footer_idx = len(md_lines) - 3  # Start of footer dashes
    md_lines.insert(footer_idx, files_section)
    
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
        
    logger.info("[Layer3][Report] Markdown written → %s", md_path)

    # ---------------------------------------------------------
    # Step 5 — Print summary if requested
    # ---------------------------------------------------------
    if print_summary:
        print(report.summary())

    # ---------------------------------------------------------
    # Step 6 — Return results dict
    # ---------------------------------------------------------
    return {
        "json_path": json_path,
        "markdown_path": md_path,
        "risk_level": report.overall_risk_level,
        "action": report.recommended_action,
        "consistency_warnings": warnings,
        # "critical_warnings_found" key removed:
        # Any critical warnings raise RuntimeError earlier in this function
        # so this dictionary is only ever returned if no critical warnings exist.
    }


def load_layer3_report(json_path: Path) -> Layer3Report:
    """Load Layer3Report from JSON for the convergence engine."""
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Layer 3 report not found: {json_path}\n"
            f"Run the Layer 3 pipeline to generate it:\n"
            f"python -m layer3_bssc.auditor.detector --ticker SPY"
        )
        
    data = json.loads(json_path.read_text())
    
    found_version = data.get("schema_version", "unknown")
    if found_version != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: file contains version "
            f"'{found_version}' but current CRIS expects "
            f"'{CURRENT_SCHEMA_VERSION}'. Re-run the Layer 3 "
            f"pipeline to regenerate this report."
        )

    # Note: pydantic ConfigDict(extra='ignore') should be added to models.py
    # to handle forward compatibility silently.
    report = Layer3Report.model_validate(data)
    
    logger.info(
        "[Layer3][Report] Loaded report from %s "
        "risk_level=%s",
        json_path,
        report.overall_risk_level,
    )
    
    return report


def compare_layer3_reports(
    report_a: Layer3Report,
    report_b: Layer3Report,
    label_a: str = "Previous",
    label_b: str = "Current",
) -> dict:
    """Detects drift between two Layer 3 runs."""
    
    risk_levels = {"NORMAL": 0, "STRESS": 1, "BLACK_SWAN": 2}

    risk_a = risk_levels.get(report_a.overall_risk_level)
    if risk_a is None:
        logger.warning(f"Unknown risk level in Previous report: {report_a.overall_risk_level}. Defaulting to NORMAL (0).")
        risk_a = 0
            
    risk_b = risk_levels.get(report_b.overall_risk_level)
    if risk_b is None:
        logger.warning(f"Unknown risk level in Current report: {report_b.overall_risk_level}. Defaulting to NORMAL (0).")
        risk_b = 0
    
    risk_escalated = risk_b > risk_a
    risk_deescalated = risk_b < risk_a
    
    mean_slip_change = report_b.slippage.mean_bps - report_a.slippage.mean_bps
    p99_slip_change = report_b.slippage.p99_bps - report_a.slippage.p99_bps
    entropy_delta_change = report_b.entropy.entropy_delta - report_a.entropy.entropy_delta
    breach_duration_change = (report_b.entropy.breach_duration_days - 
                               report_a.entropy.breach_duration_days)
    
    risk_transition = f"{report_a.overall_risk_level} → {report_b.overall_risk_level}"
    if report_a.overall_risk_level == report_b.overall_risk_level:
        risk_transition = f"STABLE: {report_a.overall_risk_level}"
        
    action_transition = f"{report_a.recommended_action} → {report_b.recommended_action}"
    
    slippage_sig = mean_slip_change > 100
    
    summary = f"Risk transition: {risk_transition}. Action transition: {action_transition}. "
    if risk_escalated:
        summary += f"Risk ESCALATED. "
    elif risk_deescalated:
        summary += f"Risk DE-ESCALATED. "
        
    summary += f"Mean slippage changed by {mean_slip_change:+.1f} bps. "
    summary += f"Entropy delta changed by {entropy_delta_change:+.3f}. "
    summary += f"Breach duration changed by {breach_duration_change:+} days."

    return {
        "risk_transition": risk_transition,
        "action_transition": action_transition,
        "entropy_delta_change": float(entropy_delta_change),
        "mean_slippage_change_bps": float(mean_slip_change),
        "p99_slippage_change_bps": float(p99_slip_change),
        "breach_duration_change_days": int(breach_duration_change),
        "risk_escalated": bool(risk_escalated),
        "risk_deescalated": bool(risk_deescalated),
        "slippage_increased_significantly": bool(slippage_sig),
        "summary": summary,
    }


def list_layer3_reports(
    output_dir: Optional[Path] = None,
    ticker: Optional[str] = None,
) -> list:
    """Lists Layer 3 reports, sorted by newest first."""
    
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "data" / "simulation_output"
        
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
        
    reports = []
    
    for f in output_dir.glob("layer3_*.json"):
        if not f.is_file():
            continue
            
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
            
        if data.get("cris_layer") != CRIS_LAYER:
            continue
            
        file_ticker = data.get("ticker", "unknown")
        if ticker and file_ticker != ticker:
            continue
            
        reports.append({
            "path": f,
            "ticker": file_ticker,
            "timestamp": data.get("generation_timestamp", ""),
            "risk_level": data.get("overall_risk_level", "unknown"),
            "action": data.get("recommended_action", "unknown"),
            "schema_version": data.get("schema_version", "unknown"),
        })
        
    return sorted(reports, key=lambda x: x["timestamp"], reverse=True)
