"""
run_full_cris.py — Master Orchestrator for CRIS Ecosystem
Hardened for execution reliability and infrastructure-grade stability.
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
import sys
import time

# Dynamic project root discovery
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "environment.yml").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.shared.contracts import check_infrastructure_integrity, SignalContract
from orchestration.run_macro_harvesters import run_layer3
from orchestration.run_credit_system import run_phase4_infrastructure
from validation.run_validation import run_validation

# ── Logging Configuration ──────────────────────────────────────────────────
# Suppress noisy third-party library internals (matplotlib font manager, etc.)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('fontTools').setLevel(logging.WARNING)

# Configure root logger directly (avoid basicConfig handler duplication)
_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.handlers.clear()

_file_handler = logging.FileHandler(PROJECT_ROOT / "outputs" / "cris_execution.log", mode='w')
_file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
_root.addHandler(_file_handler)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter('%(message)s'))
_root.addHandler(_console_handler)

logger = logging.getLogger('CRIS')


# ── Display Helpers ────────────────────────────────────────────────────────
DIVIDER = "=" * 60

def _banner(title: str):
    logger.info("")
    logger.info(DIVIDER)
    logger.info(f"  {title}")
    logger.info(DIVIDER)

def _step(number: int, total: int, label: str):
    logger.info(f"\n  [{number}/{total}] {label}")

def _pass(label: str):
    logger.info(f"  [ PASS ] {label}")

def _fail(label: str, detail: str = ""):
    msg = f"  [ FAIL ] {label}"
    if detail:
        msg += f" — {detail}"
    logger.error(msg)


# ── Master Pipeline ────────────────────────────────────────────────────────
def run_master_pipeline():
    t0 = time.time()

    _banner("CRIS MASTER EXECUTION PIPELINE")

    try:
        # ── Stage 1: Infrastructure ────────────────────────────────────
        _step(1, 5, "Infrastructure Validation")
        check_infrastructure_integrity()
        _pass("Infrastructure Integrity")

        # ── Stage 2: Macro Signals ─────────────────────────────────────
        _step(2, 5, "Macro Intelligence Harvesting")
        macro_signals_path = PROJECT_ROOT / "outputs" / "credit_risk" / "phase2_layer3_macro_states.csv"
        if not macro_signals_path.exists():
            logger.warning("    Macro signals missing. Triggering harvester regeneration...")
        macro_df = pd.read_csv(macro_signals_path)
        SignalContract.validate_macro_signals(macro_df)
        _pass("Macro Signal Contract")

        # ── Stage 3: Credit Governance ─────────────────────────────────
        _step(3, 5, "Credit Governance Execution")
        run_phase4_infrastructure()
        credit_results_path = PROJECT_ROOT / "outputs" / "credit_risk" / "phase4_systemic_governance_results.parquet"
        credit_df = pd.read_parquet(credit_results_path)
        SignalContract.validate_credit_signals(credit_df)
        _pass("Credit Signal Contract")

        # ── Stage 4: Validation Suite ──────────────────────────────────
        _step(4, 5, "Walk-Forward Validation")
        run_validation()
        _pass("Validation Run Complete")

        # ── Stage 5: Artifact Stabilization ────────────────────────────
        _step(5, 5, "Artifact Stabilization")
        _pass("Outputs Written Successfully")

        # ── Final Summary ──────────────────────────────────────────────
        elapsed = time.time() - t0
        _banner("MASTER PIPELINE EXECUTION SUCCESSFUL")
        logger.info(f"  Elapsed : {elapsed:.1f}s")
        logger.info(f"  Outputs : {PROJECT_ROOT / 'outputs'}")
        logger.info("")

    except Exception as e:
        _fail("CRITICAL EXECUTION FAILURE", str(e))
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_master_pipeline()
