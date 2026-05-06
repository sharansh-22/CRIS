"""
institutional_audit.py — Master Institutional Audit Coordinator

This script serves as the centralized entry point for the CRIS Layer 3 
Adversarial Audit Suite. It coordinates multiple specialized audits 
to falsify architectural assumptions and validate probabilistic stability.

Audits included:
1. Cross-Asset Calibration
2. Trajectory Integrity & Trend-Bias
3. Convergence Stability
4. Long-Duration Behavioral Stability
5. LSTM Advisory Generalization
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AUDIT_DIR = Path(__file__).resolve().parent

AUDIT_SUITES = {
    "calibration": "cross_asset_calibration_audit.py",
    "trajectory": "trajectory_integrity_audit.py",
    "convergence": "convergence_stability_audit.py",
    "stability": "long_duration_stability_audit.py",
    "lstm": "lstm_generalization_audit.py"
}

def run_suite(name: str):
    script = AUDIT_SUITES.get(name)
    if not script:
        print(f"Error: Audit suite '{name}' not found.")
        return False
    
    print(f"\n{'='*60}")
    print(f" EXECUTING AUDIT: {name.upper()}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run([sys.executable, str(AUDIT_DIR / script)], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Audit suite '{name}' failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="CRIS Layer 3 Institutional Audit Coordinator")
    parser.add_argument("--suite", type=str, choices=list(AUDIT_SUITES.keys()) + ["all"], default="all",
                        help="Specialized audit suite to execute (default: all)")
    
    args = parser.parse_args()
    
    success_count = 0
    suites_to_run = list(AUDIT_SUITES.keys()) if args.suite == "all" else [args.suite]
    
    for suite in suites_to_run:
        if run_suite(suite):
            success_count += 1
            
    print(f"\n{'='*60}")
    print(f" AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f" Total Suites: {len(suites_to_run)}")
    print(f" Passed:       {success_count}")
    print(f" Failed:       {len(suites_to_run) - success_count}")
    print(f"{'='*60}\n")
    
    sys.exit(0 if success_count == len(suites_to_run) else 1)

if __name__ == "__main__":
    main()
