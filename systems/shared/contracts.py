"""
contracts.py — Formal Signal Contracts for CRIS Ecosystem
Ensures data integrity and schema stability across layers.
"""

from typing import Dict, Any, List
import pandas as pd
import logging

logger = logging.getLogger('CRIS.contracts')

class SignalContract:
    @staticmethod
    def validate_macro_signals(df: pd.DataFrame):
        """Verify that macro harvester outputs contain required fields."""
        required = ['macro_stress_score', 'uncertainty_pressure', 'dominant_field']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Macro Signal Contract Violation: Missing fields {missing}")
        # Validation passed silently — master orchestrator reports PASS

    @staticmethod
    def validate_credit_signals(df: pd.DataFrame):
        """Verify that credit system outputs contain required fields."""
        required = ['pd_borrower', 'pd_macro', 'gov_state', 'routing_decision']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Credit Signal Contract Violation: Missing fields {missing}")
        # Validation passed silently — master orchestrator reports PASS

def check_infrastructure_integrity():
    """Verify that all core directories and config files exist."""
    from pathlib import Path
    import sys
    
    # Dynamic root
    root = Path(__file__).resolve().parent.parent.parent
    
    required_dirs = [
        root / "data" / "macro",
        root / "data" / "credit_risk",
        root / "outputs" / "macro",
        root / "outputs" / "credit_risk",
        root / "configs",
        root / "orchestration",
        root / "systems"
    ]
    
    missing = [str(d) for d in required_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(f"Infrastructure Integrity Violation: Missing directories {missing}")
    
    # Passed — master orchestrator reports PASS
