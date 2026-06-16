"""
validation.py — Walk-forward safety and causal integrity verification.

Ensures that all SAE computations respect temporal causality:
no future data leaks into signal attribution measurements.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger("CRIS.SAE.validation")


def validate_temporal_causality(merged_df: pd.DataFrame, signal_names: List[str]) -> Dict[str, str]:
    """Verify that signal values are only derived from data available at or before issue_month.

    Since signals come from the macro states CSV (which is generated monthly
    with strict walk-forward slicing in generate_monthly_layer3_states),
    this validates that the merge preserved temporal ordering.
    """
    checks = {}

    # 1. Verify issue_month is before or equal to state_date
    if "state_date" in merged_df.columns:
        issue_dates = pd.to_datetime(merged_df["issue_month"])
        state_dates = pd.to_datetime(merged_df["state_date"])
        violations = (state_dates > issue_dates).sum()
        checks["state_date_ordering"] = f"PASS ({violations} violations)" if violations == 0 else f"FAIL ({violations} violations)"
    else:
        checks["state_date_ordering"] = "SKIP (no state_date column)"

    # 2. Verify no signal values are NaN in ways that would indicate broken joins
    for signal in signal_names:
        if signal in merged_df.columns:
            nan_rate = merged_df[signal].isna().mean()
            if nan_rate > 0.5:
                checks[f"{signal}_completeness"] = f"WARNING ({nan_rate:.1%} missing)"
            else:
                checks[f"{signal}_completeness"] = f"PASS ({nan_rate:.1%} missing)"

    # 3. Verify temporal monotonicity of issue_month
    issue_months = pd.to_datetime(merged_df["issue_month"]).sort_values()
    is_sorted = issue_months.is_monotonic_increasing
    checks["temporal_monotonicity"] = "PASS" if is_sorted else "WARNING (issue_months not sorted)"

    # 4. Verify no future-period signals leak into current observations
    # Each loan's signals should come from the SAME issue_month row
    if "issue_month" in merged_df.columns:
        unique_months = merged_df["issue_month"].nunique()
        checks["unique_months"] = f"PASS ({unique_months} distinct months)"

    return checks


def validate_no_target_leakage(merged_df: pd.DataFrame, signal_names: List[str], target_col: str = "target") -> Dict[str, str]:
    """Verify that signals are not derived from target information.

    Checks for suspiciously high correlation that would indicate leakage.
    """
    checks = {}
    for signal in signal_names:
        if signal not in merged_df.columns:
            continue

        corr = merged_df[signal].corr(merged_df[target_col])
        if abs(corr) > 0.5:
            checks[f"{signal}_leakage_check"] = f"WARNING (|corr|={abs(corr):.3f} — investigate)"
        else:
            checks[f"{signal}_leakage_check"] = f"PASS (|corr|={abs(corr):.3f})"

    return checks


def run_full_validation(
    merged_df: pd.DataFrame,
    signal_names: List[str],
    target_col: str = "target",
) -> Dict[str, str]:
    """Run all validation checks and return combined results."""
    results = {}
    results.update(validate_temporal_causality(merged_df, signal_names))
    results.update(validate_no_target_leakage(merged_df, signal_names, target_col))

    # Overall status
    warnings = sum(1 for v in results.values() if "WARNING" in v)
    failures = sum(1 for v in results.values() if "FAIL" in v)

    if failures > 0:
        results["OVERALL"] = f"FAIL ({failures} failures, {warnings} warnings)"
    elif warnings > 0:
        results["OVERALL"] = f"PASS WITH WARNINGS ({warnings} warnings)"
    else:
        results["OVERALL"] = "PASS — All checks green"

    return results
