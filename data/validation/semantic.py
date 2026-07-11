"""Semantic validation rules for dataset families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class SemanticValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def validate_semantics(
    df: pd.DataFrame,
    rules: list[dict[str, Any]],
    date_column: str,
    expected_frequency: str,
    coverage_window: str,
    max_missing_ratio: float,
) -> SemanticValidationResult:
    """Validate range, plausibility, coverage, and missingness semantics."""

    errors: list[str] = []
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {
        "expected_frequency": expected_frequency,
        "coverage_window": coverage_window,
        "rules_checked": len(rules),
    }

    if df.empty:
        errors.append("dataset is empty")
        return SemanticValidationResult(False, errors, warnings, diagnostics)

    if date_column not in df.columns:
        errors.append(f"date column {date_column} is missing")
        return SemanticValidationResult(False, errors, warnings, diagnostics)

    missing_ratio = float(df.isna().sum().sum() / max(len(df) * max(len(df.columns), 1), 1))
    diagnostics["missingness_ratio"] = missing_ratio
    if missing_ratio > max_missing_ratio:
        errors.append(f"missingness ratio {missing_ratio:.4f} exceeds limit {max_missing_ratio:.4f}")

    if expected_frequency == "daily" and len(df) < 5:
        warnings.append("daily dataset has short history relative to expectation")

    for rule in rules:
        column = rule.get("column")
        if column not in df.columns:
            errors.append(f"semantic rule references missing column {column}")
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        if rule.get("required_non_null", True) and series.isna().any():
            errors.append(f"column {column} contains null values")
        if rule.get("reject_negative") and (series < 0).any():
            errors.append(f"column {column} contains negative values")

        min_value = rule.get("min_value")
        max_value = rule.get("max_value")
        if min_value is not None and (series < min_value).any():
            errors.append(f"column {column} fell below minimum {min_value}")
        if max_value is not None and (series > max_value).any():
            errors.append(f"column {column} exceeded maximum {max_value}")

    return SemanticValidationResult(len(errors) == 0, errors, warnings, diagnostics)
