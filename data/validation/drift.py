"""Schema drift detection for the CRIS Data Layer."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class SchemaDriftReport:
    schema_matched: bool
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    version_changed: bool = False
    alerts: list[str] = field(default_factory=list)


def detect_schema_drift(
    df: pd.DataFrame,
    expected_columns: list[str],
    schema_version: str,
    previous_schema_version: str | None = None,
) -> SchemaDriftReport:
    missing = [column for column in expected_columns if column not in df.columns]
    extra = [column for column in df.columns if column not in expected_columns and not column.startswith("__")]
    alerts: list[str] = []
    if missing:
        alerts.append(f"missing columns: {missing}")
    if extra:
        alerts.append(f"extra columns: {extra}")
    version_changed = previous_schema_version is not None and previous_schema_version != schema_version
    if version_changed:
        alerts.append(f"schema version changed from {previous_schema_version} to {schema_version}")
    return SchemaDriftReport(
        schema_matched=len(missing) == 0 and len(extra) == 0,
        missing_columns=missing,
        extra_columns=extra,
        version_changed=version_changed,
        alerts=alerts,
    )
