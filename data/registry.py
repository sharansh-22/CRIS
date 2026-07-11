"""Dataset registry for the CRIS Data Layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MissingValuePolicy:
    strategy: str
    max_missing_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class DuplicatePolicy:
    row_strategy: str = "drop"
    timestamp_strategy: str = "drop_first"
    fail_if_missing_policy: bool = True


@dataclass(frozen=True, slots=True)
class SemanticRule:
    column: str
    min_value: float | None = None
    max_value: float | None = None
    required_non_null: bool = True
    reject_negative: bool = False


@dataclass(frozen=True, slots=True)
class DatasetDefinition:
    name: str
    provider: str
    expected_columns: tuple[str, ...]
    provider_args: dict[str, Any] = field(default_factory=dict)
    date_column: str = "Date"
    date_format: str = "%Y-%m-%d"
    output_format: str = "parquet"
    schema_version: str = "1.0.0"
    cleaning_version: str = "1.0.0"
    provider_version: str = "unknown"
    source_version: str = "unknown"
    allow_extra_columns: bool = False
    missing_value_policy: MissingValuePolicy = field(default_factory=lambda: MissingValuePolicy("forward_fill"))
    duplicate_policy: DuplicatePolicy = field(default_factory=DuplicatePolicy)
    semantic_rules: tuple[SemanticRule, ...] = field(default_factory=tuple)
    expected_frequency: str = "daily"
    coverage_window: str = "unspecified"
    allowed_gap_policy: str = "strict"
    outlier_policy: str = "clip"
    max_rows: int | None = None
    max_file_size_mb: int | None = None
    source_url_template: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def source_url(self, **kwargs: Any) -> str:
        if not self.source_url_template:
            return ""
        return self.source_url_template.format(**kwargs)

    def to_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_args": dict(self.provider_args),
            "expected_columns": list(self.expected_columns),
            "date_column": self.date_column,
            "date_format": self.date_format,
            "output_format": self.output_format,
            "schema_version": self.schema_version,
            "cleaning_version": self.cleaning_version,
            "provider_version": self.provider_version,
            "source_version": self.source_version,
            "allow_extra_columns": self.allow_extra_columns,
            "missing_value_strategy": self.missing_value_policy.strategy,
            "max_missing_ratio": self.missing_value_policy.max_missing_ratio,
            "duplicate_strategy": self.duplicate_policy.timestamp_strategy,
            "row_duplicate_strategy": self.duplicate_policy.row_strategy,
            "semantic_rules": [asdict(rule) for rule in self.semantic_rules],
            "expected_frequency": self.expected_frequency,
            "coverage_window": self.coverage_window,
            "allowed_gap_policy": self.allowed_gap_policy,
            "outlier_policy": self.outlier_policy,
            "max_rows": self.max_rows,
            "max_file_size_mb": self.max_file_size_mb,
            "source_url_template": self.source_url_template,
            "source_metadata": dict(self.source_metadata),
        }


class DatasetRegistry:
    """Authoritative registry for dataset contracts and publication policy."""

    def __init__(self, definitions: dict[str, DatasetDefinition]) -> None:
        self._definitions = dict(definitions)

    def get(self, name: str) -> DatasetDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"unknown dataset definition: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions.keys())

    def as_config_map(self) -> dict[str, dict[str, Any]]:
        return {name: definition.to_config() for name, definition in self._definitions.items()}

    def source_path(self, *parts: str) -> Path:
        from data.config import DATA_DIR

        return DATA_DIR.joinpath(*parts)
