from __future__ import annotations

import json
import logging
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import polars as pl

from data.cleaning.casting import cast_numeric_columns
from data.cleaning.datetime import parse_datetimes
from data.cleaning.duplicates import DuplicateHandlingResult, handle_duplicates
from data.cleaning.missing_values import handle_missing_values
from data.cleaning.outliers import OutlierHandlingResult, handle_outliers
from data.cleaning.schema import SchemaStandardizationResult, standardize_columns
from data.config import DATASET_REGISTRY, DEFAULT_DATE_COLUMN, METADATA_DIR, PROCESSED_DIR, QUARANTINE_DIR, STAGED_DIR
from data.contracts import DataPackage, DatasetMetadata, PublicationState, TransformationStep, ValidationReport
from data.errors import DataPublishError, DataSourceError, DatasetContractError
from data.loaders.csv_loader import load_csv_data
from data.loaders.fred import load_fred_data
from data.loaders.yahoo import load_yahoo_data
from data.validation.drift import detect_schema_drift

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp, Path)):
        return _json_default(value)
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_serialize(payload), handle, indent=4, default=_json_default)


def _write_frame(df: pd.DataFrame, path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        df.to_csv(path, index=False)
    elif output_format == "parquet":
        for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            if df[col].dt.tz is None:
                raise ValueError(f"CRIS Constitution Violation: Datetime column '{col}' lacks timezone. Must be explicitly UTC.")
            if df[col].dt.tz.zone != 'UTC':
                raise ValueError(f"CRIS Constitution Violation: Datetime column '{col}' is not UTC. (Found: {df[col].dt.tz.zone})")
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    else:
        raise DataPublishError(f"unsupported output format: {output_format}")


def _build_step(step_name: str, before: pd.DataFrame, after: pd.DataFrame, outcome: str, details: dict[str, Any] | None = None) -> TransformationStep:
    return TransformationStep(
        step_name=step_name,
        rows_before=len(before),
        rows_after=len(after),
        columns_before=list(before.columns),
        columns_after=list(after.columns),
        outcome=outcome,
        details=details or {},
    )


def generate_dq_report(dataset_name: str, report: ValidationReport) -> Path:
    """Write Data Quality Report to the metadata directory."""

    report_path = METADATA_DIR / f"{dataset_name}_dq_report.json"
    _write_json(report_path, report)
    logger.info("Data Quality Report saved to %s", report_path)
    return report_path


def _save_transformation_log(dataset_name: str, steps: list[TransformationStep]) -> Path:
    log_path = METADATA_DIR / f"{dataset_name}_transformation_log.json"
    _write_json(log_path, steps)
    return log_path


def _quarantine_dataset(dataset_name: str, raw_df: pd.DataFrame, staged_df: pd.DataFrame, metadata: DatasetMetadata, report: ValidationReport, reason: str, transformation_steps: list[TransformationStep]) -> Path:
    quarantine_dir = QUARANTINE_DIR / dataset_name / _utc_now().strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(raw_df, quarantine_dir / f"raw.{metadata.output_format}", metadata.output_format)
    _write_frame(staged_df, quarantine_dir / f"staged.{metadata.output_format}", metadata.output_format)
    _write_json(quarantine_dir / "metadata.json", metadata)
    _write_json(quarantine_dir / "validation_report.json", report)
    _write_json(quarantine_dir / "transformation_log.json", transformation_steps)
    _write_json(quarantine_dir / "quarantine_manifest.json", {"dataset_name": dataset_name, "reason": reason, "quarantined_at": _utc_now()})
    return quarantine_dir


def _publish_dataset(dataset_name: str, df: pd.DataFrame, output_format: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{dataset_name}.{output_format}"
    _write_frame(df, out_path, output_format)
    return out_path


def _load_raw_dataset(dataset_name: str):
    definition = DATASET_REGISTRY.get(dataset_name)
    provider_args = definition.provider_args
    if definition.provider == "yahoo":
        return load_yahoo_data(dataset_name, provider_args.get("tickers", []))
    if definition.provider == "fred":
        return load_fred_data(dataset_name, provider_args.get("series_ids", []))
    if definition.provider == "csv":
        csv_path = Path(provider_args.get("file_path"))
        return load_csv_data(dataset_name, csv_path, max_file_size_mb=definition.max_file_size_mb, chunksize_rows=provider_args.get("chunksize_rows"))
    raise DataSourceError(f"unknown provider: {definition.provider}")


def prepare_data(dataset_name: str) -> DataPackage:
    """End-to-end pipeline for loading, cleaning, validating, staging, and publishing a dataset."""

    definition = DATASET_REGISTRY.get(dataset_name)
    logger.info("Starting prepare_data pipeline for: %s", dataset_name)
    transformation_steps: list[TransformationStep] = []

    try:
        raw_dataset = _load_raw_dataset(dataset_name)
    except Exception as exc:
        metadata = DatasetMetadata(
            dataset_name=dataset_name,
            provider=definition.provider,
            version=definition.cleaning_version,
            timestamp_generated=_utc_now(),
            num_rows=0,
            num_columns=0,
            columns=[],
            schema_version=definition.schema_version,
            cleaning_version=definition.cleaning_version,
            provider_version=definition.provider_version,
            fetch_timestamp=None,
            source_url=definition.source_url(**definition.provider_args) if definition.provider_args else "",
            expected_columns=list(definition.expected_columns),
            missing_value_strategy=definition.missing_value_policy.strategy,
            duplicate_strategy=definition.duplicate_policy.timestamp_strategy,
            output_format=definition.output_format,
            publication_state=PublicationState.QUARANTINED,
            transformation_history=[],
            row_lineage_column="__row_id",
            extra={"source_failure": str(exc)},
        )
        report = ValidationReport(
            dataset_name=dataset_name,
            is_valid=False,
            schema_version=definition.schema_version,
            missing_values_count=0,
            duplicate_rows_count=0,
            duplicate_timestamps_count=0,
            is_chronological=False,
            schema_matched=False,
            expected_frequency=definition.expected_frequency,
            coverage_window=definition.coverage_window,
            missing_value_strategy=definition.missing_value_policy.strategy,
            duplicate_strategy=definition.duplicate_policy.timestamp_strategy,
            errors=[str(exc)],
        )
        quarantine_path = _quarantine_dataset(dataset_name, pd.DataFrame(), pd.DataFrame(), metadata, report, str(exc), transformation_steps)
        generate_dq_report(dataset_name, report)
        _save_transformation_log(dataset_name, transformation_steps)
        raise

    raw_df = raw_dataset.data.copy()
    if "__row_id" not in raw_df.columns:
        raw_df["__row_id"] = [f"{dataset_name}:{idx}" for idx in range(len(raw_df))]

    try:
        schema_report = detect_schema_drift(raw_df, list(definition.expected_columns), definition.schema_version, definition.source_metadata.get("previous_schema_version"))
        schema_result: SchemaStandardizationResult = standardize_columns(raw_df.copy(), list(definition.expected_columns), allow_extra_columns=definition.allow_extra_columns)
        staged_df = schema_result.dataframe.copy()
        transformation_steps.append(_build_step("schema_standardization", raw_df, staged_df, "ok", {"added_columns": schema_result.added_columns, "dropped_columns": schema_result.dropped_columns, "schema_drift": _serialize(schema_report)}))

        if definition.date_column not in staged_df.columns:
            raise DatasetContractError(f"missing required date column: {definition.date_column}")

        staged_df = parse_datetimes(staged_df, definition.date_column, definition.date_format)
        staged_df = staged_df.sort_values(by=definition.date_column).reset_index(drop=True)
        transformation_steps.append(_build_step("datetime_parsing", raw_df, staged_df, "ok", {"date_format": definition.date_format}))

        pre_missing_count = int(raw_df.isna().sum().sum())
        pre_missing_ratio = float(pre_missing_count / max(len(raw_df) * max(len(raw_df.columns), 1), 1))
        staged_df = handle_missing_values(staged_df, definition.missing_value_policy.strategy)
        transformation_steps.append(_build_step("missing_value_handling", raw_df, staged_df, "ok", {"strategy": definition.missing_value_policy.strategy, "pre_missing_count": pre_missing_count, "pre_missing_ratio": pre_missing_ratio}))

        duplicate_result: DuplicateHandlingResult = handle_duplicates(staged_df, definition.date_column, definition.duplicate_policy.timestamp_strategy)
        staged_df = duplicate_result.dataframe.copy()
        transformation_steps.append(_build_step("duplicate_removal", raw_df, staged_df, "ok", {"strategy": duplicate_result.strategy, "duplicate_rows_removed": duplicate_result.duplicate_rows_removed, "duplicate_timestamps_removed": duplicate_result.duplicate_timestamps_removed, "affected_row_indices": duplicate_result.affected_row_indices}))

        numeric_columns = [rule.column for rule in definition.semantic_rules if rule.column in staged_df.columns and rule.column != definition.date_column]
        staged_df = cast_numeric_columns(staged_df, numeric_columns)
        transformation_steps.append(_build_step("numeric_casting", raw_df, staged_df, "ok", {"numeric_columns": numeric_columns}))

        pl_staged_df = pl.from_pandas(staged_df)
        outlier_result: OutlierHandlingResult = handle_outliers(pl_staged_df, numeric_columns, method=definition.outlier_policy)
        staged_df = outlier_result.dataframe.to_pandas()
        transformation_steps.append(_build_step("outlier_handling", raw_df, staged_df, "ok", {"method": outlier_result.method, "diagnostics": outlier_result.diagnostics}))

        from data.validation.schemas import MarketDataSchema, MacroDataSchema
        import pandera as pa

        if "Open" in definition.expected_columns:
            schema_model = MarketDataSchema
        else:
            schema_model = MacroDataSchema
        
        errors = []
        warnings = []
        
        try:
            staged_df = schema_model.validate(staged_df)
            is_valid = True
        except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
            is_valid = False
            errors.append(f"Pandera Schema Violation: {str(e)}")
        except Exception as e:
            is_valid = False
            errors.append(f"Validation Error: {str(e)}")

        missing_count = int(staged_df.isna().sum().sum())
        dup_times = int(staged_df[definition.date_column].duplicated().sum()) if definition.date_column in staged_df.columns else 0
        is_chronological = True if definition.date_column in staged_df.columns and staged_df[definition.date_column].is_monotonic_increasing else False
        schema_match = True
        warnings.extend(schema_report.alerts)

        is_valid = len(errors) == 0
        report = ValidationReport(
            dataset_name=dataset_name,
            is_valid=is_valid,
            schema_version=definition.schema_version,
            missing_values_count=missing_count,
            duplicate_rows_count=duplicate_result.duplicate_rows_removed,
            duplicate_timestamps_count=dup_times,
            is_chronological=is_chronological,
            schema_matched=schema_match,
            expected_frequency=definition.expected_frequency,
            coverage_window=definition.coverage_window,
            missing_value_strategy=definition.missing_value_policy.strategy,
            duplicate_strategy=definition.duplicate_policy.timestamp_strategy,
            dropped_columns=schema_result.dropped_columns,
            extra_columns=schema_report.extra_columns,
            schema_drift_detected=not schema_report.schema_matched,
            semantic_errors=[],
            semantic_warnings=[],
            outlier_columns=outlier_result.diagnostics.get("columns", []) if outlier_result.diagnostics else [],
            outlier_diagnostics=outlier_result.diagnostics,
            missingness_ratio=pre_missing_ratio,
            duplicate_rows_affected=duplicate_result.affected_row_indices,
            errors=errors,
            warnings=warnings,
        )

        metadata = DatasetMetadata(
            dataset_name=dataset_name,
            provider=definition.provider,
            version=definition.cleaning_version,
            timestamp_generated=_utc_now(),
            num_rows=len(staged_df),
            num_columns=len(staged_df.columns),
            columns=list(staged_df.columns),
            schema_version=definition.schema_version,
            cleaning_version=definition.cleaning_version,
            provider_version=raw_dataset.provider_version,
            fetch_timestamp=raw_dataset.fetch_timestamp,
            source_url=raw_dataset.source_url if raw_dataset.source_url else (definition.source_url(**definition.provider_args) if definition.provider_args else ""),
            expected_columns=list(definition.expected_columns),
            missing_value_strategy=definition.missing_value_policy.strategy,
            duplicate_strategy=definition.duplicate_policy.timestamp_strategy,
            output_format=definition.output_format,
            publication_state=PublicationState.VALIDATED if is_valid else PublicationState.QUARANTINED,
            transformation_history=transformation_steps,
            row_lineage_column="__row_id",
            extra={
                "source_metadata": raw_dataset.source_metadata,
                "source_row_count": raw_dataset.source_row_count,
                "source_column_count": raw_dataset.source_column_count,
                "schema_drift": _serialize(schema_report),
                "semantic_diagnostics": {},
                "raw_missing_count": pre_missing_count,
                "raw_missing_ratio": pre_missing_ratio,
            },
        )

        # Stage the dataset before publication so the validation boundary is explicit.
        STAGED_DIR.mkdir(parents=True, exist_ok=True)
        staged_path = STAGED_DIR / f"{dataset_name}.{definition.output_format}"
        _write_frame(staged_df, staged_path, definition.output_format)

        generate_dq_report(dataset_name, report)
        transformation_log_path = _save_transformation_log(dataset_name, transformation_steps)

        if is_valid:
            published_path = _publish_dataset(dataset_name, staged_df, definition.output_format)
            metadata.publication_state = PublicationState.PUBLISHED
            _write_json(METADATA_DIR / f"{dataset_name}_metadata.json", metadata)
            logger.info("Successfully processed %s. Published to %s", dataset_name, published_path)
            return DataPackage(
                dataset_name=dataset_name,
                data=staged_df,
                metadata=metadata,
                validation_report=report,
                publication_state=PublicationState.PUBLISHED,
                published_path=str(published_path),
            )

        reason = "; ".join(errors) if errors else "validation failed"
        quarantine_path = _quarantine_dataset(dataset_name, raw_df, staged_df, metadata, report, reason, transformation_steps)
        metadata.publication_state = PublicationState.QUARANTINED
        metadata.extra["transformation_log_path"] = str(transformation_log_path)
        _write_json(METADATA_DIR / f"{dataset_name}_metadata.json", metadata)
        logger.warning("Dataset %s quarantined at %s", dataset_name, quarantine_path)
        return DataPackage(
            dataset_name=dataset_name,
            data=staged_df,
            metadata=metadata,
            validation_report=report,
            publication_state=PublicationState.QUARANTINED,
            quarantine_path=str(quarantine_path),
            quarantine_reason=reason,
        )

    except Exception as exc:
        quarantine_dir = QUARANTINE_DIR / dataset_name / _utc_now().strftime("%Y%m%dT%H%M%SZ")
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        _write_json(quarantine_dir / "error.json", {"dataset_name": dataset_name, "error": str(exc), "failed_at": _utc_now()})
        logger.exception("Failed to prepare dataset %s; details quarantined to %s", dataset_name, quarantine_dir)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_data("market_macro")
