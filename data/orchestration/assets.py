import logging
from dataclasses import dataclass, field
import pandas as pd
import polars as pl
import pandera as pa
from dagster import asset, Output

from data.config import DATASET_REGISTRY
from data.contracts import DataPackage, DatasetMetadata, PublicationState, ValidationReport
from data.pipeline.prepare_data import (
    _load_raw_dataset,
    _quarantine_dataset,
    _publish_dataset,
    _utc_now,
    _build_step,
    _serialize,
    generate_dq_report,
    _save_transformation_log
)
from data.validation.drift import detect_schema_drift
from data.cleaning.schema import standardize_columns
from data.cleaning.datetime import parse_datetimes
from data.cleaning.missing_values import handle_missing_values
from data.cleaning.duplicates import handle_duplicates
from data.cleaning.casting import cast_numeric_columns
from data.cleaning.outliers import handle_outliers
from data.validation.schemas import MarketDataSchema, MacroDataSchema
from data.errors import DatasetContractError

logger = logging.getLogger(__name__)

# Hardcoding market_macro to fulfill "at least two connected Dagster assets" demonstration
DATASET_NAME = "market_macro"

@asset
def raw_market_macro() -> DataPackage:
    """Fetches raw market/macro data."""
    logger.info("Fetching raw dataset: %s", DATASET_NAME)
    # The existing _load_raw_dataset returns a DataPackage
    return _load_raw_dataset(DATASET_NAME)

@asset
def validated_market_macro(raw_market_macro: DataPackage) -> Output[DataPackage]:
    """Validates, cleans, and stages the raw market/macro data."""
    dataset_name = DATASET_NAME
    definition = DATASET_REGISTRY.get(dataset_name)
    transformation_steps = []
    
    raw_dataset = raw_market_macro
    raw_df = raw_dataset.data.copy()
    if "__row_id" not in raw_df.columns:
        raw_df["__row_id"] = [f"{dataset_name}:{idx}" for idx in range(len(raw_df))]

    # 1. Schema Standardization
    schema_report = detect_schema_drift(raw_df, list(definition.expected_columns), definition.schema_version, definition.source_metadata.get("previous_schema_version"))
    schema_result = standardize_columns(raw_df.copy(), list(definition.expected_columns), allow_extra_columns=definition.allow_extra_columns)
    staged_df = schema_result.dataframe.copy()
    transformation_steps.append(_build_step("schema_standardization", raw_df, staged_df, "ok", {"added_columns": schema_result.added_columns, "dropped_columns": schema_result.dropped_columns, "schema_drift": _serialize(schema_report)}))

    # 2. Datetime Parsing
    if definition.date_column not in staged_df.columns:
        raise DatasetContractError(f"missing required date column: {definition.date_column}")
    staged_df = parse_datetimes(staged_df, definition.date_column, definition.date_format)
    staged_df = staged_df.sort_values(by=definition.date_column).reset_index(drop=True)
    transformation_steps.append(_build_step("datetime_parsing", raw_df, staged_df, "ok", {"date_format": definition.date_format}))

    # 3. Missing Values
    pre_missing_count = int(raw_df.isna().sum().sum())
    pre_missing_ratio = float(pre_missing_count / max(len(raw_df) * max(len(raw_df.columns), 1), 1))
    staged_df = handle_missing_values(staged_df, definition.missing_value_policy.strategy)
    transformation_steps.append(_build_step("missing_value_handling", raw_df, staged_df, "ok", {"strategy": definition.missing_value_policy.strategy, "pre_missing_count": pre_missing_count, "pre_missing_ratio": pre_missing_ratio}))

    # 4. Duplicate Handling
    duplicate_result = handle_duplicates(staged_df, definition.date_column, definition.duplicate_policy.timestamp_strategy)
    staged_df = duplicate_result.dataframe.copy()
    transformation_steps.append(_build_step("duplicate_removal", raw_df, staged_df, "ok", {"strategy": duplicate_result.strategy, "duplicate_rows_removed": duplicate_result.duplicate_rows_removed, "duplicate_timestamps_removed": duplicate_result.duplicate_timestamps_removed, "affected_row_indices": duplicate_result.affected_row_indices}))

    # 5. Numeric Casting
    numeric_columns = [rule.column for rule in definition.semantic_rules if rule.column in staged_df.columns and rule.column != definition.date_column]
    staged_df = cast_numeric_columns(staged_df, numeric_columns)
    transformation_steps.append(_build_step("numeric_casting", raw_df, staged_df, "ok", {"numeric_columns": numeric_columns}))

    # 6. Polars Outlier Handling (Phase 3 Integration)
    pl_staged_df = pl.from_pandas(staged_df)
    outlier_result = handle_outliers(pl_staged_df, numeric_columns, method=definition.outlier_policy)
    staged_df = outlier_result.dataframe.to_pandas()
    transformation_steps.append(_build_step("outlier_handling", raw_df, staged_df, "ok", {"method": outlier_result.method, "diagnostics": outlier_result.diagnostics}))

    # 7. Pandera Validation
    if "Open" in definition.expected_columns:
        schema_model = MarketDataSchema
    else:
        schema_model = MacroDataSchema
    
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
        publication_state=PublicationState.VALIDATED,
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

    try:
        staged_df = schema_model.validate(staged_df)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
        logger.error(f"Pandera Validation Failed: {e}")
        # Build report for quarantine
        report = ValidationReport(
            dataset_name=dataset_name,
            is_valid=False,
            schema_version=definition.schema_version,
            missing_values_count=int(staged_df.isna().sum().sum()),
            duplicate_rows_count=duplicate_result.duplicate_rows_removed,
            duplicate_timestamps_count=int(staged_df[definition.date_column].duplicated().sum()) if definition.date_column in staged_df.columns else 0,
            is_chronological=False,
            schema_matched=False,
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
            errors=[f"Pandera Schema Violation: {str(e)}"],
            warnings=schema_report.alerts,
        )
        metadata.publication_state = PublicationState.QUARANTINED
        _quarantine_dataset(dataset_name, raw_df, staged_df, metadata, report, f"Pandera Schema Violation: {str(e)}", transformation_steps)
        # Raise exception to ensure the Dagster asset fails and blocks downstream
        raise e
        
    missing_count = int(staged_df.isna().sum().sum())
    dup_times = int(staged_df[definition.date_column].duplicated().sum()) if definition.date_column in staged_df.columns else 0
    is_chronological = True if definition.date_column in staged_df.columns and staged_df[definition.date_column].is_monotonic_increasing else False
    
    report = ValidationReport(
        dataset_name=dataset_name,
        is_valid=True,
        schema_version=definition.schema_version,
        missing_values_count=missing_count,
        duplicate_rows_count=duplicate_result.duplicate_rows_removed,
        duplicate_timestamps_count=dup_times,
        is_chronological=is_chronological,
        schema_matched=True,
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
        errors=[],
        warnings=schema_report.alerts,
    )

    metadata.publication_state = PublicationState.PUBLISHED
    
    generate_dq_report(dataset_name, report)
    _save_transformation_log(dataset_name, transformation_steps)
    
    # Save the PyArrow Parquet File using our existing writer function
    published_path = _publish_dataset(dataset_name, staged_df, definition.output_format)
    
    pkg = DataPackage(
        dataset_name=dataset_name,
        data=staged_df,
        metadata=metadata,
        validation_report=report,
        publication_state=PublicationState.PUBLISHED,
        published_path=str(published_path),
    )
    
    return Output(pkg, metadata={"path": str(published_path), "rows": len(staged_df)})
