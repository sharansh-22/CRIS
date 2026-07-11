# CRIS v2 Data Layer

## Purpose

The Data Layer is the single source of truth for all external data entering the Cascade Risk Intelligence System (CRIS). Its exclusive responsibility is to ingest raw data and process it into clean, standardized, trustworthy datasets.

Every CRIS module that consumes data (e.g., Signal Harvesters) must rely entirely on the outputs from `data/processed/` or `data/metadata/`. Downstream modules must **never** perform data cleaning or imputation.

## Responsibilities

The Data Layer performs exactly these steps, strictly in this order:
1. **Load:** Ingest data from external providers (FRED, Yahoo Finance, static CSVs).
2. **Schema Validation:** Ensure raw data matches expected configurations.
3. **Datetime Parsing:** Standardize to ISO 8601 UTC.
4. **Missing Value Handling:** Apply dataset-specific imputation strategies.
5. **Duplicate Removal:** Remove duplicate rows and duplicate timestamps.
6. **Column Standardization:** Ensure consistent column naming.
7. **Data Type Conversion:** Standardize column datatypes.
8. **Integrity Checks:** Validate chronological ordering and verify no NaNs exist post-cleaning.
9. **Export:** Save cleaned data and generate Data Quality Reports.

## What It Intentionally DOES NOT Do

- **Compute Signals:** It produces clean prices and metrics, not momentum or volatility signals.
- **Analyze Markets:** It does not detect macro regimes or systemic fragmentation.
- **Rank Signals:** It has no attribution or relevance logic.
- **Make Decisions:** It is entirely agnostic to downstream systems (Credit, Portfolio, ESG).

## Folder Structure

- `cache/`: Temporary storage for raw downloads.
- `cleaning/`: Modular functions for missing values, duplicates, datetimes, schema, and outliers.
- `contracts.py`: Strict dataclasses defining the expected shapes of `RawDataset`, `DataPackage`, `DatasetMetadata`, and `ValidationReport`.
- `config.py`: Global configuration defining input/output paths, format standards, and dataset-specific cleaning strategies.
- `loaders/`: Provider-specific data fetchers (`fred.py`, `yahoo.py`, `csv.py`).
- `metadata/`: Versioned descriptions, lineage, and validation reports for every dataset.
- `pipeline/`: The `prepare_data.py` orchestrator that ties the Data Layer together.
- `processed/`: The final, production-ready outputs.
- `raw/`: Untouched original data (when provided locally).
- `validation/`: Quality and integrity assertion logic.

## Guarantees

Every dataset leaving the Data Layer via the `DataPackage` contract guarantees:
✓ No missing values (NaNs).
✓ No duplicate timestamps.
✓ Consistent, monotonic chronological ordering.
✓ Standardized ISO 8601 UTC timestamps.
✓ Schema validation passed.
✓ Automatic Data Quality Report generated.
