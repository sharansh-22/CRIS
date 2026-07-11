# CRIS Data Layer Walkthrough

This document explains what the CRIS Data Layer does right now, how it is structured, and where the current implementation is still incomplete.

The Data Layer is the front door of CRIS. Its job is to ingest external data, clean it into a standardized form, validate the result, and write production-ready datasets for downstream layers.

It is not a signal engine. It does not analyze markets, infer regimes, rank information, or make decisions. Its only responsibility is to produce reliable cleaned datasets and the associated metadata and validation reports.

## 1. What The Data Layer Is For

The Data Layer exists to solve one problem: turn messy external inputs into datasets that the rest of CRIS can trust.

The intended consumer is not a model or a decision system. The intended consumer is the rest of the CRIS pipeline, especially signal extraction and anything downstream that assumes the data is already clean.

In practical terms, the layer should:

- fetch raw data from supported providers
- standardize dates and column names
- handle missing values according to dataset-specific rules
- remove duplicates and duplicate timestamps
- check chronological order and schema consistency
- write cleaned outputs to `data/processed/`
- write validation artifacts to `data/metadata/`

## 2. Where The Data Layer Lives

The data subsystem is organized under `data/` and currently contains:

- `config.py`: global paths, formats, and dataset configuration
- `contracts.py`: dataclasses describing raw input, cleaned output, metadata, and validation results
- `loaders/`: provider-specific ingestion functions
- `cleaning/`: cleaning helpers for schema, datetime parsing, missing values, duplicates, and outliers
- `validation/`: integrity and schema checks
- `pipeline/prepare_data.py`: the end-to-end orchestrator
- `raw/`, `cache/`, `processed/`, `metadata/`: storage locations used by the pipeline

## 3. The Current Execution Flow

The main orchestrator is `data/pipeline/prepare_data.py`. The current flow is straightforward and linear.

### Step 1. Choose a dataset configuration

The pipeline reads from `DATASET_CONFIGS` in `data/config.py`.

Each configuration currently specifies:

- provider name
- tickers or series identifiers
- missing value strategy
- duplicate handling strategy
- expected columns
- output format

At the moment, the predefined datasets are all market-oriented:

- `market_indices`
- `market_equities`
- `market_macro`
- `market_structure_etfs`

That means the current implementation is focused on financial market datasets rather than a broad general-purpose ingestion system.

### Step 2. Load raw data

The pipeline selects a loader based on the configured provider.

Current providers:

- `yahoo` via `load_yahoo_data`
- `fred` via `load_fred_data`
- `csv` via `load_csv_data`

What each loader currently does:

#### Yahoo loader

`data/loaders/yahoo.py` downloads each configured ticker with `yfinance`, concatenates the results, and returns a `RawDataset`.

Important current behavior:

- each ticker is downloaded individually
- empty ticker responses are skipped with a warning
- the result is concatenated into one dataframe
- a `Ticker` column is added
- the `Date` column is converted to string before later datetime parsing

This loader is the most complete part of the ingestion layer.

#### FRED loader

`data/loaders/fred.py` is currently a placeholder.

It logs that FRED loading is not fully implemented and returns an empty dataframe.

This is a major limitation: the configuration surface claims support for FRED-style series ingestion, but the current implementation does not actually fetch or process those series.

#### CSV loader

`data/loaders/csv_loader.py` reads a CSV file from disk and wraps it in a `RawDataset`.

If the read fails, it logs the error and returns an empty dataframe.

## 4. Cleaning Pipeline

After loading, `prepare_data.py` applies a fixed sequence of cleaning steps.

### 4.1 Schema standardization

`data/cleaning/schema.py` checks for missing expected columns and fills them with `pd.NA`.

It then reorders the dataframe to match the expected column order.

Current behavior:

- missing expected columns are added
- extra columns are dropped implicitly when the dataframe is reindexed to the expected set
- the output schema is forced into the configured order

This is useful for standardization, but it is also a source of hidden data loss if the upstream source contains useful extra columns that were not declared.

### 4.2 Datetime parsing

`data/cleaning/datetime.py` parses the configured date column using `pd.to_datetime(..., utc=True)`.

Current behavior:

- timestamps are normalized to UTC
- parsing failures are logged
- the dataframe is returned even if parsing fails

The pipeline then sorts the dataframe by the date column if it exists and the dataframe is not empty.

### 4.3 Missing value handling

`data/cleaning/missing_values.py` supports several strategies:

- `forward_fill`
- `drop`
- `mean_fill`
- `none`

Current behavior:

- `forward_fill` applies `ffill()`
- `drop` removes rows with missing values
- `mean_fill` fills numeric columns with the column mean
- unknown strategies are logged and leave NaNs intact

This is an important distinction: the documentation claims dataset-specific strategies, and the code does implement several of them, but it does not validate that a strategy is appropriate for the specific data being cleaned.

### 4.4 Duplicate removal

`data/cleaning/duplicates.py` removes:

- exact duplicate rows
- duplicate timestamps on the date column

The timestamp duplicate strategy is controlled by the dataset config:

- `drop_first`
- `drop_last`
- anything else falls back to dropping all duplicates with `keep=False` semantics on the date subset

The function logs how many rows were removed.

### 4.5 Outlier handling

`data/cleaning/outliers.py` exists, but it is currently a stub.

It returns the original dataframe unless a non-`none` method is passed, and even then it logs that outlier handling is not fully implemented.

This means the Data Layer does not yet perform meaningful outlier treatment, despite the presence of the module.

## 5. Validation And Integrity Checks

After cleaning, the pipeline runs a set of quality checks.

### Schema matching

`data/validation/quality_checks.py` checks whether the final dataframe column list exactly matches the configured expected columns.

This is strict and order-sensitive.

### Chronological order

`data/validation/integrity.py` checks whether the date column is monotonic increasing.

### Duplicate timestamps

The same integrity module counts duplicate timestamps in the date column.

### Missing values

The integrity module also counts total missing values in the dataframe.

### Overall validity

The pipeline combines the checks into a boolean `is_valid` flag.

The current validity rule is:

- chronological order must be true
- duplicate timestamps must be zero
- missing values must be zero
- schema must match

If any of these fail, the dataset is still returned, but the validation report records the errors.

This is a key point: the pipeline does not necessarily stop on invalid data. It produces a report and continues to export the cleaned output.

## 6. Contracts And Returned Artifacts

The Data Layer returns a `DataPackage` defined in `data/contracts.py`.

The package contains:

- `dataset_name`
- `data`: the cleaned dataframe
- `metadata`: dataset metadata
- `validation_report`: validation results

### Metadata contents

`DatasetMetadata` currently records:

- dataset name
- provider
- version
- generation timestamp
- row count
- column count
- column names
- missing value strategy
- an `extra` dictionary for arbitrary additions

### Validation report contents

`ValidationReport` currently records:

- whether the dataset is valid
- missing value count
- duplicate row count
- duplicate timestamp count
- chronological order status
- schema match status
- errors
- warnings

### Data quality report export

The pipeline writes a JSON report to `data/metadata/{dataset_name}_dq_report.json`.

This is one of the most useful parts of the current implementation because it gives the Data Layer an auditable trace of what happened during cleaning and validation.

## 7. Export Behavior

After validation, the pipeline writes the cleaned dataset to `data/processed/`.

Current supported formats:

- CSV
- Parquet

The filename is based on the dataset name and configured output format.

This makes the Data Layer a concrete artifact-producing stage, not just an in-memory utility.

## 8. What The Data Layer Intentionally Does Not Do

The current design correctly avoids downstream responsibilities.

It does not:

- compute signals
- infer hidden states
- classify borrowers
- rank observations
- optimize portfolios
- make predictions
- make decisions for later systems

This boundary is important. The Data Layer should only deliver standardized, validated input for later layers.

## 9. What Is Implemented Well Right Now

Several parts of the Data Layer are already useful and coherent.

### Strengths

- the pipeline is easy to understand
- the contract objects are explicit
- the output is persisted to stable locations
- the cleaning order is deterministic
- the data quality report is generated automatically
- the code keeps provider-specific ingestion separate from cleaning and validation

These are good signs for maintainability.

## 10. Current Gaps And Limitations

This section is important because it describes what the Data Layer does not fully do yet.

### 10.1 FRED ingestion is not implemented

The loader exists only as a placeholder.

Impact:

- the config surface overstates provider support
- macro ingestion paths are incomplete

### 10.2 Outlier handling is not implemented

The module exists, but the logic is a stub.

Impact:

- no robust extreme-value treatment
- the data layer cannot yet claim full cleaning coverage

### 10.3 Validation is strong on shape, weaker on semantics

The pipeline validates schema and chronology, but it does not deeply validate the meaning of the data beyond the configured columns and timestamps.

Impact:

- wrong but structurally valid data can still pass
- integrity depends heavily on upstream provider correctness

### 10.4 Invalid datasets are still exported

The pipeline writes output even when the validation report says the dataset is invalid.

Impact:

- downstream users must inspect the report manually
- bad data can still reach storage if pipeline consumers are careless

### 10.5 No explicit dataset registry abstraction exists

Dataset config is centralized in a dict inside `config.py`.

Impact:

- simple for now
- less scalable for large multi-team dataset management

### 10.6 Schema matching is strict and fully ordered

That is useful for safety, but it can also be brittle if a source changes harmlessly or adds useful columns.

Impact:

- easy to detect drift
- can create unnecessary failures when provider formats evolve

## 11. Practical End-To-End Example

For a configured Yahoo-backed dataset such as `market_macro`, the current flow is:

1. Load the tickers defined in `DATASET_CONFIGS`
2. Concatenate the per-ticker Yahoo results
3. Standardize the schema to the expected OHLCV columns
4. Parse the date column into UTC timestamps
5. Forward-fill missing values
6. Remove exact duplicates and duplicate timestamps
7. Check chronology, timestamp duplication, missingness, and schema match
8. Write a JSON quality report
9. Save the cleaned dataset to `data/processed/market_macro.csv`
10. Return a `DataPackage` containing the cleaned data and report

That is the real current behavior.

## 12. What Downstream Layers Can Trust

Downstream systems can currently assume the following only if the validation report says the dataset is valid:

- the dataframe has the configured columns
- the date column is UTC-normalized
- the rows are sorted chronologically
- duplicate timestamps have been removed
- missing values have been eliminated

They should not assume:

- FRED ingestion is available
- outlier handling is meaningful
- invalid datasets are blocked from export
- semantics have been validated beyond schema and chronology

## 13. Bottom Line

The current Data Layer is a good structural foundation with a real orchestration path, actual Yahoo ingestion, deterministic cleaning order, and explicit validation reporting.

But it is not complete.

The main weaknesses are:

- placeholder FRED support
- placeholder outlier handling
- rigid but shallow validation
- no hard stop on invalid output
- configuration that is broader than the current implementation

So the correct description of the layer today is:

It is a working, partially mature ingestion-and-cleaning pipeline that can produce standardized market datasets, but it still needs stronger validation, fuller provider support, and stricter integrity enforcement before it can be treated as fully production-hardened.
