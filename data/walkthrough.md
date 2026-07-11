# CRIS Data Layer - End to End Walkthrough

The CRIS Data Layer is fundamentally designed to produce robust, deterministic, and strictly-typed datasets for downstream quantitative signal extractors. By ensuring absolute temporal integrity and zero silent coercion, the layer forms the backbone of the CRIS v2 architecture.

## 1. Directory Architecture
The data layer isolates responsibilities across several specialized submodules:
- `loaders/`: Raw API adapters containing isolated extraction logic (e.g., `yahoo.py`, `fred.py`).
- `cleaning/`: The immutable math engine powered by Polars. This handles standardizations (`casting.py`, `datetime.py`, `missing_values.py`, `outliers.py`).
- `validation/`: The structural firewall powered by Pandera (`schemas.py`, `drift.py`), ensuring that output shapes never drift silently.
- `orchestration/`: Dagster SDAs (`assets.py`) that strictly map data provenance and prevent downstream models from ingesting broken data.
- `pipeline/`: Legacy iteration loop implementations (`prepare_data.py`), which have now been bridged into the Dagster ecosystem.

## 2. Ingestion to Execution (The Journey)
A dataset's lifecycle follows a strict sequence:
1. **Extraction (raw)**: Fetched directly via a DAG asset and held in memory as a Pandas DataFrame.
2. **Immutable Compute (staged)**: The dataframe boundary crosses into a strictly-typed Polars environment. Missing values are filled based on contract configurations, and outliers are mathematically treated without silent type coercions. Any anomaly raises a loud `TypeError`.
3. **Validation (validated)**: The data transitions back to Pandas to undergo Pandera validation against predefined domains (e.g., `MarketDataSchema`).
4. **Serialization (published)**: The dataset is mathematically written to disk using PyArrow. This preserves precise `float64` limits and specific `datetime64[ns, UTC]` markers ensuring zero fidelity loss over the terminal handoff.

## 3. Fault Tolerance & The Quarantine Protocol
CRIS never silently mutates or drops failing elements. 
If an asset fails structurally at the Pandera gate (e.g., a sudden string appears in an expected float column), a `SchemaError` triggers a physical quarantine. The pipeline writes the unmodified raw payload, the intermediate staged payload, and a precise error trace into `quarantine/<timestamp>/` for human review, and then strictly fails the Dagster pipeline to protect downstream consumers.

## 4. Health & Constraints
- **State**: The `environment.yml` tracks required dependencies natively (`polars`, `dagster`, `pandera`, `pyarrow`).
- **Assurances**: The serialization logic has been proven to mathematically eliminate downcasting, meaning no float compression occurs between the pipeline layer and the signal extractor layer.
