# CRIS Signal Extractor Layer Walkthrough

This walkthrough explains how to use the Signal Extractor Layer that was added for CRIS phase 2.

## What This Layer Does

The Signal Extractor Layer converts already-clean datasets from the Data Layer into standardized environmental signals.

It observes the environment only. It does not forecast, classify borrowers, infer market regimes, optimize portfolios, or make decisions.

## Core Flow

1. The Data Layer provides production-ready datasets.
2. You build an `ExtractionContext` from those datasets.
3. The `SignalExtractionManager` discovers built-in extractors and optional plugins.
4. The manager resolves explicit dependencies between extractors.
5. Each extractor emits a `SignalSet` containing typed `Signal` objects.
6. Telemetry records timing, signal counts, warnings, and failures.

## Main Package Pieces

- `signal_extractors.contracts`: signal, extractor, dependency, context, metadata, and error contracts.
- `signal_extractors.core`: registry, discovery, and execution manager.
- `signal_extractors.macro`: macro-focused extractors.
- `signal_extractors.market`: market-focused extractors.
- `signal_extractors.shared`: generic rolling, normalization, transform, statistics, and time helpers.
- `signal_extractors.plugins`: plugin discovery for external extractors.
- `signal_extractors.telemetry`: execution tracking.

## Signal Contract

Every signal includes:

- `name`
- `category`
- `value`
- `timestamp`
- `confidence`
- `source`
- `metadata`

Signals are observations only. They are not model outputs or decisions.

## Running the Layer

The basic pattern is:

1. Prepare a dictionary of already-clean `pandas.DataFrame` objects.
2. Wrap them in an `ExtractionContext`.
3. Create a `SignalExtractionManager`.
4. Call `execute(context)`.

The result contains:

- `signals`: the flattened signal list
- `signal_sets`: extractor-level outputs
- `execution_order`: the resolved dependency order
- `telemetry`: batch execution telemetry

## Example Usage

```python
from datetime import datetime, timezone

import pandas as pd

from signal_extractors import ExtractionContext, SignalExtractionManager

datasets = {
    "inflation_series": pd.DataFrame({"cpi": [1.0, 2.0, 3.0, 4.0]}),
    "volatility_series": pd.DataFrame({"vix": [15.0, 16.0, 18.0, 17.0]}),
}

context = ExtractionContext(
    datasets=datasets,
    as_of=datetime.now(timezone.utc),
)

manager = SignalExtractionManager()
result = manager.execute(context)

for signal in result.signals:
    print(signal.name, signal.value, signal.source)
```

## Built-In Extractors

### Macro

The built-in macro extractors include:

- Inflation
- Growth
- Labour
- Housing
- Rates
- Credit
- Currency
- Commodities
- Trade

These extractors convert cleaned macro datasets into standardized observations such as level, momentum, trend, and acceleration.

### Market

The built-in market extractors include:

- Volatility
- Breadth
- Liquidity
- Bonds
- Sectors
- Currencies
- Commodities

These extractors emit standardized observations from cleaned market datasets.

## Dependency Handling

Extractor dependencies are explicit.

If an extractor depends on another extractor, the manager resolves that dependency first and executes the graph in topological order.

This keeps execution deterministic and avoids hidden coupling.

## Adding a New Extractor

1. Subclass `SignalExtractor` or one of the shared series-based base classes.
2. Define a unique identity for the extractor.
3. Declare any required extractor dependencies.
4. Implement `extract(context)` and return a `SignalSet`.
5. Place the module under `signal_extractors/macro`, `signal_extractors/market`, or package it as a plugin.
6. If you ship it externally, expose it through the `signal_extractors.extractors` entry-point group.

New extractors register automatically when their module is imported.

## Plugin Support

Plugin discovery supports future extractors without changing core code.

The manager can load:

- Built-in extractor modules from `signal_extractors.macro` and `signal_extractors.market`
- External packages exposing the `signal_extractors.extractors` entry-point group

## Telemetry

Each extractor run records:

- execution time
- success or failure
- warnings
- errors
- number of signals produced

This is useful for observability without introducing any downstream decision logic.

## Validation

The current implementation is covered by unit, contract, integration, golden, and randomized property tests under `signal_extractors/tests`.

To run them in the CRIS environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CRIS
python -m pytest -q signal_extractors/tests
```
