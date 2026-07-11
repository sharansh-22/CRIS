# Signal Extractor Layer

## Purpose

The Signal Extractor Layer converts already-clean datasets from the Data Layer into standardized environmental signals.

It observes the environment and emits structured observations for later layers. It does not forecast, classify borrowers, infer regimes, optimize portfolios, or make decisions.

## Responsibilities

- Accept production-ready datasets only.
- Convert raw observations into typed Signal objects.
- Keep extractor logic domain-specific but interpretation-free.
- Declare explicit extractor and dataset dependencies.
- Remain agnostic to downstream systems.

## Lifecycle

1. A caller builds an ExtractionContext from cleaned datasets.
2. The manager discovers available extractors through built-ins and plugin entry points.
3. The manager resolves dependencies and establishes execution order.
4. Each extractor emits a SignalSet.
5. Telemetry records timing, success state, warnings, errors, and signal counts.

## Signal Contract

Every signal must provide:

- name
- category
- value
- timestamp
- confidence
- source
- metadata

Signals are immutable observations. They are not decisions or predictions.

## Adding New Extractors

1. Subclass SignalExtractor or one of the shared series-based base classes.
2. Define an identity with a unique extractor name and category.
3. Declare extractor dependencies explicitly when required.
4. Implement extract(context) and return a SignalSet.
5. Place the module under signal_extractors/macro, signal_extractors/market, or a plugin package.

New extractors can also be distributed as plugins by exposing them through the signal_extractors.extractors entry-point group.

## Dependency Resolution

Extractor dependencies are explicit and directed. The manager expands required upstream extractors before execution and sorts the resulting graph topologically.

This keeps the layer deterministic and keeps dependency knowledge inside the extractor contract rather than the caller.

## Plugin Discovery

Plugin discovery loads two kinds of extractors:

- Built-in modules under signal_extractors.macro and signal_extractors.market.
- External packages exposing the signal_extractors.extractors entry point group.

Discovery imports modules so extractors can register themselves automatically.

## Signal Lifecycle

Signals are created from a selected cleaned series, enriched with metadata, recorded with telemetry, and passed forward as standardized observations.

They are never transformed into decisions in this layer.
