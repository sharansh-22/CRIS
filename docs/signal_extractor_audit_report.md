# Signal Extractor Layer Audit Report

## Scope

This report audits the entire `signal_extractors/` subsystem end to end, including:

- `contracts/`
- `core/`
- `macro/`
- `market/`
- `shared/`
- `plugins/`
- `config/`
- `telemetry/`
- `tests/`

The audit focuses on architecture, implementation correctness, research integrity, reproducibility, determinism, performance, maintainability, extensibility, and production readiness.

The Signal Extractor Layer is intended to do one thing only: convert already-clean datasets into standardized environmental signals. It must not forecast, classify, infer regimes, optimize, rank, or make downstream decisions.

## Executive Summary

The subsystem is directionally sound and aligned with CRIS philosophy at the conceptual level, but it is not yet production-ready or research-hardened.

The core architecture is cleanly layered and the test suite passes, but several structural weaknesses remain:

- configuration options exist but are not fully wired into execution
- registry collisions can silently overwrite extractors
- plugin discovery relies on import side effects
- contract validation is thin for a research framework
- timestamp and dataset selection fallbacks can introduce leakage risk
- concurrency and caching are incomplete
- several test categories required by the requested audit are not implemented

In short: the framework is usable as a scaffold, but not yet strong enough for long-term, multi-team, multi-system research production.

## Overall Health Scores

- Overall Health Score: 66/100
- Architecture Score: 72/100
- Implementation Score: 64/100
- Research Integrity Score: 54/100
- Reliability Score: 61/100
- Maintainability Score: 63/100
- Scalability Score: 56/100
- Performance Score: 55/100
- Security Score: 70/100
- Production Readiness: 52/100

## Verification Outcome

The existing test slice under `signal_extractors/tests` passes, but the suite is narrow and leaves major architectural and integrity paths untested.

Observed result in the CRIS environment:

- `24 passed`

Important caveat: passing tests do not imply full framework health. Several categories requested in the audit are not implemented at all, and others are only partially enforced.

## Architecture Review

### What is good

The folder structure is coherent and matches the intended architecture:

- contracts define the public data model and failure modes
- core owns discovery, registry, and execution
- macro and market contain concrete extractor families
- shared contains generic math and time utilities
- telemetry is separated from business logic
- tests are isolated under the subsystem

The layer also respects the CRIS philosophy at a high level: extractors observe and emit standardized signals rather than making decisions.

### Architectural flaws

#### 1. Configuration is partially dead

`ExtractorRuntimeSettings` exposes values such as `default_lookback` and `telemetry_enabled`, but execution does not consistently honor them.

Impact:

- architectural: configuration is misleading rather than controlling behavior
- research: experiment settings may not actually take effect
- maintenance: users will trust knobs that do nothing

Recommendation:

- wire settings into extractor base classes and manager behavior
- make telemetry optional when disabled
- remove unused settings if they are not meant to be honored

#### 2. Registry collisions are silently overwritten

The registry stores extractor classes in a dict keyed by name, with no collision detection.

Impact:

- architectural: import order becomes behavior
- research: two plugins with the same name can produce different outputs without warning
- maintenance: accidental overwrite bugs are hard to diagnose

Recommendation:

- reject duplicate extractor names by default
- require explicit namespace scoping or override flags

#### 3. Plugin discovery is side-effect driven

Discovery imports built-in modules and loads entry points to trigger registration, then the manager relies on registry state afterward.

Impact:

- architectural: discovery and registration are coupled via implicit side effects
- research: plugin success is harder to validate explicitly
- maintenance: import-time failures can break the whole subsystem

Recommendation:

- make discovery return explicit candidates
- validate schema before registration
- isolate plugin failures from core execution

#### 4. The manager mutates extractor output to hide contract violations

If an extractor returns a mismatched `extractor_name`, the manager overwrites it in place.

Impact:

- architectural: broken implementations are hidden instead of surfaced
- research: provenance can become untrustworthy
- maintenance: invalid extractors can survive unnoticed

Recommendation:

- fail fast on contract mismatch
- do not mutate extractor output to compensate for incorrect implementations

#### 5. The registry is a mutable singleton without synchronization

The global registry is easy to use, but not thread-safe.

Impact:

- architectural: hidden shared state increases coupling
- reliability: concurrent discovery and execution can race
- scalability: parallel workflows are risky

Recommendation:

- remove global mutable state where possible
- or guard it with synchronization and immutable snapshots

## Contract and Interface Review

### What is good

The subsystem has the right public abstractions:

- `Signal`
- `SignalSet`
- `ExtractionContext`
- `ExtractorDependencies`
- `SignalExtractor`
- `ExtractorMetadata`

This is a strong foundation for a framework that should remain extensible.

### Contract flaws

#### 6. Signal validation is incomplete

`Signal` validates confidence, name, and source, but not:

- timestamp type or timezone consistency
- category consistency
- metadata schema
- value type or finiteness

Impact:

- architectural: contract is not strict enough for cross-team use
- research: malformed signals may be treated as valid observations
- maintenance: downstream layers cannot rely on a stable schema

Recommendation:

- extend validation to cover timestamp, category, metadata, and numeric stability
- add a schema validator for signal payloads

#### 7. SignalSet is mutable

The container can be edited after creation.

Impact:

- research: output provenance can change after execution
- maintenance: post-run mutation can invalidate audit trails

Recommendation:

- make SignalSet immutable or tuple-backed
- ensure signals are frozen or treated as value objects

#### 8. ExtractionContext exposes mutable artifacts and raw datasets

The context is a thin wrapper rather than a protected runtime contract.

Impact:

- research: extractor code can mutate context state unexpectedly
- maintenance: hidden side effects become hard to trace

Recommendation:

- make datasets and artifacts read-only views
- define a stricter context contract for allowed fields

#### 9. Validation errors are defined but not used

`SignalValidationError` exists, but the package does not raise it in meaningful contract validation paths.

Impact:

- architectural: error hierarchy is incomplete in practice
- maintenance: callers cannot distinguish contract failures reliably

Recommendation:

- use domain-specific exceptions in validation paths
- keep contract violations separate from execution failures

## Dependency Graph Review

### What is good

The manager has explicit topological dependency resolution and cycle detection.

### Dependency flaws

#### 10. Shared dependency caching is not implemented

Required upstream extractors are resolved, but there is no caching or reuse of intermediate outputs.

Impact:

- performance: repeated computation is possible
- scalability: larger dependency graphs will incur unnecessary cost

Recommendation:

- cache extractor outputs per execution context
- compute shared upstream dependencies once

#### 11. Dependency resolution is not tied to explicit dataset contracts

Dependencies are only expressed at the extractor level, not at the required dataset/schema level.

Impact:

- research: the manager can order extractors correctly but still feed them ambiguous inputs
- maintenance: dependencies are incomplete as a contract surface

Recommendation:

- add explicit dataset and column requirements to the dependency model

## Data Validation Review

This is the largest integrity gap.

The subsystem assumes clean data, but it does not actually enforce the properties that a clean signal-extraction layer needs.

### Missing validation categories

#### 12. Required dataset presence is not enforced

No extractor declares a formal required-dataset contract that is checked before extraction.

Impact:

- runtime failures become late failures
- research integrity suffers because inputs are not formally constrained

Recommendation:

- declare required datasets per extractor
- validate context before extraction begins

#### 13. Required columns are not validated

The code currently falls back to the first numeric series when hints do not match.

Impact:

- severe research risk: an extractor may silently use the wrong column
- reproducibility: same code can behave differently across similar datasets

Recommendation:

- reject ambiguous inputs
- require explicit column mappings per extractor

#### 14. Timestamp order is not validated

No extractor or manager-level check enforces sorted timestamps.

Impact:

- future-row leakage becomes possible
- rolling metrics can become lookahead-contaminated on malformed inputs

Recommendation:

- require monotonic ordering before any rolling or trend calculation
- fail fast on unsorted timestamps

#### 15. Duplicate timestamps are not validated

Duplicate time indices can distort rolling metrics and trend estimates.

Impact:

- research: time series may be double counted
- maintenance: debugging signal anomalies becomes difficult

Recommendation:

- reject duplicate timestamps or define deterministic aggregation rules

#### 16. UTC enforcement is incomplete

Timestamps are normalized when inferred, but the framework does not enforce UTC for all incoming data.

Impact:

- reproducibility: timezone drift can shift window boundaries
- research: monthly or daily alignment can become ambiguous

Recommendation:

- require UTC-aware timestamps at the contract boundary

## Research Integrity Review

### Critical leakage risks

#### 17. Fallback to current time can introduce time dependence

If a timestamp cannot be inferred, the framework falls back to `datetime.now(...)`.

Why this matters:

- outputs can vary by execution time
- generated signals lose reproducibility
- hidden temporal dependence can contaminate audit trails

Recommendation:

- never invent timestamps silently
- require timestamps to exist in the source dataset or context

#### 18. First-numeric-column fallback can select the wrong feature

If name hints do not match, the first numeric column is used.

Why this matters:

- this is a major source of silent semantic leakage
- extractors can appear to work while using unintended data

Recommendation:

- make column selection explicit and mandatory
- raise errors on ambiguous series selection

#### 19. Rolling calculations assume correct ordering but do not verify it

Trailing windows are safe only if the data is already sorted.

Why this matters:

- one malformed dataset can create lookahead leakage
- research findings become unreproducible

Recommendation:

- sort and validate inputs before calculation, or reject malformed input

#### 20. No explicit leakage guardrails exist

The framework does not have protective checks for future rows, duplicate timestamps, or misordered inputs.

Why this matters:

- the entire purpose of an observational layer is undermined if it can accidentally see future information

Recommendation:

- add a reusable input validator for time-series integrity

- make it part of the contract, not an optional helper

## Determinism Review

### What is good

The current built-in extractors are deterministic for clean synthetic inputs.

### Determinism flaws

#### 21. Discovery order is a hidden influence

Import-based registration and dict insertion order can affect execution order when the registry changes.

Impact:

- repeated runs are less robust than they appear

Recommendation:

- stabilize discovery and registration order explicitly

#### 22. Time fallback breaks full determinism

If timestamps are missing, current time is used.

Impact:

- repeated runs can differ across executions

Recommendation:

- eliminate runtime-dependent fallback timestamps

## Numerical Stability Review

### What is good

- constant-series behavior is handled reasonably
- zero-variance paths do not explode numerically

### Numerical flaws

#### 23. NaN handling is not fully standardized

Short histories or ambiguous inputs can still yield NaN signal values.

Impact:

- downstream consumers need to guess whether NaN is expected or an error

Recommendation:

- define a policy for NaN outputs
- either reject insufficient history or encode explicit missingness metadata

#### 24. Outlier behavior is not bounded

There is no robust clipping or winsorization policy in the extractor metrics.

Impact:

- extreme values can dominate trend-based outputs
- reproducibility across unstable periods may weaken

Recommendation:

- define bounded transforms or explicit outlier handling at the shared-utility layer

## Behavioral Correctness Review

### Main issue

Several extractor names are broader than the behavior they currently implement.

#### 25. Volatility extractor does not compute a true variance-based volatility metric

It emits generic level, rolling mean, trend, and acceleration outputs instead of a variance-derived volatility observation.

Why this matters:

- the signal name can mislead future researchers
- semantics and implementation are not tightly aligned

Recommendation:

- compute an explicit volatility measure, such as rolling standard deviation or realized variance, and name it precisely

#### 26. Breadth and sector signals are too generic

Breadth and sector extractors rely on generic series metrics and do not fully encode the behavior implied by their names.

Why this matters:

- future users may assume domain-specific metrics that are not present

Recommendation:

- define extractor-specific semantics more tightly
- document the exact observation each signal represents

## Performance Review

### What is good

The subsystem is lightweight and has no obvious heavy compute bottlenecks in the current test slice.

### Performance flaws

#### 27. Discovery is rerun on every manager execution

Package scanning and entry-point loading happen repeatedly.

Impact:

- avoidable overhead
- duplicated work in repeated batch runs

Recommendation:

- cache discovery results
- separate startup discovery from per-run execution

#### 28. No explicit caching layer exists

Shared upstream computations are not reused.

Impact:

- dependency-heavy graphs could scale poorly

Recommendation:

- cache per-context extractor outputs for shared dependencies

#### 29. No benchmarking or memory profiling exists

Requested performance verification categories are not implemented.

Impact:

- performance assumptions are unproven

Recommendation:

- add latency and memory benchmarks to the test or validation suite

## Plugin System Review

### What is good

The entry-point idea is appropriate for future extensibility.

### Plugin flaws

#### 30. Plugin schema validation is missing

There is no explicit validation for third-party plugin shape or metadata.

Impact:

- invalid plugins can enter the registry too easily

Recommendation:

- validate plugin class identity, dependencies, and signal contract before registration

#### 31. Plugin failure isolation is missing

A plugin failure can abort the whole workflow.

Impact:

- one bad plugin can take down the entire extraction run

Recommendation:

- isolate plugin failures and allow the core subsystem to continue when safe

#### 32. Plugin timeout handling is missing

There is no execution timeout or watchdog support.

Impact:

- a stuck plugin can block the pipeline

Recommendation:

- add plugin execution guards or timeouts if plugins are expected to be untrusted or expensive

## Telemetry Review

### What is good

Telemetry captures signal counts, warnings, errors, and execution timing.

### Telemetry flaws

#### 33. Telemetry is not fully configuration-driven

The settings surface implies control, but the manager always creates telemetry.

Impact:

- observability is less configurable than it appears

Recommendation:

- make telemetry optional and respect runtime configuration

#### 34. Telemetry does not enforce output immutability

The current code keeps telemetry separate, but output mutation protections are not strong enough overall.

Impact:

- auditability can degrade if objects are mutated after execution

Recommendation:

- freeze outputs or copy them before recording telemetry if mutability remains

## Parallel Execution Review

### Main issue

The subsystem is not proven parallel-safe.

#### 35. Shared mutable registry state is a race risk

Concurrent discovery and registration can collide.

Impact:

- race conditions
- nondeterministic discovery state

Recommendation:

- avoid mutable global registry state or add synchronization

#### 36. Thread safety is not tested

No parallel-safe execution test exists.

Impact:

- concurrency bugs may remain hidden until production

Recommendation:

- add explicit parallel execution tests if concurrent use is a goal

## Testing Coverage Review

### What is covered

The current suite validates:

- contract basics
- extractor output shape
- one dependency-order example
- a golden snapshot
- a small randomized property check
- an integration run

### What is missing

The requested audit categories that remain unimplemented or untested include:

- duplicate registry collision handling
- plugin discovery schema validation
- plugin failure isolation
- plugin timeouts
- missing dependency failures at the contract level
- explicit required-dataset checks
- required-column checks
- timezone and timestamp ordering enforcement
- duplicate timestamp rejection
- no-future-row leakage tests
- output immutability tests
- parallel safety tests
- latency and memory benchmarks

## Severity Summary

### Critical

- none currently observed as a direct runtime failure in the existing test slice, but research leakage risk is close to critical in malformed-input scenarios

### High

- configuration surface not wired to behavior
- silent registry collisions
- weak research-integrity validation
- hidden fallback timestamps
- ambiguous numeric-column fallback

### Medium

- plugin discovery is side-effect driven
- contract validation is thin
- shared dependency caching missing
- output mutability is too permissive
- volatility/breadth semantics are too generic
- concurrency hardening incomplete

### Low

- some helper implementations are stylistically inconsistent in their coercion strategy
- telemetry and configuration have room for tightening and simplification

## Technical Debt Estimate

Moderate-high.

The architecture is promising, but the framework still carries debt in the form of implicit behavior, weak validation, unbounded discovery side effects, and incomplete reproducibility guardrails.

## Recommendations by Priority

### Priority 1

- make Signal immutable and strengthen SignalSet constraints
- enforce dataset and timestamp contracts before extraction
- remove fallback timestamps and ambiguous series selection
- fail fast on registry collisions
- wire configuration into actual runtime behavior

### Priority 2

- add plugin schema validation and failure isolation
- cache discovery and shared dependencies
- add tests for leakage, ordering, duplicates, and parallel safety
- define stricter semantics for each market and macro extractor

### Priority 3

- add performance benchmarks
- add telemetry toggles
- refine error hierarchy usage
- reduce mutable global state

## Final Verdict

### Is the Signal Extractor Layer architecturally sound?

Mostly yes. The package layout, abstraction boundaries, and intent are sound.

### Is it production-ready?

No. It needs stronger validation, collision handling, plugin isolation, and concurrency hardening.

### Is it research-ready?

Partially. It is suitable for controlled, clean-input experiments, but not yet for strict reproducibility standards.

### Is it extensible?

Yes in design, but the extension mechanism still needs enforcement and clearer contracts.

### Is it consistent with CRIS philosophy?

Yes. It remains observational and does not perform downstream decision logic.

### Can implementation proceed to the Environmental Intelligence Engine?

Not yet. The current subsystem should be hardened first, especially around validation, determinism, plugin safety, and leakage prevention.
