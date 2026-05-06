# CRIS — Component Architecture Map

## Layer 3: Probabilistic Interpretation Framework
```text
layer3/
├── core/
│   ├── orchestrator.py         # Main execution pipeline
│   ├── schema.py               # Pydantic probabilistic contracts
│   └── config.py               # Interpretation thresholds & baselines
│
├── audit/                      # Institutional Falsification Suites
│   ├── institutional_audit.py  # Master audit coordinator (CLI)
│   ├── adversarial_stress_audit.py
│   ├── trajectory_integrity_audit.py
│   ├── convergence_stability_audit.py
│   └── long_duration_stability_audit.py
│
├── validation/                 # Behavioral Integrity Tests
│   ├── behavioral_suite.py     # Master validation runner
│   ├── stress_interpretation_tests.py
│   └── recovery_tests.py
│
├── fast_shock/                 # Short-horizon instability field
├── slow_structural/            # Persistent structural stress field
├── trajectory_engine/          # Resilience degradation engine
├── convergence/                # Probabilistic temporal coordinator
└── shared/                     # Normalized math & utilities
```

## Documentation & Visuals
```text
docs/
├── diagrams/                   # Institutional Architecture Visuals
│   ├── architecture_overview.png
│   ├── probabilistic_flow.png
│   ├── convergence_dynamics.png
│   ├── validation_pipeline.png
│   ├── audit_pipeline.png
│   └── workflow_map.png
│
├── architecture_overview.md    # System design & field definitions
├── probabilistic_philosophy.md # Why interpretation > prediction
├── convergence_design.md       # Weight arbitration & smoothing
├── validation_methodology.md   # Behavioral testing standards
├── adversarial_testing.md      # Falsification audit methodology
├── limitations.md              # Operational boundaries
├── future_work.md              # Multi-layer integration roadmap
└── workflow_map.md             # Execution & tracing guide
```

## System Readiness
- **Layer 3**: ✅ **COMPLETE** (Institutionally Hardened)
- **Layer 2**: ⏸️ NOT STARTED
- **Layer 1**: ⏸️ NOT STARTED
- **Layer 4**: ⏸️ NOT STARTED
