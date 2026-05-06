# Workflow Map: From Raw Data to Risk Interpretation

## 1. High-Level Flow
The CRIS Layer 3 pipeline follows a strictly linear, decoupled workflow designed for reproducibility and institutional traceability.

![Workflow Map](diagrams/workflow_map.png)

```mermaid
graph TD
    A[Raw Market Data: OHLCV] --> B[Data Ingestion: data/ingest_data.py]
    B --> C[Orchestrator: layer3/core/orchestrator.py]
    
    subgraph "Independent Interpretation Engines"
        C --> D1[FAST SHOCK: short-horizon reflex]
        C --> D2[SLOW STRUCTURAL: persistence & entropy]
        C --> D3[TRAJECTORY: resilience & LSTM advisory]
    end
    
    D1 --> E[Convergence Coordinator: layer3/convergence/manager.py]
    D2 --> E
    D3 --> E
    
    E --> F[Probabilistic Transition & Smoothing]
    F --> G[Validated Output Contract: schema.py]
    
    subgraph "Verification & Assurance"
        G --> H1[Adversarial Audits: layer3/audit/]
        G --> H2[Behavioral Validation: layer3/validation/]
    end
```

---

## 2. Execution Tracing

### 2.1 Interpretation Step
1.  **Ingestion:** `ingest_data.py` pulls and cleans historical/live CSVs into a standardized format.
2.  **Dispatch:** `orchestrator.py` initializes the `Layer3State` (persistent EMA states + LSTM models).
3.  **Engine Run:** Three sub-engines execute in parallel, mapping raw returns/prices into local risk and confidence scores.
4.  **Arbitration:** `convergence/manager.py` applies inter-layer influence and adaptive weighting.
5.  **Persistence:** `ConvergenceState` is updated to preserve temporal dynamics for the next trading day.

### 2.2 Assurance Step
1.  **Validation:** `validation/behavioral_suite.py` runs synthetic scenarios to verify the system still "heals" and "panics" as expected.
2.  **Audit:** Institutional audit scripts (e.g., `trajectory_integrity_audit.py`) falsify the interpretation logic against historical adversarial eras.

---

## 3. Operational Entry Points
*   **Production Inference:** `run_layer3(...)` inside `orchestrator.py`.
*   **Structural Training:** `train_lstm(...)` inside `orchestrator.py`.
*   **System Check:** `python layer3/validation/behavioral_suite.py`.
*   **Institutional Audit:** `python layer3/audit/institutional_audit.py`.
