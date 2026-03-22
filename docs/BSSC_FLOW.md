# BSSC — Data Flow
## Layer 3: Black Swan and Slippage Computer

## Section 1 — Pipeline Overview

  ┌─────────────────────────────────────────────────┐
  │           BSSC LAYER 3 PIPELINE                 │
  │                                                 │
  │  INPUT: ticker, csv_path, config                │
  │              ↓                                  │
  │  ┌─────────────────────┐                        │
  │  │   simulation.py     │                        │
  │  │  Merton Jump-       │                        │
  │  │  Diffusion          │                        │
  │  │  2000 GBM paths     │                        │
  │  │  2000 JD paths      │                        │
  │  └──────────┬──────────┘                        │
  │             │ SimulationResult                  │
  │             ↓                                   │
  │  ┌─────────────────────┐                        │
  │  │    entropy.py       │                        │
  │  │  Two-Signal System  │                        │
  │  │  ① Perm Entropy     │                        │
  │  │    Alarm            │                        │
  │  │  ② Vol Ratio        │                        │
  │  │    Confirmation     │                        │
  │  └──────────┬──────────┘                        │
  │             │ EntropyResult                     │
  │             ↓                                   │
  │  ┌─────────────────────┐                        │
  │  │    slippage.py      │                        │
  │  │  Almgren-Chriss     │                        │
  │  │  Monte Carlo IS     │                        │
  │  │  2000 paths         │                        │
  │  └──────────┬──────────┘                        │
  │             │ SlippageResult                    │
  │             ↓                                   │
  │  ┌─────────────────────┐                        │
  │  │    models.py        │                        │
  │  │  Pydantic v2        │                        │
  │  │  Layer3Report       │                        │
  │  └──────────┬──────────┘                        │
  │             │                                   │
  │             ↓                                   │
  │  ┌─────────────────────┐                        │
  │  │    detector.py      │                        │
  │  │  Orchestrator       │                        │
  │  │  + Assembler        │                        │
  │  └──────────┬──────────┘                        │
  │             │                                   │
  │             ↓                                   │
  │  ┌─────────────────────┐                        │
  │  │    report.py        │                        │
  │  │  JSON (atomic)      │                        │
  │  │  Markdown           │                        │
  │  └──────────┬──────────┘                        │
  │             │                                   │
  │  OUTPUT: Layer3Report → Convergence Engine      │
  └─────────────────────────────────────────────────┘

## Section 2 — Two-Signal Detection Detail

  MARKET DATA
       │
       ├──────────────────────────────────────┐
       │                                      │
       ↓                                      ↓
  ┌──────────────────┐             ┌──────────────────┐
  │ SIGNAL 1         │             │ SIGNAL 2         │
  │ Permutation      │             │ Volatility       │
  │ Entropy Alarm    │             │ Ratio            │
  │                  │             │ Confirmation     │
  │ Rolling 20-day   │             │ Rolling 10-day   │
  │ perm entropy     │             │ mean abs return  │
  │                  │             │ ÷ 0.714%         │
  │ Fires when:      │             │                  │
  │ perm < base-0.05 │             │ STRESS:          │
  │                  │             │ ratio > 1.5x     │
  │ No persistence   │             │ for 10+ days     │
  │ gate — fires     │             │                  │
  │ immediately      │             │ BLACK SWAN:      │
  │                  │             │ ratio > 3.0x     │
  │ Lead time:       │             │ for 5+ days      │
  │ COVID:  56 days  │             │                  │
  │ Q4 2018: 11 days │             │ Persistence gate │
  │ Vaccine: 0 days  │             │ filters noise    │
  └────────┬─────────┘             └────────┬─────────┘
           │                                │
           └──────────────┬─────────────────┘
                          ↓
                  ┌───────────────┐
                  │ COMBINED      │
                  │ CLASSIFICATION│
                  │               │
                  │ NORMAL        │
                  │ WATCH         │
                  │ STRESS        │
                  │ BLACK SWAN    │
                  └───────────────┘

## Section 3 — State Transition Logic

| Vol Ratio State | Alarm Active | Final State  |
|-----------------|--------------|--------------|
| NORMAL          | No           | NORMAL       |
| NORMAL          | Yes          | WATCH        |
| STRESS          | Any          | STRESS       |
| BLACK_SWAN      | Any          | BLACK_SWAN   |

## Section 4 — Slippage Pipeline Detail

  Price paths from simulation.py
       ↓
  Max drawdown peak identified per path
  (forced liquidation scenario)
       ↓
  Market impact = η × σ_daily × √(Q/ADV)
  (path volatility only — no regime multiplier)
       ↓
  Spread cost = base_spread × regime_multiplier
  (regime multiplier applied here only)
       ↓
  Total IS = market_impact + spread_cost + timing_risk
       ↓
  Distribution across 2000 paths:
  Mean: 490 bps | P95: 1948 bps | P99: 2729 bps

## Section 5 — Output Structure

  Layer3Report
  ├── SimulationResult  (kurtosis, skewness, min returns)
  ├── EntropyResult     (market state, breach duration,
  │                      vol ratio, alarm status)
  ├── SlippageResult    (IS distribution, JD/GBM ratio)
  ├── overall_risk_level (NORMAL/STRESS/BLACK_SWAN)
  └── recommended_action (HOLD/REDUCE/LIQUIDATE)
       │
       ├── JSON → data/simulation_output/ (atomic write)
       ├── Markdown → reports/
       └── → Convergence Engine
