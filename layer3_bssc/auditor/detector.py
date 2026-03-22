"""
detector.py — Layer 3 Pipeline Orchestrator and Assembler

Two-function architecture:

_assemble_layer3_report() [PRIVATE]
  Pure assembly — zero computation.
  Takes raw dicts from three pipeline stages.
  Constructs Pydantic models. Returns Layer3Report.
  Use for testing and development.

run_layer3_pipeline() [PUBLIC]
  Full orchestration — single convergence engine entry.
  One function call returns complete Layer3Report.
  Internally calls _assemble_layer3_report().

Detection Architecture (TS-002 validated):
  Signal 1: Permutation Entropy Alarm
    Fires when perm entropy drops below baseline - 0.05
    No persistence gate — early warning
    Validated: 56-day lead time on COVID
               11-day lead time on Q4 2018
               Silent on vaccine rally

  Signal 2: Volatility Ratio Confirmation
    Baseline: 0.714% daily move (B Split, TS-002)
    STRESS:     vol_ratio > 1.5x for 10+ days
    BLACK_SWAN: vol_ratio > 3.0x for 5+ days
    Validated: filters vaccine rally 8-day streak
               confirms COVID 23-day black swan

  States: NORMAL / WATCH / STRESS / BLACK_SWAN
  WATCH = alarm active, vol ratio not yet confirmed
  WATCH maps to STRESS in Pydantic models
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from layer3_bssc.engine.models import (
    SimulationResult,
    EntropyResult,
    SlippageResult,
    Layer3Report,
)
from layer3_bssc.engine.simulation import (
    calibrate_from_data,
    simulate_gbm,
    simulate_jumps,
)
from layer3_bssc.engine.entropy import (
    compute_permutation_entropy,
    compute_volatility_regime,
    compute_permutation_alarm,
    classify_market_state,
)
from layer3_bssc.engine.slippage import (
    run_monte_carlo_slippage,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

# TS-002 validated baseline (B Split winner)
TS002_BASELINE_MEAN_ABS = 0.00714

# Volatility ratio thresholds
STRESS_MULTIPLIER   = 1.5
BSWAN_MULTIPLIER    = 3.0

# Persistence gates (empirically validated)
# 10-day gate: filters vaccine rally 8-day streak
# 5-day gate: fires on COVID 23-day black swan streak
STRESS_PERSISTENCE_DAYS = 10
BSWAN_PERSISTENCE_DAYS  = 5

# Permutation entropy alarm
# Validated: fires 56 days before vol ratio on COVID
#            fires 11 days before vol ratio on Q4 2018
#            silent on vaccine rally (0 false alarms)
PERM_ALARM_WINDOW     = 20
PERM_ALARM_DROP       = 0.05
PERM_BASELINE_START   = "2018-01-01"
PERM_BASELINE_END     = "2018-03-31"

# Default pipeline config
_DEFAULT_CONFIG = {
    "n_paths":          200,
    "T":                1.0,
    "dt":               1 / 252,
    "order_frac":       0.01,
    "eta":              0.3,
    "execution_window": 5,
    "lambda_j":         2.0,
    "mu_j":            -0.15,
    "sigma_j":          0.10,
    "calm_start":       "2018-01-01",
    "calm_end":         "2018-12-31",
    "event_start":      "2020-02-01",
    "event_end":        "2020-03-31",
}

def _assemble_layer3_report(
    simulation_dict: dict,
    entropy_dict: dict,
    slippage_dict: dict,
    notes: Optional[str] = None,
) -> Layer3Report:
    """
    Pure assembly. Zero computation. Zero pipeline calls.
    Maps raw dicts to Pydantic models.
    Assembles Layer3Report.
    Runs validate_pipeline_consistency().
    Logs warnings. Returns report.
    """
    try:
        sim = SimulationResult(
            ticker=simulation_dict["ticker"],
            S0=simulation_dict["S0"],
            mu=simulation_dict["mu"],
            sigma=simulation_dict["sigma"],
            n_paths=simulation_dict["n_paths"],
            kurtosis_gbm=simulation_dict["kurtosis_gbm"],
            kurtosis_jd=simulation_dict["kurtosis_jd"],
            skewness_jd=simulation_dict["skewness_jd"],
            min_return_gbm=simulation_dict["min_return_gbm"],
            min_return_jd=simulation_dict["min_return_jd"],
            notes=simulation_dict.get("notes")
        )
    except Exception as e:
        raise RuntimeError(f"Failed to assemble SimulationResult from {simulation_dict.keys()}: {e}") from e

    try:
        ent = EntropyResult(
            ticker=entropy_dict.get("ticker", sim.ticker),
            primary_method=entropy_dict.get("primary_method", "volatility_ratio"),
            confirmation_method=entropy_dict.get("confirmation_method", "permutation_alarm"),
            baseline_entropy=entropy_dict["baseline_entropy"],
            event_entropy=entropy_dict["event_entropy"],
            entropy_delta=entropy_dict["entropy_delta"],
            peak_entropy_date=entropy_dict.get("peak_entropy_date", ""),
            market_state=entropy_dict["market_state"],
            breach_duration_days=entropy_dict["breach_duration_days"],
            baseline_period_used=entropy_dict.get("baseline_period_used", "")
        )
    except Exception as e:
        raise RuntimeError(f"Failed to assemble EntropyResult from {entropy_dict.keys()}: {e}") from e

    try:
        if "total_slippage" in slippage_dict:
            mean_bps = slippage_dict["total_slippage"]["mean"]
            median_bps = slippage_dict["total_slippage"]["median"]
            p95_bps = slippage_dict["total_slippage"]["p95"]
            p99_bps = slippage_dict["total_slippage"]["p99"]
            max_bps = slippage_dict["total_slippage"]["max"]
            std_bps = slippage_dict["total_slippage"]["std"]
        else:
            mean_bps = slippage_dict["mean_bps"]
            median_bps = slippage_dict["median_bps"]
            p95_bps = slippage_dict["p95_bps"]
            p99_bps = slippage_dict["p99_bps"]
            max_bps = slippage_dict["max_bps"]
            std_bps = slippage_dict["std_bps"]

        slip = SlippageResult(
            ticker=slippage_dict.get("ticker", sim.ticker),
            mode=slippage_dict["mode"],
            n_paths=slippage_dict["n_paths"],
            mean_bps=mean_bps,
            median_bps=median_bps,
            p95_bps=p95_bps,
            p99_bps=p99_bps,
            max_bps=max_bps,
            std_bps=std_bps,
            jd_gbm_ratio=slippage_dict["jd_gbm_ratio"],
            regime_breakdown=slippage_dict["regime_breakdown"],
            entropy_state_at_execution=slippage_dict["entropy_state_at_execution"],
            wandb_run_url=slippage_dict.get("wandb_run_url")
        )
    except Exception as e:
        raise RuntimeError(f"Failed to assemble SlippageResult from {slippage_dict.keys()}: {e}") from e

    try:
        report = Layer3Report(
            ticker=sim.ticker,
            simulation=sim,
            entropy=ent,
            slippage=slip,
            notes=notes
        )
    except Exception as e:
        raise RuntimeError(f"Failed to assemble Layer3Report: {e}") from e

    warnings = report.validate_pipeline_consistency()
    for w in warnings:
        logger.warning(w)

    return report


def run_layer3_pipeline(
    ticker: str,
    csv_path: Path,
    config: Optional[dict] = None,
    use_wandb: bool = True,
) -> Layer3Report:
    """
    Full pipeline orchestration.
    Single entry point for convergence engine.
    """
    try:
        # STAGE 1 — Validate inputs and merge config
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing data file {csv_path}. Please run ingest_data.py first.")

        cfg = dict(_DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        
        # Enforce reasonable paths for speed
        if cfg["n_paths"] > 200:
            cfg["n_paths"] = 200

        logger.debug("[Layer3] Config: %s", cfg)

        # STAGE 2 — WandB initialization
        wandb_run_url = None
        if use_wandb and _WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="CRIS",
                    name=f"Layer3-{ticker}-pipeline",
                    tags=["layer3", "pipeline", ticker],
                    config=cfg,
                )
                wandb_run_url = wandb.run.get_url()
            except Exception as e:
                logger.warning(f"WandB initialization failed: {e}")

        # STAGE 3 — Simulation
        try:
            S0, mu, sigma = calibrate_from_data(csv_path)

            gbm_returns_all = []
            jd_returns_all  = []

            for i in range(cfg["n_paths"]):
                seed_gbm = 42 + i
                seed_jd  = 10042 + i

                _, gbm_path = simulate_gbm(
                    S0, mu, sigma,
                    cfg["T"], cfg["dt"],
                    seed=seed_gbm
                )
                _, jd_path, _ = simulate_jumps(
                    S0, mu, sigma,
                    cfg["lambda_j"],
                    cfg["mu_j"],
                    cfg["sigma_j"],
                    cfg["T"], cfg["dt"],
                    seed=seed_jd,
                )

                gbm_returns_all.append(
                    np.diff(np.log(np.maximum(gbm_path, 1e-10)))
                )
                jd_returns_all.append(
                    np.diff(np.log(np.maximum(jd_path, 1e-10)))
                )

            gbm_returns_flat = np.concatenate(gbm_returns_all)
            jd_returns_flat  = np.concatenate(jd_returns_all)

            kurtosis_gbm = float(stats.kurtosis(gbm_returns_flat))
            kurtosis_jd = float(stats.kurtosis(jd_returns_flat))
            skewness_jd = float(stats.skew(jd_returns_flat))
            min_return_gbm = float(gbm_returns_flat.min())
            min_return_jd = float(jd_returns_flat.min())

            simulation_dict = {
                "ticker": ticker,
                "S0": float(S0),
                "mu": float(mu),
                "sigma": float(sigma),
                "n_paths": int(cfg["n_paths"]),
                "kurtosis_gbm": kurtosis_gbm,
                "kurtosis_jd": kurtosis_jd,
                "skewness_jd": skewness_jd,
                "min_return_gbm": min_return_gbm,
                "min_return_jd": min_return_jd,
                "notes": None,
            }
            logger.info("[Layer3][Stage 1] Simulation complete — S0=%.2f sigma=%.4f", S0, sigma)
        except Exception as e:
            raise RuntimeError(f"[Layer3][Stage 1] Simulation failed: {e}") from e

        # STAGE 4 — Two-Signal Market State Detection
        try:
            # Load price data — yfinance multi-level header
            df_prices = pd.read_csv(
                csv_path, header=[0, 1],
                index_col=0, parse_dates=True
            )
            df_prices.columns = [
                col[0] for col in df_prices.columns
            ]
            close_series = (
                df_prices["Close"].dropna().astype(float)
            )
            returns_full = close_series.pct_change().dropna()
            returns_full = returns_full.loc[
                ~returns_full.index.duplicated(keep="first")
            ]

            # Event window
            event_returns = returns_full[
                cfg["event_start"]:cfg["event_end"]
            ]
            if len(event_returns) < 10:
                raise ValueError(
                    f"Event window {cfg['event_start']} to "
                    f"{cfg['event_end']} has only "
                    f"{len(event_returns)} days. Min: 10."
                )

            # Baseline period
            baseline_returns = returns_full[
                cfg["calm_start"]:cfg["calm_end"]
            ]
            if len(baseline_returns) < 30:
                baseline_returns = returns_full.iloc[:252]
                baseline_period_used = (
                    f"{returns_full.index[0].date()} to "
                    f"{returns_full.index[min(251,len(returns_full)-1)].date()}"
                    f" (252 day fallback)"
                )
            else:
                baseline_period_used = (
                    f"{cfg['calm_start']} to {cfg['calm_end']}"
                )

            # Permutation entropy baseline from 2018-Q1
            perm_baseline_data = returns_full[
                PERM_BASELINE_START:PERM_BASELINE_END
            ]
            if len(perm_baseline_data) < 20:
                perm_baseline_data = returns_full.iloc[:63]

            baseline_perm = compute_permutation_entropy(
                perm_baseline_data
            )

            # Signal 1: Permutation Entropy Alarm
            perm_alarm = compute_permutation_alarm(
                returns=event_returns,
                baseline_perm_entropy=baseline_perm,
                alarm_drop_threshold=PERM_ALARM_DROP,
                rolling_window=PERM_ALARM_WINDOW,
            )

            # Signal 2: Volatility Ratio Confirmation
            vol_regime = compute_volatility_regime(
                returns=event_returns,
                baseline_mean_abs=TS002_BASELINE_MEAN_ABS,
                stress_multiplier=STRESS_MULTIPLIER,
                bswan_multiplier=BSWAN_MULTIPLIER,
                stress_persistence_days=STRESS_PERSISTENCE_DAYS,
                bswan_persistence_days=BSWAN_PERSISTENCE_DAYS,
                rolling_window=10,
            )

            # Combined classification
            market_state = classify_market_state(
                current_entropy=0.0,
                baseline_entropy=0.0,
                vol_regime=vol_regime,
                perm_alarm=perm_alarm,
            )

            # Breach duration
            if market_state in ["BLACK_SWAN", "STRESS"]:
                breach_duration = vol_regime["stress_streak"]
            elif market_state == "WATCH":
                breach_duration = int(
                    perm_alarm["alarm_series"].sum()
                )
            else:
                breach_duration = 0

            # Peak volatility date
            vol_series = vol_regime["rolling_vol_series"]
            if len(vol_series.dropna()) > 0:
                peak_idx = vol_series.idxmax()
                peak_entropy_date = str(peak_idx.date())
            else:
                peak_entropy_date = cfg["event_end"]

            # WATCH maps to STRESS for Pydantic compatibility
            # EntropyResult Literal: NORMAL/STRESS/BLACK_SWAN
            pydantic_state = (
                "STRESS" if market_state == "WATCH"
                else market_state
            )

            # Build entropy_dict
            # entropy_delta = vol_ratio - 1.0
            # represents how far above baseline we are
            entropy_dict = {
                "ticker":               ticker,
                "primary_method":       "volatility_ratio",
                "confirmation_method":  "permutation_alarm",
                "baseline_entropy":     TS002_BASELINE_MEAN_ABS,
                "event_entropy":        float(vol_regime["vol_ratio"]),
                "entropy_delta":        float(vol_regime["vol_ratio"] - 1.0),
                "peak_entropy_date":    peak_entropy_date,
                "market_state":         pydantic_state,
                "breach_duration_days": breach_duration,
                "baseline_period_used": baseline_period_used,
            }

            logger.info(
                "[Layer3][Stage 2] Two-signal detection — "
                "vol_ratio=%.2fx alarm=%s state=%s breach=%dd",
                vol_regime["vol_ratio"],
                perm_alarm["alarm_active"],
                market_state,
                breach_duration,
            )

        except Exception as e:
            logger.error(
                "[Layer3][Stage 2] Detection failed "
                "ticker=%s event=%s to %s: %s",
                ticker, cfg["event_start"], cfg["event_end"], e,
            )
            raise RuntimeError(
                f"[Layer3][Stage 2] Detection failed "
                f"ticker={ticker} "
                f"event={cfg['event_start']} to "
                f"{cfg['event_end']}. Error: {e}"
            ) from e

        # STAGE 5 — Slippage
        try:
            jd_results = run_monte_carlo_slippage(
                S0=S0, mu=mu, sigma=sigma,
                n_paths=cfg["n_paths"],
                T=cfg["T"], dt=cfg["dt"],
                order_frac=cfg["order_frac"],
                eta=cfg["eta"],
                execution_window=cfg["execution_window"],
                mode="jump_diffusion",
                lambda_j=cfg["lambda_j"],
                mu_j=cfg["mu_j"],
                sigma_j=cfg["sigma_j"],
                base_seed=10_042,
            )
            
            gbm_results = run_monte_carlo_slippage(
                S0=S0, mu=mu, sigma=sigma,
                n_paths=cfg["n_paths"],
                T=cfg["T"], dt=cfg["dt"],
                order_frac=cfg["order_frac"],
                eta=cfg["eta"],
                execution_window=cfg["execution_window"],
                mode="gbm",
                base_seed=42,
            )

            gbm_mean = gbm_results["total_slippage"]["mean"]
            jd_mean  = jd_results["total_slippage"]["mean"]
            jd_gbm_ratio = jd_mean / max(gbm_mean, 0.01)

            slippage_dict = {
                "ticker": ticker,
                "mode": "jump_diffusion",
                "n_paths": int(cfg["n_paths"]),
                "total_slippage": jd_results["total_slippage"],
                "jd_gbm_ratio": round(float(jd_gbm_ratio), 4),
                "regime_breakdown": jd_results["regime_breakdown"],
                "entropy_state_at_execution": pydantic_state,
                "wandb_run_url": wandb_run_url,
            }

            logger.info("[Layer3][Stage 3] Slippage — mean=%.2f bps ratio=%.2fx gate=%s", jd_mean, jd_gbm_ratio, "PASSED" if jd_gbm_ratio > 1.5 else "FAILED")
        except Exception as e:
            raise RuntimeError(f"[Layer3][Stage 3] Slippage failed: {e}") from e

        # STAGE 6 — Assemble and return
        report = _assemble_layer3_report(
            simulation_dict,
            entropy_dict,
            slippage_dict,
            notes=f"Full pipeline run for {ticker}",
        )

        if use_wandb and _WANDB_AVAILABLE:
            try:
                wandb.log({
                    "layer3_risk_level":      report.overall_risk_level,
                    "layer3_action":          report.recommended_action,
                    "layer3_kurtosis_jd":     report.simulation.kurtosis_jd,
                    "layer3_vol_ratio":       vol_regime["vol_ratio"],
                    "layer3_perm_alarm":      perm_alarm["alarm_active"],
                    "layer3_market_state":    report.entropy.market_state,
                    "layer3_breach_days":     report.entropy.breach_duration_days,
                    "layer3_mean_slippage":   report.slippage.mean_bps,
                    "layer3_p99_slippage":    report.slippage.p99_bps,
                    "layer3_jd_gbm_ratio":    report.slippage.jd_gbm_ratio,
                    "layer3_validation":      report.slippage.validation_passed,
                })
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to log final WandB metrics: {e}")

        return report

    except Exception as e:
        if use_wandb and _WANDB_AVAILABLE:
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
        raise
