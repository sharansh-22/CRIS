"""
manager.py — Convergence Manager: probabilistic temporal coordinator.

Coordinates the continuous probabilistic fields.
Does NOT classify markets, create hard states, or generate actions.
ONLY manages bounded influence, temporal smoothing, and emergent meta-dynamics
(stabilization, uncertainty, and signal coherence).
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from harvesters.macro.schema import (
    FastShockOutput,
    SlowStructuralOutput,
    DecayTrajectoryOutput,
    MetaDynamicsOutput,
    DominantField,
)

from .weighting import compute_dynamic_weights
from .smoothing import SmootherState, smooth_weights, update_state
from .arbitration import apply_partner_influence
from .transitions import EvolutionTracker, determine_dominant_field, update_tracker, compute_evolution_score
from .uncertainty import classify_uncertainty_state, compute_uncertainty_score

@dataclass
class ConvergenceState:
    """Persistent state for the convergence manager."""
    smoother: SmootherState = None
    evolution: EvolutionTracker = None
    # Continuous trackers replacing boolean states
    stabilization_strength: float = 0.0
    uncertainty_pressure: float = 0.0
    signal_coherence: float = 1.0

    def __post_init__(self):
        if self.smoother is None:
            self.smoother = SmootherState()
        if self.evolution is None:
            self.evolution = EvolutionTracker()

def run_convergence(
    fast: FastShockOutput,
    slow: SlowStructuralOutput,
    decay: DecayTrajectoryOutput,
    state: Optional[ConvergenceState] = None,
) -> tuple:
    if state is None:
        state = ConvergenceState()

    # ── 1. Bounded Partner Influence (max 5%) ──
    # Fast -> Slow -> Decay flow logic
    adj_fast_risk, adj_slow_risk, adj_decay_risk = apply_partner_influence(
        fast_risk=fast.shock_intensity,
        fast_confidence=fast.confidence,
        slow_risk=slow.structural_instability,
        slow_confidence=slow.confidence,
        decay_risk=decay.erosion_strength,
        decay_confidence=decay.confidence,
    )

    # ── 2. Dynamic Component Weighting ──
    raw_weights = compute_dynamic_weights(
        fast_confidence=fast.confidence,
        slow_confidence=slow.confidence,
        decay_confidence=decay.confidence,
        # Proxy persistence for the weighting logic
        fast_persistence=1, 
        slow_persistence=int(slow.stress_persistence * 15),
        decay_persistence=int(decay.rebound_failure * 30),
    )

    smoothed_weights = smooth_weights(raw_weights, state.smoother)
    w_fast, w_slow, w_decay = smoothed_weights

    # ── 3. Signal Coherence & Evolution ──
    risks = [adj_fast_risk, adj_slow_risk, adj_decay_risk]
    signal_coherence = float(1.0 - np.std(risks))
    
    overall_stress = (w_fast * adj_fast_risk + w_slow * adj_slow_risk + w_decay * adj_decay_risk)
    
    # Determine dominant field
    raw_dominant = determine_dominant_field(
        w_fast, w_slow, w_decay,
        adj_fast_risk, adj_slow_risk, adj_decay_risk
    )
    
    # Update evolution tracker
    state.evolution = update_tracker(
        state.evolution,
        overall_stress,
        raw_dominant,
        smoothed_weights
    )
    
    evolution_score = compute_evolution_score(state.evolution)

    # ── 4. Uncertainty Pressure ──
    # Arises when signals diverge strongly but all claim high confidence
    uncertainty = compute_uncertainty_score(
        adj_fast_risk, adj_slow_risk, adj_decay_risk,
        fast.confidence, slow.confidence, decay.confidence,
        evolution_score
    )
    
    # Final classification of interpretation state
    dominant_field = classify_uncertainty_state(
        raw_dominant,
        uncertainty,
        adj_fast_risk, adj_slow_risk, adj_decay_risk,
        fast.confidence, slow.confidence, decay.confidence,
        evolution_score
    )
    
    # Smooth uncertainty
    state.uncertainty_pressure = state.uncertainty_pressure * 0.8 + uncertainty * 0.2

    # ── 5. Stabilization Strength (Continuous Recovery) ──
    # Asymmetric healing: panic fast, heal slowly.
    if overall_stress < 0.3:
        state.stabilization_strength += 0.05
    else:
        state.stabilization_strength -= overall_stress * 0.20

    state.stabilization_strength = float(np.clip(state.stabilization_strength, 0.0, 1.0))
    state.signal_coherence = float(np.clip(signal_coherence, 0.0, 1.0))

    # Update smoother state
    state.smoother = update_state(
        state.smoother,
        weights=smoothed_weights,
        overall_risk=overall_stress,
        overall_confidence=(w_fast * fast.confidence + w_slow * slow.confidence + w_decay * decay.confidence),
        fast_risk=adj_fast_risk,
        slow_risk=adj_slow_risk,
        decay_risk=adj_decay_risk,
    )

    meta_output = MetaDynamicsOutput(
        stabilization_strength=round(state.stabilization_strength, 2),
        uncertainty_pressure=round(state.uncertainty_pressure, 2),
        signal_coherence=round(state.signal_coherence, 2),
        dominant_field=dominant_field
    )

    return meta_output, state
