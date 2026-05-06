"""
config.py — Centralized configuration for Layer 3 Probabilistic Interpretation.
Contains all thresholds, constants, and hyperparameters for all subfields.
"""

# ──────────────────────────────────────────────────────────
#  SHARED
# ──────────────────────────────────────────────────────────

DEFAULT_LOOKBACK_WINDOW = 252  # 1 trading year
ROLLING_WINDOW_SMALL = 10
ROLLING_WINDOW_MEDIUM = 20
ROLLING_WINDOW_LARGE = 30

# ──────────────────────────────────────────────────────────
#  SLOW STRUCTURAL THRESHOLDS
# ──────────────────────────────────────────────────────────

ENTROPY_STRESS_THRESHOLD = 0.15
ENTROPY_CRITICAL_THRESHOLD = 0.30

VOL_STRESS_MULTIPLIER = 1.5
VOL_CRITICAL_MULTIPLIER = 3.0

STRESS_PERSISTENCE_DAYS = 10
CRITICAL_PERSISTENCE_DAYS = 5

# ──────────────────────────────────────────────────────────
#  FAST SHOCK THRESHOLDS
# ──────────────────────────────────────────────────────────

PERM_ALARM_DROP_THRESHOLD = 0.05
JUMP_VALIDATION_RATIO = 1.5
FAST_VOL_SPIKE_THRESHOLD = 2.0
FAST_VOL_EXTREME_THRESHOLD = 4.0

# ──────────────────────────────────────────────────────────
#  TRAJECTORY ENGINE THRESHOLDS
# ──────────────────────────────────────────────────────────

DECAY_TREND_WINDOW = 60
DECAY_LONG_TREND_WINDOW = 120
DECAY_MOMENTUM_WINDOW = 30
DECAY_DRAWDOWN_MILD = 0.10
DECAY_DRAWDOWN_MODERATE = 0.20
DECAY_DRAWDOWN_SEVERE = 0.35
DECAY_PERSISTENCE_THRESHOLD = 30
DECAY_MIN_CONFIDENCE_DAYS = 15

# LSTM configuration
LSTM_HIDDEN_DIM = 32
LSTM_NUM_LAYERS = 1
LSTM_SEQUENCE_LENGTH = 60
LSTM_FEATURE_DIM = 6
LSTM_LEARNING_RATE = 0.001
LSTM_INFLUENCE_CAP = 0.10  # LSTM is advisory, not dominant

# ──────────────────────────────────────────────────────────
#  CONVERGENCE LAYER
# ──────────────────────────────────────────────────────────

# Tighter coupling: max 5% partner influence
PARTNER_INFLUENCE_CAP = 0.05

# Smoothing: EMA decay factor
WEIGHT_SMOOTHING_ALPHA = 0.15

# Weight base priors
WEIGHT_PRIOR_FAST = 0.40
WEIGHT_PRIOR_SLOW = 0.35
WEIGHT_PRIOR_DECAY = 0.25

# Persistence-based weight transition rates
PERSISTENCE_SLOW_BOOST_RATE = 0.003
PERSISTENCE_DECAY_BOOST_RATE = 0.002

# Confidence floor
CONFIDENCE_FLOOR = 0.05

# ──────────────────────────────────────────────────────────
#  META DYNAMICS (Continuous Recovery & Uncertainty)
# ──────────────────────────────────────────────────────────

# Continuous exponential relaxation for stabilization
STABILIZATION_GROWTH_RATE = 0.05  # Slow psychological healing
STABILIZATION_SHOCK_PENALTY = 0.20 # Fast panic

# Uncertainty scaling
UNCERTAINTY_CONFLICT_SCALE = 1.2
UNCERTAINTY_NOISE_PENALTY = 0.1

# ──────────────────────────────────────────────────────────
#  SIMULATION DEFAULTS
# ──────────────────────────────────────────────────────────

MC_PATHS = 2000
SLIPPAGE_P99_LIQUIDATE_THRESHOLD = 1000

# ──────────────────────────────────────────────────────────
#  DATA MAPPING
# ──────────────────────────────────────────────────────────

SPY_BASELINE_VOL = 0.00714
SPY_BASELINE_PERM_ENTROPY = 0.95
ENTROPY_SAMPLE_BASELINE = 0.50
