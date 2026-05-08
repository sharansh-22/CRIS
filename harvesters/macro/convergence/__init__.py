"""
convergence — Probabilistic coordination and stress_field arbitration.

This layer is a DISPATCHER, not an analyzer. It:
  - Receives outputs from fast, slow, and decay engines
  - Manages dynamic probability weights
  - Smooths transitions over time
  - Tracks stress_field evolution

It does NOT:
  - Process raw market data
  - Generate new signals
  - Become a master prediction model
"""
