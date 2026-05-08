"""
jumps.py — Merton Jump-Diffusion simulation logic for Fast Shock modeling.
"""

import numpy as np

def simulate_jumps(
    S0: float,
    mu: float,
    sigma: float,
    lambda_j: float = 2.0,
    mu_j: float = -0.15,
    sigma_j: float = 0.10,
    T: float = 1.0,
    dt: float = 1.0 / 252,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Simulate a Merton Jump-Diffusion price path."""
    rng = np.random.default_rng(seed)
    n_steps = round(T / dt)

    t = np.arange(n_steps + 1)
    S = np.zeros(n_steps + 1)
    S[0] = S0

    jump_compensator = lambda_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    drift = (mu - 0.5 * sigma**2 - jump_compensator) * dt
    diffusion = sigma * np.sqrt(dt)

    jump_indices: list[int] = []
    for i in range(1, n_steps + 1):
        Z = rng.standard_normal()
        diffusion_factor = np.exp(drift + diffusion * Z)
        n_jumps = rng.poisson(lambda_j * dt)
        jump_factor = 1.0
        if n_jumps > 0:
            J = rng.normal(mu_j, sigma_j, size=n_jumps).sum()
            jump_factor = np.exp(J)
            jump_indices.append(i)
        S[i] = S[i - 1] * diffusion_factor * jump_factor

    return t, S, jump_indices

def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float = 1.0,
    dt: float = 1.0 / 252,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a Geometric Brownian Motion price path."""
    rng = np.random.default_rng(seed)
    n_steps = round(T / dt)
    t = np.arange(n_steps + 1)
    S = np.zeros(n_steps + 1)
    S[0] = S0
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    for i in range(1, n_steps + 1):
        Z = rng.standard_normal()
        S[i] = S[i - 1] * np.exp(drift + diffusion * Z)
    return t, S
