"""
Geometric Brownian Motion (GBM) Simulation
============================================
Core mathematical engine that generates Monte Carlo price paths under
the GBM model using **pure NumPy vectorization** (zero Python loops).

Mathematical Formulation
------------------------
Under the risk-neutral GBM model with discrete daily steps (Δt = 1):

    S_t = S₀ · exp( (μ − σ²/2)·t  +  σ · W_t )

where  W_t = Σ_{i=1}^{t} Z_i ,  Z_i ~ N(0, 1)

Implementation: we construct the entire (horizon × num_paths) matrix
of standard-normal shocks in a single ``rng.standard_normal()`` call,
apply ``np.cumsum`` along the time axis for the Brownian motion, and
then broadcast the exponential formula across the full matrix.

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

import numpy as np


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    horizon: int,
    num_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate Monte Carlo price paths via Geometric Brownian Motion.

    This is a **pure function**: given the same inputs (including seed)
    it always returns the same output, with no side effects.

    Parameters
    ----------
    S0 : float
        Initial asset price (must be > 0).
    mu : float
        Daily drift (mean of log-returns).
    sigma : float
        Daily volatility (std dev of log-returns, must be > 0).
    horizon : int
        Number of trading days to simulate (must be ≥ 1).
    num_paths : int
        Number of independent Monte Carlo paths (must be ≥ 1).
    seed : int | None, optional
        Random seed for reproducibility.  Pass ``None`` for
        non-deterministic runs.

    Returns
    -------
    np.ndarray
        Price matrix of shape ``(horizon, num_paths)``.  Each column is
        one simulated price path starting from ``S0``.  Row 0 is the
        price at t=1, row -1 is the price at t=horizon.

    Raises
    ------
    ValueError
        If any input violates its domain constraints.

    Notes
    -----
    The implementation avoids **all** Python-level loops.  The entire
    computation is a chain of vectorised NumPy operations:

    1. Draw a ``(horizon, num_paths)`` matrix of i.i.d. N(0,1) shocks.
    2. Compute cumulative Brownian motion via ``np.cumsum`` (axis=0).
    3. Construct the deterministic drift array ``(μ − σ²/2) · t``.
    4. Broadcast-multiply to obtain the full price matrix.
    """

    # ── Input validation ──────────────────────────────────────────────
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")
    if num_paths < 1:
        raise ValueError(f"num_paths must be ≥ 1, got {num_paths}")

    # ── Random number generation (new-style NumPy Generator) ─────────
    rng: np.random.Generator = np.random.default_rng(seed)
    Z: np.ndarray = rng.standard_normal((horizon, num_paths))  # (H, N)

    # ── Brownian motion: cumulative sum of shocks along time axis ────
    W: np.ndarray = np.cumsum(Z, axis=0)  # (H, N)

    # ── Deterministic drift component: (μ − σ²/2) · t  ──────────────
    # t is a column vector [1, 2, …, horizon]
    t: np.ndarray = np.arange(1, horizon + 1).reshape(-1, 1)  # (H, 1)
    drift: np.ndarray = (mu - 0.5 * sigma**2) * t              # (H, 1) → broadcast

    # ── Diffusion component: σ · W_t  ────────────────────────────────
    diffusion: np.ndarray = sigma * W  # (H, N)

    # ── GBM formula: S_t = S₀ · exp(drift + diffusion) ──────────────
    price_paths: np.ndarray = S0 * np.exp(drift + diffusion)  # (H, N)

    return price_paths
