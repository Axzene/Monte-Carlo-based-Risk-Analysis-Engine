"""
Parameter Estimation Module
============================
Estimates drift (μ) and volatility (σ) from a historical time series of
daily logarithmic returns, providing the statistical inputs required by
the Geometric Brownian Motion simulator.

Mathematical Background
-----------------------
Given a series of daily log-returns  r₁, r₂, …, rₙ  we estimate:

    μ̂  =  (1/n) Σ rᵢ          (sample mean  – daily drift)
    σ̂  =  √[ (1/(n-1)) Σ (rᵢ - μ̂)² ]   (sample std dev – daily vol)

These are *daily* parameters.  The GBM simulator uses them directly
because the simulation step size Δt = 1 day.

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Data contract – immutable container for estimated parameters
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GBMParameters:
    """Immutable container for the estimated GBM parameters.

    Attributes
    ----------
    mu : float
        Estimated daily drift (mean of log-returns).
    sigma : float
        Estimated daily volatility (std dev of log-returns).
    n_observations : int
        Number of return observations used in the estimation.
    """

    mu: float
    sigma: float
    n_observations: int

    def __repr__(self) -> str:
        return (
            f"GBMParameters(μ={self.mu:.8f}, σ={self.sigma:.8f}, "
            f"n={self.n_observations})"
        )


# ──────────────────────────────────────────────────────────────────────
# Pure function – parameter estimation
# ──────────────────────────────────────────────────────────────────────

def estimate_parameters(returns: pd.Series) -> GBMParameters:
    """Estimate daily drift and volatility from historical log-returns.

    This is a **pure function**: no side-effects, no mutation, and
    referentially transparent for any given input series.

    Parameters
    ----------
    returns : pd.Series
        A pandas Series of daily logarithmic returns.  Must contain at
        least 2 observations (needed for an unbiased σ estimate with
        ddof=1).

    Returns
    -------
    GBMParameters
        A frozen dataclass holding ``mu``, ``sigma``, and
        ``n_observations``.

    Raises
    ------
    ValueError
        If the input series is empty or contains fewer than 2
        observations.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(42)
    >>> fake_returns = pd.Series(rng.normal(0.0005, 0.02, size=252))
    >>> params = estimate_parameters(fake_returns)
    >>> params.n_observations
    252
    """

    # ── Input validation ──────────────────────────────────────────────
    if returns is None or len(returns) == 0:
        raise ValueError(
            "Cannot estimate parameters from an empty return series."
        )
    if len(returns) < 2:
        raise ValueError(
            "At least 2 observations are required for an unbiased "
            "volatility estimate (ddof=1)."
        )

    # ── Drop any NaN entries (defensive; upstream should clean data) ──
    clean: pd.Series = returns.dropna()
    if len(clean) < 2:
        raise ValueError(
            "Fewer than 2 non-NaN observations remain after cleaning."
        )

    # ── Estimation (vectorised, no loops) ─────────────────────────────
    mu: float = float(np.mean(clean.values))
    sigma: float = float(np.std(clean.values, ddof=1))

    return GBMParameters(mu=mu, sigma=sigma, n_observations=len(clean))
