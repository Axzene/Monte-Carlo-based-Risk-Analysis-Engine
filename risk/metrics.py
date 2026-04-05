"""
Risk Metrics Module – Monte Carlo VaR & CVaR
=============================================
Computes **Monte Carlo VaR** and **CVaR (Expected Shortfall)** from a
simulated distribution of portfolio / asset returns.

This module covers the *forward-looking* (simulation-based) risk
metrics.  For *historical* VaR/CVaR, see ``risk.historical_var``
(Abhinav's module).

Definitions
-----------
Value at Risk (VaR) at confidence level α:
    The (1 − α)-th percentile of the simulated return distribution.
    Reported as a *positive number* representing the potential loss.

    VaR_α = −Percentile(R, 100·(1 − α))

Conditional Value at Risk (CVaR / Expected Shortfall):
    The expected loss given that the loss exceeds VaR:

    CVaR_α = −E[ R | R ≤ −VaR_α ]

Design Notes
------------
Both functions are **pure**, stateless, and use only NumPy vectorised
operations – no Python loops.

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

import numpy as np


def compute_var(
    simulated_returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Compute Monte Carlo Value at Risk (VaR).

    Parameters
    ----------
    simulated_returns : np.ndarray
        1-D array of simulated cumulative returns (can be negative).
        This is the output of ``simulation.engine.run_simulation``.
    confidence_level : float
        Confidence level in (0, 1).  Typical values: 0.95, 0.99.

    Returns
    -------
    float
        VaR expressed as a **positive** number (magnitude of loss).

    Raises
    ------
    ValueError
        If inputs violate domain constraints.
    """

    _validate_inputs(simulated_returns, confidence_level)

    percentile_value: float = float(
        np.percentile(simulated_returns, (1 - confidence_level) * 100)
    )
    return -percentile_value


def compute_cvar(
    simulated_returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Compute Monte Carlo Conditional Value at Risk (CVaR / ES).

    CVaR (Expected Shortfall) is the average of all losses that exceed
    the VaR threshold, providing a measure of **tail risk** more
    conservative than VaR alone.  By construction, CVaR ≥ VaR.

    Parameters
    ----------
    simulated_returns : np.ndarray
        1-D array of simulated cumulative returns.
        This is the output of ``simulation.engine.run_simulation``.
    confidence_level : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        CVaR expressed as a **positive** number.
    """

    _validate_inputs(simulated_returns, confidence_level)

    var_threshold: float = float(
        np.percentile(simulated_returns, (1 - confidence_level) * 100)
    )

    tail_losses: np.ndarray = simulated_returns[simulated_returns <= var_threshold]

    if len(tail_losses) == 0:
        return -var_threshold

    return -float(np.mean(tail_losses))


# ──────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────

def _validate_inputs(
    simulated_returns: np.ndarray,
    confidence_level: float,
) -> None:
    """Shared input validation for VaR/CVaR."""
    if simulated_returns is None or len(simulated_returns) == 0:
        raise ValueError("simulated_returns must be a non-empty array.")
    if not (0 < confidence_level < 1):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )
