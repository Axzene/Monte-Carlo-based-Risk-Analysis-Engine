"""
Risk Metrics Module – Value at Risk & Conditional Value at Risk
================================================================
Computes **Monte Carlo VaR** and **CVaR (Expected Shortfall)** from a
simulated distribution of portfolio / asset returns.

Definitions
-----------
Value at Risk (VaR) at confidence level α:
    The (1 − α)-th percentile of the simulated return distribution.
    Reported as a *positive number* representing the potential loss.

    VaR_α = −Percentile(R, 100·(1 − α))

Conditional Value at Risk (CVaR / Expected Shortfall):
    The expected loss given that the loss exceeds VaR:

    CVaR_α = −E[ R | R ≤ −VaR_α ]

    i.e., the mean of all returns that fall at or below the VaR
    threshold, sign-flipped to express as a positive loss.

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

    Examples
    --------
    >>> import numpy as np
    >>> rets = np.array([-0.05, -0.02, 0.01, 0.03, -0.08, 0.04])
    >>> var = compute_var(rets, 0.95)
    >>> var >= 0
    True
    """

    _validate_inputs(simulated_returns, confidence_level)

    # The (1-α) percentile gives the loss threshold
    percentile_value: float = float(
        np.percentile(simulated_returns, (1 - confidence_level) * 100)
    )

    # VaR is the magnitude of this threshold (positive number)
    return -percentile_value


def compute_cvar(
    simulated_returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Compute Monte Carlo Conditional Value at Risk (CVaR / ES).

    CVaR (Expected Shortfall) is the average of all losses that exceed
    the VaR threshold, providing a measure of **tail risk** that is
    more conservative than VaR alone.

    Parameters
    ----------
    simulated_returns : np.ndarray
        1-D array of simulated cumulative returns.
    confidence_level : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        CVaR expressed as a **positive** number.  By construction
        CVaR ≥ VaR for any given confidence level.

    Raises
    ------
    ValueError
        If inputs violate domain constraints.
    """

    _validate_inputs(simulated_returns, confidence_level)

    # Threshold = the (1-α) percentile (actual return value, not sign-flipped)
    var_threshold: float = float(
        np.percentile(simulated_returns, (1 - confidence_level) * 100)
    )

    # Select all returns at or below the threshold (tail losses)
    tail_losses: np.ndarray = simulated_returns[simulated_returns <= var_threshold]

    if len(tail_losses) == 0:
        # Edge case: no returns below the threshold (extremely unlikely
        # with continuous distributions, but handle defensively).
        return -var_threshold

    # CVaR = −mean(tail returns)  →  positive number
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
