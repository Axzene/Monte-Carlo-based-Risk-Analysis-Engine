"""
Simulation Engine – High-Level Orchestrator
=============================================
Provides a single entry-point that chains **parameter estimation** →
**GBM simulation** → **terminal return extraction**, producing the
1-D array of simulated cumulative returns consumed by the risk metrics
and Abhinav's backtesting framework.

Data Contract
-------------
    Input  : pd.Series of daily log-returns  (from data pipeline)
    Output : np.ndarray of shape (num_paths,) – simulated cumulative
             simple returns over the horizon: (S_T − S₀) / S₀

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.parameter_estimation import GBMParameters, estimate_parameters
from simulation.gbm import simulate_gbm


def run_simulation(
    returns: pd.Series,
    S0: float,
    horizon: int,
    num_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Run a full Monte Carlo simulation and return terminal returns.

    Orchestration flow:
        1. Estimate (μ, σ) from historical log-returns.
        2. Generate ``num_paths`` GBM price paths of length ``horizon``.
        3. Extract the **terminal simple return** for each path:
           ``r_i = (S_T^{(i)} − S₀) / S₀``

    Parameters
    ----------
    returns : pd.Series
        Historical daily log-returns from Abhinav's data pipeline.
    S0 : float
        Current (most recent) asset price.
    horizon : int
        Forward-looking simulation window in trading days.
    num_paths : int
        Number of Monte Carlo paths.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(num_paths,)`` containing the simulated
        cumulative simple return for each path over the full horizon.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> fake = pd.Series(rng.normal(0.0003, 0.015, 252))
    >>> rets = run_simulation(fake, S0=150.0, horizon=21, num_paths=5000, seed=42)
    >>> rets.shape
    (5000,)
    """

    # Step 1 – Parameter estimation
    params: GBMParameters = estimate_parameters(returns)

    # Step 2 – GBM simulation  →  (horizon, num_paths)
    price_paths: np.ndarray = simulate_gbm(
        S0=S0,
        mu=params.mu,
        sigma=params.sigma,
        horizon=horizon,
        num_paths=num_paths,
        seed=seed,
    )

    # Step 3 – Terminal simple returns  →  (num_paths,)
    terminal_prices: np.ndarray = price_paths[-1, :]       # last row
    simulated_returns: np.ndarray = (terminal_prices - S0) / S0

    return simulated_returns
