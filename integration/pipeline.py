"""
Integration Pipeline
=====================
End-to-end pipeline that combines **Abhinav's data layer** with
**Nithik's quant core** into a single callable function.

Data Flow
---------
    data.fetcher.fetch_price_data
        → data.preprocessor.compute_log_returns
        → models.parameter_estimation.estimate_parameters   (Nithik)
        → simulation.engine.run_simulation                  (Nithik)
        → risk.metrics.compute_var / compute_cvar           (Nithik)
        → risk.historical_var.compute_historical_var        (Abhinav)
        → validation.sanity_checks (optional)               (Nithik)

This file is intentionally kept thin – it orchestrates existing
modules without duplicating logic.

Author  : Nithik Deva (integration)  |  Abhinav (data layer)
Course  : CS1204 – Software Engineering
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.fetcher import fetch_price_data
from data.preprocessor import compute_log_returns
from models.parameter_estimation import GBMParameters, estimate_parameters
from risk.historical_var import compute_historical_cvar, compute_historical_var
from risk.metrics import compute_cvar, compute_var
from simulation.engine import run_simulation

from config.config_loader import load_config
from backtest.breach_detector import detect_breaches
from backtest.rolling_window import rolling_window
from backtest.confidence_optimizer import confidence_optimizer


# ──────────────────────────────────────────────────────────────────────
# Output contract
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RiskReport:
    """Consolidated risk report merging historical and MC estimates.

    Attributes
    ----------
    ticker : str
    params : GBMParameters
        Estimated daily μ and σ.
    mc_var : float
        Monte Carlo VaR (forward-looking).
    mc_cvar : float
        Monte Carlo CVaR (forward-looking).
    hist_var : float
        Historical VaR (Abhinav's module).
    hist_cvar : float
        Historical CVaR (Abhinav's module).
    simulated_returns : np.ndarray
        1-D array of shape (num_paths,) – raw MC return distribution,
        ready for consumption by Abhinav's backtesting engine.
    """

    ticker: str
    params: GBMParameters
    mc_var: float
    mc_cvar: float
    hist_var: float
    hist_cvar: float
    simulated_returns: np.ndarray

    def summary(self) -> str:
        lines = [
            f"{'='*55}",
            f"  RISK REPORT  –  {self.ticker}",
            f"{'='*55}",
            f"  GBM Parameters   μ = {self.params.mu:.6f}   "
            f"σ = {self.params.sigma:.6f}",
            f"  Observations     n = {self.params.n_observations}",
            f"{'-'*55}",
            f"  {'Metric':<28}  {'Value':>10}",
            f"  {'-'*(28+12)}",
            f"  {'MC VaR (forward)':<28}  {self.mc_var:>10.4%}",
            f"  {'MC CVaR (forward)':<28}  {self.mc_cvar:>10.4%}",
            f"  {'Historical VaR':<28}  {self.hist_var:>10.4%}",
            f"  {'Historical CVaR':<28}  {self.hist_cvar:>10.4%}",
            f"{'='*55}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline entry-point
# ──────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    ticker: str,
    start: str,
    end: str,
    horizon: int = 21,
    num_paths: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = 42,
) -> RiskReport:
    """Run the complete end-to-end risk analysis pipeline.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. ``"AAPL"``.
    start : str
        Historical data start date ``"YYYY-MM-DD"``.
    end : str
        Historical data end date ``"YYYY-MM-DD"``.
    horizon : int
        Forward simulation window in trading days (default 21 = 1 month).
    num_paths : int
        Number of Monte Carlo paths (default 10 000).
    confidence_level : float
        VaR/CVaR confidence level, e.g. 0.95 or 0.99.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    RiskReport
        Frozen dataclass containing all risk metrics and the raw
        simulated return distribution.

    Examples
    --------
    >>> report = run_full_pipeline("AAPL", "2020-01-01", "2024-01-01")
    >>> print(report.summary())
    """

    # ── Step 1: Fetch historical prices (Abhinav) ─────────────────────
    prices: pd.Series = fetch_price_data(ticker, start=start, end=end)

    # ── Step 2: Compute log-returns (Abhinav) ─────────────────────────
    log_returns: pd.Series = compute_log_returns(prices)

    # ── Step 3: Estimate GBM parameters (Nithik) ──────────────────────
    params: GBMParameters = estimate_parameters(log_returns)

    # ── Step 4: Monte Carlo simulation (Nithik) ───────────────────────
    S0: float = float(prices.iloc[-1])          # most recent closing price
    simulated_returns: np.ndarray = run_simulation(
        returns=log_returns,
        S0=S0,
        horizon=horizon,
        num_paths=num_paths,
        seed=seed,
    )

    # ── Step 5: Monte Carlo risk metrics (Nithik) ─────────────────────
    mc_var: float = compute_var(simulated_returns, confidence_level)
    mc_cvar: float = compute_cvar(simulated_returns, confidence_level)

    # ── Step 6: Historical risk metrics (Abhinav) ────────────────────
    hist_var: float = compute_historical_var(log_returns, confidence_level)
    hist_cvar: float = compute_historical_cvar(log_returns, confidence_level)

    return RiskReport(
        ticker=ticker,
        params=params,
        mc_var=mc_var,
        mc_cvar=mc_cvar,
        hist_var=hist_var,
        hist_cvar=hist_cvar,
        simulated_returns=simulated_returns,
    )


def run_pipeline(config_path: str | None = None) -> dict:
    # step 1 loag config
    config = load_config(config_path)
    ticker = config["ticker"]
    start = config["start_date"]
    end = config["end_date"]
    confidence = config["confidence_level"]
    window = 252
    
    print(f"Running pipeline for {ticker} from {start} to {end}")
    #step 2 fetch preprocess the data
    prices = fetch_price_data(ticker, start, end)
    returns = compute_log_returns(prices)

    print(f"Fetched {len(returns)} day of returns")
    #step 3 historical VaR

    hist_var = compute_historical_var(returns, confidence)
    hist_cvar = compute_historical_cvar(returns, confidence)

    return {
        "ticker": ticker,
        "hist_var": hist_var,
        "hist_cvar": hist_cvar,
    }


# ──────────────────────────────────────────────────────────────────────
# Quick smoke-test:  python -m integration.pipeline
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = run_full_pipeline(
        ticker="AAPL",
        start="2020-01-01",
        end="2024-01-01",
        horizon=21,
        num_paths=10_000,
        confidence_level=0.95,
    )
    print(report.summary())
