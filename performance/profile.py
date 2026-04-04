"""
Performance Profiler
=====================
Benchmarks the Monte Carlo simulation engine across increasing path
counts to demonstrate compliance with **NFR-2**: the system must
handle 10 000+ paths efficiently.

Usage
-----
Run directly::

    python -m performance.profile

The script synthesises fake historical returns (so it can run
standalone without a live data feed), then times the simulation engine
for 10 000, 50 000, and 100 000 paths, printing a formatted report.

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from simulation.engine import run_simulation


def _generate_synthetic_returns(
    n_days: int = 252,
    mu: float = 0.0005,
    sigma: float = 0.02,
    seed: int = 0,
) -> pd.Series:
    """Create a synthetic series of daily log-returns for benchmarking.

    Parameters
    ----------
    n_days : int
        Number of daily observations.
    mu, sigma : float
        Parameters for the normal distribution.
    seed : int
        Random seed.

    Returns
    -------
    pd.Series
        Synthetic log-returns.
    """
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, size=n_days), name="log_return")


def run_benchmark(
    path_counts: list[int] | None = None,
    horizon: int = 21,
    S0: float = 150.0,
    seed: int = 42,
) -> dict[int, float]:
    """Benchmark the simulation engine across multiple path counts.

    Parameters
    ----------
    path_counts : list[int] | None
        List of path counts to benchmark.  Defaults to
        ``[10_000, 50_000, 100_000]``.
    horizon : int
        Simulation horizon in days.
    S0 : float
        Initial price.
    seed : int
        Random seed passed to the engine.

    Returns
    -------
    dict[int, float]
        Mapping of ``{num_paths: elapsed_seconds}``.
    """

    if path_counts is None:
        path_counts = [10_000, 50_000, 100_000]

    returns: pd.Series = _generate_synthetic_returns()
    results: dict[int, float] = {}

    # ── Header ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MONTE CARLO SIMULATION – PERFORMANCE REPORT")
    print("=" * 60)
    print(f"  Horizon : {horizon} days   |   S₀ : ${S0:.2f}")
    print(f"  Seed    : {seed}           |   Returns : {len(returns)} obs")
    print("-" * 60)
    print(f"  {'Paths':>10}  │  {'Time (s)':>10}  │  {'Paths/sec':>12}")
    print("-" * 60)

    for n in path_counts:
        t_start: float = time.perf_counter()
        _ = run_simulation(
            returns=returns,
            S0=S0,
            horizon=horizon,
            num_paths=n,
            seed=seed,
        )
        elapsed: float = time.perf_counter() - t_start
        results[n] = elapsed

        throughput: float = n / elapsed if elapsed > 0 else float("inf")
        print(f"  {n:>10,}  │  {elapsed:>10.4f}  │  {throughput:>12,.0f}")

    print("=" * 60)
    print("  ✅  All benchmarks completed successfully.")
    print("=" * 60 + "\n")

    return results


# ──────────────────────────────────────────────────────────────────────
# CLI entry-point:  python -m performance.profile
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_benchmark()
