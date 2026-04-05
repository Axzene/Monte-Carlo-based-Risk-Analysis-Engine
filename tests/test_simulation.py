"""
Unit Tests – Parameter Estimation & GBM Simulation
====================================================
Run:  ``pytest tests/test_simulation.py -v``

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.parameter_estimation import GBMParameters, estimate_parameters
from simulation.gbm import simulate_gbm
from simulation.engine import run_simulation


class TestParameterEstimation:

    def test_raises_on_empty_series(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            estimate_parameters(pd.Series([], dtype=float))

    def test_raises_on_single_observation(self) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            estimate_parameters(pd.Series([0.01]))

    def test_raises_on_none(self) -> None:
        with pytest.raises(ValueError):
            estimate_parameters(None)  # type: ignore[arg-type]

    def test_known_values(self) -> None:
        data = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        params = estimate_parameters(data)
        assert params.mu == pytest.approx(0.03, abs=1e-10)
        assert params.sigma == pytest.approx(np.std(data.values, ddof=1), abs=1e-10)
        assert params.n_observations == 5

    def test_returns_frozen_dataclass(self) -> None:
        params = estimate_parameters(pd.Series([0.01, 0.02, 0.03]))
        with pytest.raises(AttributeError):
            params.mu = 999  # type: ignore[misc]

    def test_handles_nan_values(self) -> None:
        data = pd.Series([0.01, float("nan"), 0.03, float("nan"), 0.05])
        params = estimate_parameters(data)
        assert params.n_observations == 3


class TestGBMSimulation:

    @pytest.mark.parametrize(
        "horizon, num_paths",
        [(1, 1), (10, 100), (252, 10_000), (21, 50_000)],
    )
    def test_output_shape(self, horizon: int, num_paths: int) -> None:
        paths = simulate_gbm(
            S0=100.0, mu=0.001, sigma=0.02,
            horizon=horizon, num_paths=num_paths, seed=42,
        )
        assert paths.shape == (horizon, num_paths)

    def test_all_prices_positive(self) -> None:
        paths = simulate_gbm(
            S0=100.0, mu=-0.01, sigma=0.05,
            horizon=252, num_paths=1000, seed=7,
        )
        assert np.all(paths > 0)

    def test_deterministic_with_seed(self) -> None:
        kwargs = dict(S0=100.0, mu=0.001, sigma=0.02, horizon=10, num_paths=50)
        a = simulate_gbm(**kwargs, seed=123)
        b = simulate_gbm(**kwargs, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        kwargs = dict(S0=100.0, mu=0.001, sigma=0.02, horizon=10, num_paths=50)
        a = simulate_gbm(**kwargs, seed=1)
        b = simulate_gbm(**kwargs, seed=2)
        assert not np.array_equal(a, b)

    def test_raises_on_negative_S0(self) -> None:
        with pytest.raises(ValueError, match="S0"):
            simulate_gbm(S0=-10, mu=0.001, sigma=0.02, horizon=10, num_paths=10)

    def test_raises_on_zero_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            simulate_gbm(S0=100, mu=0.001, sigma=0, horizon=10, num_paths=10)

    def test_raises_on_zero_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            simulate_gbm(S0=100, mu=0.001, sigma=0.02, horizon=0, num_paths=10)


class TestSimulationEngine:

    def test_output_shape(self) -> None:
        returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.02, 252))
        result = run_simulation(returns, S0=150.0, horizon=21, num_paths=500, seed=42)
        assert result.shape == (500,)

    def test_deterministic_with_seed(self) -> None:
        returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.02, 252))
        a = run_simulation(returns, S0=150.0, horizon=21, num_paths=100, seed=99)
        b = run_simulation(returns, S0=150.0, horizon=21, num_paths=100, seed=99)
        np.testing.assert_array_equal(a, b)
