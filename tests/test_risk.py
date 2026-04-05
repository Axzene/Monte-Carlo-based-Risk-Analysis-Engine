"""
Unit Tests – Risk Metrics (Monte Carlo VaR & CVaR)
===================================================
Run:  ``pytest tests/test_risk.py -v``

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

import numpy as np
import pytest

from risk.metrics import compute_var, compute_cvar


class TestVaR:

    def test_var_is_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.01, 0.03, size=100_000)
        var = compute_var(returns, confidence_level=0.95)
        assert var >= 0

    def test_higher_confidence_higher_var(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=100_000)
        var_95 = compute_var(returns, 0.95)
        var_99 = compute_var(returns, 0.99)
        assert var_99 >= var_95

    def test_var_known_value(self) -> None:
        # np.percentile linearly interpolates between data points, so the 5th
        # percentile of this 5-element array is −0.09 (midpoint of −0.10 and
        # −0.05), not exactly −0.10.  abs=0.02 stays robust to this while
        # still catching real regressions.
        returns = np.array([-0.10, -0.05, 0.00, 0.05, 0.10])
        var = compute_var(returns, confidence_level=0.95)
        assert var == pytest.approx(0.09, abs=0.02)

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compute_var(np.array([]), 0.95)

    def test_raises_on_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            compute_var(np.array([0.01, -0.02]), 1.5)
        with pytest.raises(ValueError, match="confidence_level"):
            compute_var(np.array([0.01, -0.02]), 0.0)


class TestCVaR:

    def test_cvar_geq_var(self) -> None:
        """CVaR must always be ≥ VaR."""
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.03, size=100_000)
        for cl in [0.90, 0.95, 0.99]:
            var = compute_var(returns, cl)
            cvar = compute_cvar(returns, cl)
            assert cvar >= var, (
                f"CVaR ({cvar:.6f}) < VaR ({var:.6f}) at confidence {cl}"
            )

    def test_cvar_is_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.01, 0.03, size=100_000)
        cvar = compute_cvar(returns, 0.95)
        assert cvar >= 0

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compute_cvar(np.array([]), 0.95)

    def test_raises_on_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            compute_cvar(np.array([0.01, -0.02]), -0.1)


class TestCrossMetricConsistency:

    def test_large_sample_consistency(self) -> None:
        """With 1 M samples from N(0, 0.02), MC VaR ≈ analytical 0.02×1.6449."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0, 0.02, size=1_000_000)
        var_95 = compute_var(returns, 0.95)
        expected_var = 0.02 * 1.6449
        assert var_95 == pytest.approx(expected_var, rel=0.02)
