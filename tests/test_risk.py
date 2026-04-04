"""
Unit Tests – Risk Metrics (VaR & CVaR)
=======================================
Validates mathematical properties, edge cases, and the relationship
between VaR and CVaR.

Run:  ``pytest tests/test_risk.py -v``

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

import numpy as np
import pytest

from risk.metrics import compute_var, compute_cvar


# =====================================================================
#  Value at Risk
# =====================================================================

class TestVaR:
    """Tests for risk.metrics.compute_var."""

    def test_var_is_non_negative(self) -> None:
        """VaR (as a loss magnitude) should generally be ≥ 0 for a
        distribution centered near zero with enough negative mass."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.01, 0.03, size=100_000)
        var = compute_var(returns, confidence_level=0.95)
        assert var >= 0

    def test_higher_confidence_higher_var(self) -> None:
        """VaR at 99 % confidence must be ≥ VaR at 95 % confidence."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=100_000)
        var_95 = compute_var(returns, 0.95)
        var_99 = compute_var(returns, 0.99)
        assert var_99 >= var_95

    def test_var_known_value(self) -> None:
        """Deterministic check: for a simple array we can compute VaR
        by hand."""
        # Returns: [-0.10, -0.05, 0.00, 0.05, 0.10]
        # np.percentile linearly interpolates between data points, so the 5th
        # percentile of this 5-element array is −0.09 (midpoint of −0.10 and
        # −0.05), not exactly −0.10.  We allow abs=0.02 to stay robust while
        # still catching any real regression.
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


# =====================================================================
#  Conditional Value at Risk
# =====================================================================

class TestCVaR:
    """Tests for risk.metrics.compute_cvar."""

    def test_cvar_geq_var(self) -> None:
        """CVaR must always be ≥ VaR (CVaR is the *mean* of the tail,
        which is at least as extreme as the threshold)."""
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.03, size=100_000)

        for cl in [0.90, 0.95, 0.99]:
            var = compute_var(returns, cl)
            cvar = compute_cvar(returns, cl)
            assert cvar >= var, (
                f"CVaR ({cvar:.6f}) < VaR ({var:.6f}) at "
                f"confidence {cl} — violates mathematical bound!"
            )

    def test_cvar_is_non_negative(self) -> None:
        """Like VaR, CVaR should be ≥ 0 for a loss-centred distribution."""
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


# =====================================================================
#  Cross-metric consistency
# =====================================================================

class TestCrossMetricConsistency:
    """Integration-level tests that verify VaR and CVaR work together
    correctly on realistic simulated data."""

    def test_large_sample_consistency(self) -> None:
        """With 1 M samples from N(0, 0.02), the analytical 95 % VaR ≈
        0.02 × 1.645 ≈ 0.0329.  Check that Monte Carlo VaR is close."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0, 0.02, size=1_000_000)

        var_95 = compute_var(returns, 0.95)
        expected_var = 0.02 * 1.6449  # z_{0.95} for standard normal

        assert var_95 == pytest.approx(expected_var, rel=0.02), (
            f"MC VaR ({var_95:.6f}) deviates from analytical "
            f"({expected_var:.6f}) by > 2 %"
        )
