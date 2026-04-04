"""
Validation – Sanity Checks for Simulation Correctness
=======================================================
Provides diagnostic functions that verify mathematical properties
expected from a well-calibrated GBM simulation.

Key Property: Square-Root-of-Time Scaling
------------------------------------------
Under the i.i.d. normal returns assumption of GBM, Value at Risk
scales with the square root of the time horizon:

    VaR_T  ≈  VaR_1 × √T

This module checks whether the simulated VaR at horizon T is
consistent with this theoretical relationship, within a configurable
tolerance.

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_square_root_time_scaling(
    VaR_1_day: float,
    VaR_T_days: float,
    T: int,
    tolerance: float = 0.15,
) -> bool:
    """Verify the √T scaling property of Value at Risk.

    Checks whether  VaR_T ≈ VaR_1 × √T  holds within the given
    relative tolerance.  This is a necessary (but not sufficient)
    condition for a correctly calibrated GBM simulation.

    Parameters
    ----------
    VaR_1_day : float
        VaR computed from a 1-day horizon simulation (positive number).
    VaR_T_days : float
        VaR computed from a T-day horizon simulation (positive number).
    T : int
        The multi-day horizon (must be ≥ 1).
    tolerance : float, optional
        Maximum acceptable relative error, by default 0.15 (15 %).
        Monte Carlo estimates have inherent sampling noise, so a tight
        tolerance (< 5 %) may cause false negatives unless path counts
        are very large (>100 k).

    Returns
    -------
    bool
        ``True`` if the scaling relationship holds within tolerance.

    Notes
    -----
    The diagnostic result is also emitted via ``logging.info`` /
    ``logging.warning`` so it can be captured by any logging handler
    attached to the root logger.

    Examples
    --------
    >>> check_square_root_time_scaling(0.02, 0.09, 21, tolerance=0.15)
    ...  # True or False depending on closeness
    """

    if T < 1:
        raise ValueError(f"T must be ≥ 1, got {T}")
    if VaR_1_day <= 0:
        raise ValueError(f"VaR_1_day must be positive, got {VaR_1_day}")

    import math

    expected_VaR_T: float = VaR_1_day * math.sqrt(T)
    relative_error: float = abs(VaR_T_days - expected_VaR_T) / expected_VaR_T

    passed: bool = relative_error <= tolerance

    # ── Diagnostic message ────────────────────────────────────────────
    diagnostic = (
        f"√T Scaling Check  |  T = {T} days\n"
        f"  VaR(1-day)       = {VaR_1_day:.6f}\n"
        f"  VaR({T}-day) SIM  = {VaR_T_days:.6f}\n"
        f"  VaR({T}-day) THY  = {expected_VaR_T:.6f}  "
        f"(VaR_1 × √{T})\n"
        f"  Relative Error   = {relative_error:.4%}\n"
        f"  Tolerance        = {tolerance:.2%}\n"
        f"  Result           = {'✅ PASS' if passed else '❌ FAIL'}"
    )

    if passed:
        logger.info(diagnostic)
    else:
        logger.warning(diagnostic)

    # Also print for immediate visibility during profiling / dev runs
    print(diagnostic)

    return passed
