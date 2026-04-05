"""
Validation – Sanity Checks for Simulation Correctness
=======================================================
Diagnostic functions that verify mathematical properties expected from
a well-calibrated GBM simulation.

Key Property: Square-Root-of-Time Scaling
------------------------------------------
Under the i.i.d. normal returns assumption of GBM, Value at Risk
scales with the square root of the time horizon:

    VaR_T  ≈  VaR_1 × √T

Author : Nithik Deva
Course : CS1204 – Software Engineering
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def check_square_root_time_scaling(
    VaR_1_day: float,
    VaR_T_days: float,
    T: int,
    tolerance: float = 0.15,
) -> bool:
    """Verify the √T scaling property of Value at Risk.

    Checks whether  VaR_T ≈ VaR_1 × √T  holds within the given
    relative tolerance.  A necessary (but not sufficient) condition for
    a correctly calibrated GBM simulation.

    Parameters
    ----------
    VaR_1_day : float
        VaR from a 1-day horizon simulation (positive number).
    VaR_T_days : float
        VaR from a T-day horizon simulation (positive number).
    T : int
        The multi-day horizon (must be ≥ 1).
    tolerance : float, optional
        Maximum acceptable relative error, by default 0.15 (15 %).

    Returns
    -------
    bool
        ``True`` if the scaling relationship holds within tolerance.
    """

    if T < 1:
        raise ValueError(f"T must be ≥ 1, got {T}")
    if VaR_1_day <= 0:
        raise ValueError(f"VaR_1_day must be positive, got {VaR_1_day}")

    expected_VaR_T: float = VaR_1_day * math.sqrt(T)
    relative_error: float = abs(VaR_T_days - expected_VaR_T) / expected_VaR_T
    passed: bool = relative_error <= tolerance

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
    print(diagnostic)

    return passed
