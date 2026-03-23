import pandas as pd
import numpy as np

from backtest.breach_detector import detect_breaches
from backtest.rolling_window import rolling_window

def confidence_optimizer(returns: pd.Series, window: int = 252)->dict:
    """
        for all confidence from 0.90 to 0.98
        this loop first gets the var series
        then it checks the breaches in that series
        it calculates the actual breach rate and compares it to expected breach rate
        and gets the best confidence level

        returns: pd.Series
        window : int = 252
        return dict
    """
    best_confidence = None
    # candidates looks like this
    # 0.9000000..1 0.910000..1 and so on
    candidates = np.arange(0.90,0.99,0.01)
    best_diff = float("inf")
    results = []
    
    

    for confidence in candidates:
        confidence = round(float(confidence),2)
        expected_breach_rate = round(1-confidence,2)

        var_series = rolling_window(returns, confidence, window)
        breach_info = detect_breaches(var_series,returns,confidence)

        actual_breach_rate = breach_info["breach_rate"]
        diff = abs(actual_breach_rate-expected_breach_rate)

        results.append(
            {
                "confidence_level": confidence,
                "expected_breach_rate": expected_breach_rate,
                "actual_breach_rate": actual_breach_rate,
                "difference": diff
            }
        )
        if diff<best_diff:
            best_diff = diff
            best_confidence = confidence
    # the for loop is lazy generator and wont run until it is called so next is used
    return {
            "optimal_confidence": best_confidence,
            "all_results": results,
            "optimal_breach_rate": next(r["actual_breach_rate"]for r in results if r["confidence_level"]==best_confidence)
        }
