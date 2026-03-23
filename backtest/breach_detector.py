import pandas as pd
import numpy as np

def detect_breaches(var_series:pd.Series, returns:pd.Series,confidence_level:int = 0.95)->dict:


    aligned_returns, aligned_var = returns.align(var_series, join = "inner")

    breaches = -aligned_returns>aligned_var
    breach_dates = aligned_returns.index[breaches].to_list()
    len_breach = len(breaches)
    breach_count = breaches.sum()
    breach_rate = breach_count/len_breach

    return {
        "breaches dates":breach_dates,
        "breach rate": float(breach_rate),
        "expected breach rate": round(1-confidence_level,2),
        "total days":len_breach,
        "breach count": int(breach_count)

    }