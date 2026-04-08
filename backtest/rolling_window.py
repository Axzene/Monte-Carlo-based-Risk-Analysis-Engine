import pandas as pd
import numpy as np

from risk.historical_var import compute_historical_var

def rolling_window(returns:pd.Series, confidence_level : float = 0.95, window :int = 252)->pd.Series:
    """
        Computes the rolling window of historical_var
        returns : pd.Series
        confidence_level:float = 0.95
        window: int = 252
        returns pd.Series of the rolling window
    """
    # if the total length of the returns is less than or equals to window
    # we wont be able to calculate the VaR
    if len(returns)<=window:
        raise ValueError("The given data is too small")
    var_series={}
    # window_returns has all the returns of 252 day period
    # var_series will look something like this
    # 2020-02-01 0.234(both arbitrary values)
    for i in range(window,len(returns)):
        window_returns = returns.iloc[i-window:i]
        var_series[returns.index[i]]= compute_historical_var(window_returns,confidence_level)
    return pd.Series(var_series,name = "rolling_var")