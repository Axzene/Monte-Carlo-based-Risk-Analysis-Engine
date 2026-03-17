import numpy as np 
import pandas as pd 

def compute_log_returns(prices:pd.Series) ->pd.Series:
    """
    Computes the log returns of the prices series
    price: pd.Series of the closing prices
    returns : pd.series of the log returns
    """
    if prices.isnull().any():
        prices = prices.dropna()
    # basically taking ln(105/100) 
    # we are doing shift 1 : ln(today/yesterday)
    #drop na show that the first one we will be dropped since there is no yesterday for that
    log_returns = np.log(prices /prices.shift(1)).dropna()
    log_returns.name = f"log_return_{prices.name}"
    return log_returns 