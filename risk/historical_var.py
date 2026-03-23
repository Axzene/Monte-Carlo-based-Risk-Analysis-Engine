import pandas as pd 
import numpy as np

def compute_historical_var(returns: pd.Series, confidence_level: float = 0.95) ->float:

    """ 
    Computes the historical VaR
    returns : series of log returns
    confidence_level = 0.95
    returns VaR as a float"""
    if returns.empty:
        raise ValueError("Returns series is empty")
    if not (0<confidence_level<1):
        raise ValueError("Confidence level must be within 0 and 1")
    # taking the 5th percentile of the returns(the worst 5% of days)
    # doing - as VaR is positive and var is supposed to be positive

    var = -np.percentile(returns,(1-confidence_level)*100)
    return float(var)

def compute_historical_cvar(returns:pd.Series, confidence_level:float=0.95)->float:
    """ 
    Computes the historical CVaR
    returns : series of log returns
    confidence_level = 0.95
    returns CVaR as a float"""
    if returns.empty:
        raise ValueError("Returns series is empty") 
    if not (0<confidence_level<1):
        raise ValueError("Confidence level must be within 0 and 1")
    
    losses = -returns
    var = compute_historical_var(returns,confidence_level)
    #cvar is average of the worst 5% of the days
    cvar = losses[losses>=var].mean()
    return float(cvar)
