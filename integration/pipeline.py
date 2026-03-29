import pandas as pd
import numpy as np

from data.fetcher import fetch_price_data
from data.preprocessor import compute_log_returns
from config.config_loader import load_config
from risk.historical_var import compute_historical_var, compute_historical_cvar
from backtest.breach_detector import detect_breaches
from backtest.rolling_window import rolling_window
from backtest.confidence_optimizer import confidence_optimizer

def run_pipeline(config_path : str = None)->dict:
    """
     Runs the full risk analysis pipeline.

    config_path: optional path to a JSON config file
                 if None, uses default config

    Returns: dict with all results
    """
    # step 1 loag config
    config = load_config(config_path)
    ticker = config["ticker"]
    start = config["start_date"]
    end = config["end_date"]
    confidence = config["confidence_level"]
    window = 252
    
    print(f"Running pipeline for {ticker} from {start} to {end}")

    #step 2 fetch preprocess the data
    prices = fetch_price_data(ticker,start,end)
    returns = compute_log_returns(prices)

    print(f"Fetched {len(returns)} day of returns")

    #step 3 historical VaR
    hist_var = compute_historical_var(returns,confidence)
    hist_cvar = compute_historical_cvar(returns,confidence)

    print(f"Historical Var:{hist_var:.4f},CVaR:{hist_cvar:.4f}")

    #step 4 backtesting
    var_series = rolling_window(returns,confidence,window)
    breach_info = detect_breaches(var_series,returns,confidence)
    optimal = confidence_optimizer(returns, window)

    print(f"Breach rate {breach_info['breach_rate']:.4f}")
    print(f"Optimal confidence {optimal['optimal_confidence']:.4f}")

    #step 5 return the values as a dict
    return{
        "ticker":ticker,
        "hist_var":hist_var,
        "hist_cvar":hist_cvar,
        "breach_info":breach_info,
        "var_series":var_series,
        "returns":returns,
        "optimal_confidence":optimal
    }

