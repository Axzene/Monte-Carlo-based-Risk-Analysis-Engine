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

    
