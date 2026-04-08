import pandas as pd
import numpy as np

from visualization.plots import (
    plot_returns,
    plot_var_vs_returns,
    plot_breach_rate,
    plot_return_distribution
)

def run_dashboard(results:dict)->None:

    ticker = results["ticker"]
    hist_var = results["hist_var"]
    hist_cvar = results["hist_cvar"]
    returns = results["returns"]
    var_series = results["var_series"]
    breach_info = results["breach_info"]

    print(f"\n{'='*50}")
    print(f"  RISK ANALYSIS DASHBOARD — {ticker}")
    print(f"{'='*50}")
    print(f"  Historical VaR  : {hist_var:.4f}")
    print(f"  Historical CVaR : {hist_cvar:.4f}")
    print(f"  Breach Rate     : {breach_info['breach_rate']:.4f}")
    print(f"  Expected Rate   : {breach_info['expected_breach_rate']:.4f}")
    print(f"  Breach Count    : {breach_info['breach_count']}")
    print(f"  Total Days      : {breach_info['total_days']}")
    print(f"  Optimal Conf.   : {results['optimal_confidence']['optimal_confidence']}")
    print(f"{'='*50}\n")

    plot_returns(returns, ticker)
    plot_var_vs_returns(returns, var_series, breach_info, ticker)
    plot_return_distribution(returns, hist_var, hist_cvar, ticker)
    plot_breach_rate(breach_info, ticker)