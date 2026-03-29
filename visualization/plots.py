import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_returns(returns :pd.Series, ticker:str )->None:

    plt.figure(figsize=(12,4))
    plt.plot(returns.index, returns.values,color = "steelblue", linewidth=0.8)
    plt.axhline(0,color="black",linewidth=0.5,linestyle="--")
    plt.title(f"{ticker}- Daily log returns")
    plt.xlabel("Date")
    plt.ylabel("Log Returns")
    plt.tight_layout()
    plt.show()

def plot_var_vs_returns(returns:pd.Series,var_series:pd.Series,breach_info:dict,ticker:str)->None:
    
    aligned_returns,aligned_var = returns.align(join="inner")
    breach_dates = breach_info["breaches_date"]

    plt.figure(figsize=(12,5))
    plt.plot(aligned_returns.index,-aligned_returns.values,color="steelblue",linewidth = 0.8,label ="Actual loss")
    plt.plot(aligned_var.index,aligned_var.values, color = "orange", linewidth=1.2, label="Rolling VaR")

    #highlight the breach points
    for date in breach_dates:
        plt.axvline(x=date, color="red", alpha = 0.3, linewidth = 0.5)
    
    plt.title(f"{ticker}- Rolling Window vs Actual loss")
    plt.xlabel("Date")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_return_distribution(returns:pd.Series,hist_var:float,hist_cvar:float, ticker:str)->None:


    plt.figure(figsize=(10,5))
    plt.hist(returns.values, bins = 100, color = "steelblue", alpha = 0.7, edgecolor = "white")

    plt.axvline( -hist_var, color = "orange", linewidth = 1.5, label = f"VaR : {hist_var :.4f}")
    plt.axvline( -hist_cvar, color = "red", linewidth = 1.5, label = f"VaR : {hist_cvar :.4f}")
    plt.title(f"{ticker}- Returns Distribution")
    plt.xlabel("Log Returns")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_breach_rate (breach_info:dict,ticker:str)->None:
    actual = breach_info["breach_rate"]
    expected = breach_info["expected_berach_rate"]

    plt.figure(figsize= (6,4))
    plt.bar(["Actual Breach Rate, Expected Breach Rate"],[actual,expected],color = ["tomato","steelblue"])

    plt.title(f"{ticker} - Backtesting Breach Rate")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.show()
