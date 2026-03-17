import yfinance as yf
import pandas as pd 

def fetch_price_data(ticker:str, start:str, end:str) ->pd.Series:
    """ 
    Downloads historical closing prices for a stock
    ticker : stock symbol for e.g "AAPL"
    start : start date in "YYYY-MM-DD"
    end : end date in "YYYY-MM-DD"
    Returns pd.Series of closing prices
    """
    #pulls daily closing prices
    data = yf.download(ticker , start = start, end = end, auto_adjust = True)

    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    #ensures it is a series not a dataframe
    # basically converts nto series if it is only one column
    prices = data['Close'].squeeze()
    prices.name = ticker
    return prices