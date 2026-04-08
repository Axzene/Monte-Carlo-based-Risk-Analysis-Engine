import pytest
import numpy as np
import pandas as pd

from data.fetcher import fetch_price_data
from data.preprocessor import compute_log_returns

#check if it is pd series

def test_fetch_price_data():
    prices = fetch_price_data("AAPL", "2022-01-01","2023-01-01")
    assert isinstance(prices,pd.Series)

def test_fetch_non_empty():
    prices = fetch_price_data("AAPL", "2022-01-01","2023-01-01")
    assert len(prices)>0

def test_fetch_invalid_ticker():
    with pytest.raises(ValueError):
        fetch_price_data("INVALIDTICKER123","2022-01-01","2023-01-01")

def test_log_returns():
    prices = fetch_price_data("AAPL", "2022-01-01","2023-01-01")
    returns = compute_log_returns(prices)
    assert isinstance(returns,pd.Series)

def test_log_returns_no_nan():
    prices = fetch_price_data("AAPL", "2022-01-01","2023-01-01")
    returns = compute_log_returns(prices)
    assert returns.isnull().sum()==0

def test_log_returns_length():
    prices = fetch_price_data("AAPL", "2022-01-01","2023-01-01")
    returns = compute_log_returns(prices)
    assert len(returns)==len(prices)-1