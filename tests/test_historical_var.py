import pytest
import pandas as pd
import numpy as np
from risk.historical_var import compute_historical_var, compute_historical_cvar

# create a simple known returns series for testing
@pytest.fixture
def sample_returns():
    np.random.seed(42)
    data = np.random.normal(0, 0.01, 1000)
    return pd.Series(data)

def test_var_is_positive(sample_returns):
    var = compute_historical_var(sample_returns)
    assert var > 0

def test_var_is_float(sample_returns):
    var = compute_historical_var(sample_returns)
    assert isinstance(var, float)

def test_cvar_greater_than_var(sample_returns):
    var = compute_historical_var(sample_returns)
    cvar = compute_historical_cvar(sample_returns)
    assert cvar > var

def test_var_empty_series():
    with pytest.raises(ValueError):
        compute_historical_var(pd.Series([], dtype=float))

def test_var_invalid_confidence(sample_returns):
    with pytest.raises(ValueError):
        compute_historical_var(sample_returns, confidence_level=1.5)

def test_higher_confidence_gives_higher_var(sample_returns):
    var_95 = compute_historical_var(sample_returns, 0.95)
    var_99 = compute_historical_var(sample_returns, 0.99)
    assert var_99 > var_95