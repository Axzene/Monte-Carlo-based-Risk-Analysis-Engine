import json
import os

DEFAULT_CONFIG = {
    "ticker" : "AAPL",
    "start_date" : "2020-01-01",
    "end_date" : "2024-01-01",
    "confidence_level" : 0.95,
    "time_horizon" : 1,
    "num_simulations" : 10000
}
def load_config(path : str = None)-> dict :
    """
    Loads configuration parameters
    If json file is provided then it is loaded from that
    else returns default config
    Returns dict of config
    """
    if path is None:
        return DEFAULT_CONFIG.copy()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open (path , "r") as f:
        config = json.load(f)
    # means if there are any missing keys then use the default values
    for key,value in DEFAULT_CONFIG.items():
        config.setdefault(key,value)
    return config
