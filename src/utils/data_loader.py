import pandas as pd
import os

# Build paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_reviews(filename="reviews.csv"):
    """Load reviews dataset"""
    filepath = os.path.join(DATA_DIR, filename)
    return pd.read_csv(filepath)

def load_weather(filename="weather.csv"):
    """Load weather dataset"""
    filepath = os.path.join(DATA_DIR, filename)
    return pd.read_csv(filepath)
