import pandas as pd

def load_reviews(path="data/reviews/sample_reviews.csv"):
    return pd.read_csv(path)

def load_weather(path="data/weather/sample_weather.csv"):
    return pd.read_csv(path)
