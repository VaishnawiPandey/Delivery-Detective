import pandas as pd
from textblob import TextBlob
from tqdm import tqdm
import os

tqdm.pandas()

# === 1. Load raw data (for merging later) ===
raw_df = pd.read_csv("dataS/raw_reviews.csv")
print(f"Raw data loaded: {raw_df.shape} rows")

# === 2. Load processed reviews ===
reviews_df = pd.read_csv("dataS/processed_reviews.csv")
print(f"Reviews data loaded: {reviews_df.shape} rows")

# === 3. Define sentiment function ===
def get_sentiment(text):
    if pd.isnull(text) or not isinstance(text, str):
        return 0
    return TextBlob(text).sentiment.polarity

# === 4. Apply sentiment analysis on 'cleaned_review' ===
reviews_df['sentiment_score'] = reviews_df['cleaned_review'].progress_apply(get_sentiment)

# === 5. Convert polarity to categorical label ===
def label_sentiment(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

reviews_df['sentiment_label'] = reviews_df['sentiment_score'].apply(label_sentiment)

# === 6. Save sentiment-enhanced reviews ===
reviews_output_path = "dataS/reviews_with_sentiment.csv"
reviews_df.to_csv(reviews_output_path, index=False)
print(f"✅ Sentiment analysis completed and saved to {reviews_output_path}")

# === 7. Optional: Merge with raw data for next step ===
merged_df = pd.merge(raw_df, reviews_df, on="title", how="left")  # or use 'order_id' if exists
merged_output_path = "dataS/merged_raw_reviews.csv"
merged_df.to_csv(merged_output_path, index=False)
print(f"✅ Merged dataset saved to {merged_output_path}")
