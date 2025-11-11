import os
import pandas as pd
import sqlite3
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------
# INITIAL SETUP
# ---------------------------------------
download('stopwords')
download('wordnet')

DATA_DIR = "dataS"
RAW_CSV_PATHS = [
    os.path.join(DATA_DIR, "raw_reviews.csv"),   # From book scraping
    os.path.join(DATA_DIR, "Reviews.csv") # From Kaggle
]
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "processed_reviews.csv")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "delivery_detective.db")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove digits
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------------------------------------
# LOAD AND COMBINE DATA
# ---------------------------------------
dfs = []
for path in RAW_CSV_PATHS:
    if os.path.exists(path):
        print(f"📂 Loading: {path}")
        dfs.append(pd.read_csv(path))
    else:
        print(f"⚠️ File not found: {path}")

if not dfs:
    raise FileNotFoundError("No review datasets found in dataS/ directory!")

df = pd.concat(dfs, ignore_index=True)

# Standardize column names
expected_cols = ['title', 'review', 'rating', 'date']
for col in expected_cols:
    if col not in df.columns:
        print(f"⚠️ Missing column '{col}' — creating empty column.")
        df[col] = None

df = df[expected_cols]

# ---------------------------------------
# CLEAN TEXT
# ---------------------------------------
print("🧹 Cleaning review text...")
df["cleaned_review"] = df["review"].apply(clean_text)

# Normalize rating (convert text like '5 stars' → 5)
def normalize_rating(val):
    if isinstance(val, str):
        match = re.search(r"(\d+)", val)
        return int(match.group(1)) if match else None
    return val

df["rating"] = df["rating"].apply(normalize_rating)

# ---------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------
print("💬 Performing sentiment analysis...")
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_label(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_label"] = df["cleaned_review"].apply(get_sentiment_label)

# ---------------------------------------
# SAVE TO SQLITE
# ---------------------------------------
print("💾 Saving cleaned data to SQLite...")
conn = sqlite3.connect(SQLITE_DB_PATH)
df.to_sql("reviews_preprocessed", conn, if_exists="replace", index=False)
conn.close()

# ---------------------------------------
# SAVE CSV BACKUP
# ---------------------------------------
df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"✅ Preprocessing complete! Cleaned data saved to:\n - {OUTPUT_CSV_PATH}\n - SQLite: reviews_preprocessed table")
