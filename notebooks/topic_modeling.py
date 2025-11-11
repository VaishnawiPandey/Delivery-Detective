import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm

tqdm.pandas()

# === 1. Load merged reviews with sentiment ===
df = pd.read_csv("dataS/merged_raw_reviews.csv")
print(f"Data loaded: {df.shape} rows")

# === 2. Extract text for topic modeling ===
texts = df['cleaned_review'].astype(str).tolist()

# === 3. Create BERTopic model ===
topic_model = BERTopic(verbose=True)

# === 4. Fit model to text ===
topics, probs = topic_model.fit_transform(texts)

# === 5. Add topics to dataframe ===
df['topic'] = topics
df['topic_probability'] = [prob.max() if prob is not None else 0 for prob in probs]

# === 6. Save topic-enhanced dataset ===
output_path = "dataS/reviews_with_topics.csv"
df.to_csv(output_path, index=False)
print(f"✅ Topic modeling completed and saved to {output_path}")

# === 7. Optional: View top topics ===
print(topic_model.get_topic_info().head(10))
