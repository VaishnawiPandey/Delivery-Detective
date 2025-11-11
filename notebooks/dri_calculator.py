# # analysis/dri_calculator.py
# import pandas as pd

# df = pd.read_csv("merged_insights.csv")

# # Normalize sub-scores
# df["SentimentIndex"] = (df["sentiment_score"] - df["sentiment_score"].min()) / (df["sentiment_score"].max() - df["sentiment_score"].min())
# df["WeatherIndex"] = 1 - (df["weather_severity_index"] / df["weather_severity_index"].max())
# df["DamageRiskIndex"] = 1 - df["probability_damaged"]

# # Weighted average
# df["DeliveryResilienceIndex"] = (
#     0.4 * df["SentimentIndex"] +
#     0.3 * df["WeatherIndex"] +
#     0.3 * df["DamageRiskIndex"]
# )

# df.to_csv("delivery_resilience.csv", index=False)
# print("✅ Delivery Resilience Index saved to delivery_resilience.csv")

# dri/dri_from_separate_sources.py
import pandas as pd

# Load review-weather dataset
reviews = pd.read_csv("final_features.csv")

# Load CNN image predictions
cnn = pd.read_csv("full_test_predictions.csv")

# --- Compute average normalized scores ---
sentiment_mean = (reviews["sentiment_score"].mean() + 5) / 10  # normalize roughly between 0–1
weather_mean = 1 - (reviews["weather_severity_index"].mean() / reviews["weather_severity_index"].max())
if "probability_damaged" in cnn.columns:
    damage_mean = 1 - cnn["probability_damaged"].mean()
else:
    damage_mean = 1  # if all intact or missing column

# --- Compute weighted DRI ---
dri = (0.4 * sentiment_mean) + (0.3 * weather_mean) + (0.3 * damage_mean)

print(f"📦 Average Sentiment Index: {sentiment_mean:.3f}")
print(f"🌦 Weather Stability Index: {weather_mean:.3f}")
print(f"📸 Packaging Integrity Index: {damage_mean:.3f}")
print(f"✅ Delivery Resilience Index (DRI): {dri:.3f}")

# dri/dri_grouped.py
grouped = reviews.groupby("location").agg({
    "sentiment_score": "mean",
    "weather_severity_index": "mean"
}).reset_index()

grouped["SentimentIndex"] = (grouped["sentiment_score"] + 5) / 10
grouped["WeatherIndex"] = 1 - (grouped["weather_severity_index"] / reviews["weather_severity_index"].max())
grouped["DamageIndex"] = damage_mean  # global CNN-based mean
grouped["DRI"] = 0.4*grouped["SentimentIndex"] + 0.3*grouped["WeatherIndex"] + 0.3*grouped["DamageIndex"]

grouped.to_csv("dri_by_location.csv", index=False)
print("✅ DRI by location saved to dri_by_location.csv")

