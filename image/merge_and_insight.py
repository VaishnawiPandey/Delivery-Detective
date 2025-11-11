import pandas as pd

# ---------------- Paths ---------------- #
final_csv = r"C:\Users\91824\Desktop\Delivery Detective\dataS\final_features.csv"
pred_csv = r"C:\Users\91824\Desktop\Delivery Detective\full_test_predictions.csv"
output_csv = r"C:\Users\91824\Desktop\Delivery Detective\merged_insights.csv"

# ---------------- Load CSVs ---------------- #
df = pd.read_csv(final_csv)
pred_df = pd.read_csv(pred_csv)

print("✅ Final CSV columns:", df.columns)
print("✅ CNN predictions columns:", pred_df.columns)

# ---------------- Merge datasets ---------------- #
# Adjust merge key if needed
# If you have 'image_name' in final.csv that matches 'image' in CNN predictions, use that
# Otherwise, merge on '_id' if it corresponds to images
if 'image_name' in df.columns:
    merged_df = pd.merge(df, pred_df, left_on='image_name', right_on='image', how='left')
elif '_id' in df.columns:
    merged_df = pd.merge(df, pred_df, left_on='_id', right_on='image', how='left')
else:
    print("⚠️ No merge key found, predictions will be added separately.")
    merged_df = df.copy()
    merged_df['predicted_label'] = 'intact'
    merged_df['probability_damaged'] = 0

# ---------------- Fill missing predictions ---------------- #
merged_df['predicted_label'] = merged_df['predicted_label'].fillna('intact')
merged_df['probability_damaged'] = merged_df['probability_damaged'].fillna(0)

# ---------------- Basic Insights ---------------- #

# 1️⃣ Damaged vs Intact counts
damage_counts = merged_df['predicted_label'].value_counts()
print("\n📊 Damaged vs Intact packages:\n", damage_counts)

# 2️⃣ Average damage probability by sentiment
if 'sentiment_score' in merged_df.columns:
    avg_damage_by_sentiment = merged_df.groupby('sentiment_score')['probability_damaged'].mean()
    print("\n📊 Average damage probability by sentiment score:\n", avg_damage_by_sentiment)

# 3️⃣ Damage probability by location
if 'location' in merged_df.columns:
    avg_damage_by_location = merged_df.groupby('location')['probability_damaged'].mean().sort_values(ascending=False)
    print("\n📊 Average damage probability by location:\n", avg_damage_by_location)

# 4️⃣ Damage probability by season
if 'season' in merged_df.columns:
    avg_damage_by_season = merged_df.groupby('season')['probability_damaged'].mean()
    print("\n📊 Average damage probability by season:\n", avg_damage_by_season)

# ---------------- Save merged insights ---------------- #
merged_df.to_csv(output_csv, index=False)
print(f"\n✅ Merged insights saved to {output_csv}")
