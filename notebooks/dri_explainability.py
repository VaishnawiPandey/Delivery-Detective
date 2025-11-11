import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load dataset
df = pd.read_csv("dri_by_location.csv")

# Step 2: Clean string-style numerics
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('[', '', regex=False)
            .str.replace(']', '', regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 3: Keep numeric columns only
df = df.select_dtypes(include=[np.number]).dropna()

if 'DRI' not in df.columns:
    raise ValueError("❌ No 'DRI' column found in dri_by_location.csv!")

X = df.drop(columns=['DRI'])
y = df['DRI']

print(f"✅ Cleaned data shape: {X.shape}, Target mean: {y.mean():.3f}")

# Step 4: Train fresh model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    objective="reg:squarederror",
    verbosity=0
)
model.fit(X, y)

# 🧩 Step 5: Patch model internal parameters (fix base_score issue)
model.get_booster().set_attr(base_score=str(float(model.get_booster().attr("base_score") or 0.5)))

# Step 6: SHAP with safe Explainer fallback
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
except Exception as e:
    print("⚠️ TreeExplainer failed, switching to KernelExplainer:", e)
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

# Step 7: Visualize SHAP summary
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title("Feature Importance (SHAP Summary)")
plt.tight_layout()
plt.savefig("dri_shap_summary.png")
print("✅ SHAP summary plot saved as dri_shap_summary.png")

# Step 8: Save feature importances
# shap_values is an Explanation object → use .values
shap_vals_array = shap_values.values  # shape: (num_samples, num_features)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Mean |SHAP|": np.abs(shap_vals_array).mean(axis=0)
}).sort_values("Mean |SHAP|", ascending=False)

importance.to_csv("dri_feature_importance.csv", index=False)
print("✅ Feature importance saved to dri_feature_importance.csv")
