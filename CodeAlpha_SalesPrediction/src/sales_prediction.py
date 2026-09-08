"""
TASK 4: Sales Prediction using Python
----------------------------------------
Goal: Predict future sales based on advertising spend across different
platforms (TV, Radio, Newspaper), and analyze how changes in advertising
impact sales outcomes to deliver actionable marketing insights.

Works with the Kaggle "Advertising.csv" dataset (bumba5341/advertisingcsv),
which has columns: TV, Radio, Newspaper, Sales (ad spend in thousands of
dollars per platform, sales in thousands of units).

Usage:
    import kagglehub
    path = kagglehub.dataset_download("bumba5341/advertisingcsv")
    # Point CSV_PATH below at the CSV inside that folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

# -----------------------------------------------------------------------
# 1. LOAD THE DATASET
# -----------------------------------------------------------------------
# After downloading with kagglehub, set this to the actual CSV file, e.g.:
#   CSV_PATH = f"{path}/Advertising.csv"
CSV_PATH = "Advertising.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset from {CSV_PATH}")
except FileNotFoundError:
    # Fallback so the pipeline is always runnable end-to-end even before
    # you've pointed CSV_PATH at your downloaded file. Replace with the
    # real file for your actual submission.
    print(f"'{CSV_PATH}' not found — generating a small synthetic sample "
          f"so the pipeline can be demonstrated. Point CSV_PATH at your "
          f"real downloaded CSV to train on the actual data.")
    rng = np.random.default_rng(42)
    n = 200
    tv = rng.uniform(0, 300, n)
    radio = rng.uniform(0, 50, n)
    newspaper = rng.uniform(0, 100, n)
    sales = (7 + 0.045 * tv + 0.19 * radio + 0.01 * newspaper
             + rng.normal(0, 1.5, n)).clip(1, None)
    df = pd.DataFrame({"TV": tv.round(1), "Radio": radio.round(1),
                        "Newspaper": newspaper.round(1), "Sales": sales.round(2)})

# Drop stray index column some CSV exports include
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nSummary statistics:")
print(df.describe())

# -----------------------------------------------------------------------
# 2. CLEANING & FEATURE PREP
# -----------------------------------------------------------------------
df = df.dropna()
feature_cols = [c for c in ["TV", "Radio", "Newspaper"] if c in df.columns]

# -----------------------------------------------------------------------
# 3. EDA — correlation + advertising vs sales scatter plots
# -----------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Ad Spend and Sales")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, len(feature_cols), figsize=(5 * len(feature_cols), 4))
for ax, col in zip(axes, feature_cols):
    ax.scatter(df[col], df["Sales"], alpha=0.6, color="#4C72B0")
    ax.set_xlabel(f"{col} spend")
    ax.set_ylabel("Sales")
    ax.set_title(f"{col} vs Sales")
plt.tight_layout()
plt.savefig("spend_vs_sales.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------
X = df[feature_cols]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------
# 5. TRAIN SEVERAL REGRESSION MODELS
# -----------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}
best_name, best_r2, best_model, best_preds = None, -np.inf, None, None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2:  {r2:.3f}")

    if r2 > best_r2:
        best_name, best_r2, best_model, best_preds = name, r2, model, preds

# -----------------------------------------------------------------------
# 6. COMPARE MODELS
# -----------------------------------------------------------------------
plt.figure(figsize=(7, 4))
r2_values = [results[m]["R2"] for m in models]
bars = plt.bar(models.keys(), r2_values, color="#55A868")
plt.ylabel("R^2 score")
plt.title("Model Comparison — Sales Prediction")
plt.xticks(rotation=15)
for bar, val in zip(bars, r2_values):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 7. ACTUAL VS PREDICTED FOR BEST MODEL
# -----------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color="#DD8452")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, "--", color="gray")
plt.xlabel("Actual sales")
plt.ylabel("Predicted sales")
plt.title(f"Actual vs Predicted — {best_name} (R2={best_r2:.3f})")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 8. WHICH PLATFORM DRIVES SALES MOST? (feature importance / coefficients)
# -----------------------------------------------------------------------
if isinstance(best_model, RandomForestRegressor):
    impact = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
    impact_label = "Feature importance"
else:
    impact = pd.Series(best_model.coef_, index=feature_cols).sort_values()
    impact_label = "Standardized coefficient"

plt.figure(figsize=(7, 4))
impact.plot(kind="barh", color="#8172B2")
plt.title(f"Advertising Platform Impact on Sales — {best_name}")
plt.xlabel(impact_label)
plt.tight_layout()
plt.savefig("platform_impact.png", dpi=150)
plt.close()

top_platform = impact.abs().idxmax()

# -----------------------------------------------------------------------
# 9. KEY INSIGHTS SUMMARY
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"- Best model: {best_name} (R^2 = {best_r2:.3f})")
print(f"- Platform with the strongest effect on sales: {top_platform}")
print(f"- Correlation of each platform with sales:\n{df[feature_cols + ['Sales']].corr()['Sales'][:-1]}")
print("\nSaved plots: correlation_heatmap.png, spend_vs_sales.png, model_comparison.png, "
      "actual_vs_predicted.png, platform_impact.png")
