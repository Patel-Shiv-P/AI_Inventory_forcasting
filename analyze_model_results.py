
import pandas as pd

# Load evaluation results
df = pd.read_csv("data/model_evaluation_results.csv")

# Overall performance
print("\n--- OVERALL PERFORMANCE ---")
print("Combinations evaluated:", len(df))

print(
    "Combinations where Random Forest wins:",
    (df["random_forest_mae"] < df["baseline_mae"]).sum()
)

print(
    "Percentage where Random Forest wins:",
    round(
        (df["random_forest_mae"] < df["baseline_mae"]).mean() * 100,
        2
    ),
    "%"
)

# Find the 10 weakest cases by MAE improvement
print("\n--- 10 WORST IMPROVEMENTS ---")

worst = df.sort_values(
    "mae_improvement_percent"
).head(10)

print(worst.to_string(index=False))

# Find the 10 strongest cases
print("\n--- 10 BEST IMPROVEMENTS ---")

best = df.sort_values(
    "mae_improvement_percent",
    ascending=False
).head(10)

print(best.to_string(index=False))

# Store-level performance
print("\n--- PERFORMANCE BY STORE ---")

store_summary = df.groupby("store").agg(
    average_baseline_mae=("baseline_mae", "mean"),
    average_model_mae=("random_forest_mae", "mean"),
    average_model_rmse=("random_forest_rmse", "mean"),
    average_improvement=("mae_improvement_percent", "mean")
).round(2)

print(store_summary.to_string())

# Save the summaries
worst.to_csv("data/worst_model_results.csv", index=False)
best.to_csv("data/best_model_results.csv", index=False)
store_summary.to_csv("data/store_model_summary.csv")

print("\nAnalysis files saved in the data folder.")
