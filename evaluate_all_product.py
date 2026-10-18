
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load historical data
df = pd.read_csv(
    "data/clean_sales.csv",
    parse_dates=["date"]
)

results = []

# Evaluate each store-product combination
for (store_id, item_id), group in df.groupby(["store", "item"]):

    group = group.sort_values("date").reset_index(drop=True)

    # Historical sales features
    group["sales_7_days_ago"] = group["sales"].shift(7)
    group["sales_14_days_ago"] = group["sales"].shift(14)
    group["sales_28_days_ago"] = group["sales"].shift(28)

    group["day_of_week"] = group["date"].dt.dayofweek
    group["month"] = group["date"].dt.month
    group["day_of_year"] = group["date"].dt.dayofyear

    group = group.dropna().reset_index(drop=True)

    features = [
        "sales_7_days_ago",
        "sales_14_days_ago",
        "sales_28_days_ago",
        "day_of_week",
        "month",
        "day_of_year"
    ]

    X = group[features]
    y = group["sales"]

    # Chronological split
    split_index = int(len(group) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # Baseline predictions
    baseline_predictions = X_test["sales_7_days_ago"]

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate metrics
    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    model_mae = mean_absolute_error(
        y_test, predictions
    )

    model_rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    results.append({
        "store": store_id,
        "item": item_id,
        "test_rows": len(X_test),
        "baseline_mae": round(baseline_mae, 2),
        "random_forest_mae": round(model_mae, 2),
        "random_forest_rmse": round(model_rmse, 2),
        "mae_improvement_percent": round(
            (baseline_mae - model_mae) / baseline_mae * 100,
            2
        ) if baseline_mae > 0 else None
    })

# Save and display results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "data/model_evaluation_results.csv",
    index=False
)

print("\n--- MULTI-PRODUCT MODEL EVALUATION ---")
print("Store-product combinations evaluated:", len(results_df))

print("\nFirst 10 results:")
print(results_df.head(10).to_string(index=False))

print("\nAverage Random Forest MAE:",
      round(results_df["random_forest_mae"].mean(), 2))

print("Average Random Forest RMSE:",
      round(results_df["random_forest_rmse"].mean(), 2))

print("Average baseline MAE:",
      round(results_df["baseline_mae"].mean(), 2))

print("\nResults saved to data/model_evaluation_results.csv")
