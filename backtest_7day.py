
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv(
    "data/clean_sales.csv",
    parse_dates=["date"]
)

# Select one store and product
df = df[
    (df["store"] == 1) &
    (df["item"] == 1)
].sort_values("date").reset_index(drop=True)

FEATURES = [
    "sales_7_days_ago",
    "sales_14_days_ago",
    "sales_28_days_ago",
    "day_of_week",
    "month",
    "day_of_year"
]


def create_features(data):
    result = data.copy()

    result["sales_7_days_ago"] = result["sales"].shift(7)
    result["sales_14_days_ago"] = result["sales"].shift(14)
    result["sales_28_days_ago"] = result["sales"].shift(28)

    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["day_of_year"] = result["date"].dt.dayofyear

    return result


# Reserve the final 20% for testing
split_index = int(len(df) * 0.8)

history = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

actual_values = []
baseline_values = []
model_values = []
forecast_dates = []

# Predict in seven-day blocks
for start in range(0, len(test), 7):

    block = test.iloc[start:start + 7].copy()

    if len(block) < 7:
        break

    # Train only on data available before this block
    train_data = create_features(history).dropna()

    X_train = train_data[FEATURES]
    y_train = train_data["sales"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Keep actual historical sales for lag features.
    # Future predictions are appended for recursive forecasting.
    working = history[["date", "sales"]].copy()

    for _, row in block.iterrows():

        next_date = row["date"]

        # Calendar features
        day_of_week = next_date.dayofweek
        month = next_date.month
        day_of_year = next_date.dayofyear

        # Find sales from 7, 14 and 28 days earlier
        lag_values = {}

        for lag in [7, 14, 28]:
            target_date = next_date - pd.Timedelta(days=lag)

            previous = working.loc[
                working["date"] == target_date, "sales"
            ]

            lag_values[f"sales_{lag}_days_ago"] = (
                previous.iloc[0] if not previous.empty
                else np.nan
            )

        features = {
            **lag_values,
            "day_of_week": day_of_week,
            "month": month,
            "day_of_year": day_of_year
        }

        # If a lag is unavailable, stop rather than silently
        # generating an invalid prediction.
        if pd.isna(pd.Series(features)).any():
            raise ValueError(
                f"Missing lag value for date {next_date}"
            )

        X_next = pd.DataFrame([features])[FEATURES]

        prediction = max(0, model.predict(X_next)[0])

        # Seasonal baseline: same weekday one week earlier
        baseline = lag_values["sales_7_days_ago"]

        actual = row["sales"]

        forecast_dates.append(next_date)
        actual_values.append(actual)
        baseline_values.append(baseline)
        model_values.append(prediction)

        # Append prediction, not actual future sales.
        # This prevents future actual values leaking into forecasts.
        working = pd.concat(
            [
                working,
                pd.DataFrame({
                    "date": [next_date],
                    "sales": [prediction]
                })
            ],
            ignore_index=True
        )

    # Advance history using actual sales only after the
    # seven-day block has finished.
    history = pd.concat(
        [history, block[["date", "sales"]]],
        ignore_index=True
    )


# Calculate metrics
baseline_mae = mean_absolute_error(
    actual_values, baseline_values
)

model_mae = mean_absolute_error(
    actual_values, model_values
)

baseline_rmse = mean_squared_error(
    actual_values, baseline_values
) ** 0.5

model_rmse = mean_squared_error(
    actual_values, model_values
) ** 0.5

improvement = (
    (baseline_mae - model_mae) / baseline_mae * 100
    if baseline_mae > 0 else 0
)

print("\n--- SEVEN-DAY ROLLING BACKTEST ---")
print("Store: 1 | Item: 1")
print("Forecasted test days:", len(actual_values))

print("\nBaseline MAE:", round(baseline_mae, 2))
print("Random Forest MAE:", round(model_mae, 2))

print("Baseline RMSE:", round(baseline_rmse, 2))
print("Random Forest RMSE:", round(model_rmse, 2))

print("MAE improvement:", round(improvement, 2), "%")

# Save detailed predictions
results = pd.DataFrame({
    "date": forecast_dates,
    "actual_sales": actual_values,
    "baseline_prediction": baseline_values,
    "model_prediction": model_values
})

results.to_csv(
    "data/backtest_7day_results.csv",
    index=False
)

print("\nPredictions saved to data/backtest_7day_results.csv")
