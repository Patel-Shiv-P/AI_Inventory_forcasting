
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 1. Load prepared sales data
df = pd.read_csv(
    "data/clean_sales.csv",
    parse_dates=["date"]
)

# 2. Select one store and product
store_id = 1
item_id = 1

df = df[
    (df["store"] == store_id) &
    (df["item"] == item_id)
].copy()

df = df.sort_values("date").reset_index(drop=True)

# 3. Create historical sales features
df["sales_7_days_ago"] = df["sales"].shift(7)
df["sales_14_days_ago"] = df["sales"].shift(14)
df["sales_28_days_ago"] = df["sales"].shift(28)

# 4. Create calendar features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

# 5. Prepare training data
df = df.dropna().reset_index(drop=True)

features = [
    "sales_7_days_ago",
    "sales_14_days_ago",
    "sales_28_days_ago",
    "day_of_week",
    "month",
    "day_of_year",
]

X_train = df[features]
y_train = df["sales"]

# 6. Train the model using all available historical data
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model trained successfully.")
print("Store:", store_id, "| Item:", item_id)

# 7. Store historical sales for recursive forecasting
sales_history = df["sales"].tolist()
last_date = df["date"].max()

forecast_results = []

# 8. Predict the next seven days
for day in range(1, 8):

    future_date = last_date + pd.Timedelta(days=day)

    # Historical sales features
    input_features = pd.DataFrame([{
        "sales_7_days_ago": sales_history[-7],
        "sales_14_days_ago": sales_history[-14],
        "sales_28_days_ago": sales_history[-28],
        "day_of_week": future_date.dayofweek,
        "month": future_date.month,
        "day_of_year": future_date.dayofyear,
    }])

    # Predict sales for this day
    predicted_sales = model.predict(input_features)[0]

    # Sales cannot be negative
    predicted_sales = max(0, predicted_sales)

    # Save the forecast
    forecast_results.append({
        "date": future_date,
        "store": store_id,
        "item": item_id,
        "predicted_sales": round(predicted_sales, 2)
    })

    # Add prediction to history for the next day's forecast
    sales_history.append(predicted_sales)

# 9. Convert results into a DataFrame
forecast_df = pd.DataFrame(forecast_results)

print("\n--- NEXT 7 DAYS SALES FORECAST ---")
print(forecast_df.to_string(index=False))

# 10. Calculate total forecast demand
total_demand = forecast_df["predicted_sales"].sum()

print("\nTotal predicted demand for 7 days:",
      round(total_demand, 2), "units")

# 11. Save forecast results
forecast_df.to_csv(
    "data/forecast_next_7_days.csv",
    index=False
)

print("\nForecast saved to data/forecast_next_7_days.csv")

# 12. Plot the forecast
plt.figure(figsize=(10, 5))

plt.plot(
    forecast_df["date"],
    forecast_df["predicted_sales"],
    marker="o",
    label="Predicted Sales"
)

plt.title("7-Day Sales Forecast - Store 1, Item 1")
plt.xlabel("Date")
plt.ylabel("Predicted Units Sold")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
