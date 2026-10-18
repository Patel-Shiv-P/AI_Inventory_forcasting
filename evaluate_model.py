
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load sales data for one store and product
df = pd.read_csv(
    "data/clean_sales.csv",
    parse_dates=["date"]
)

df = df[
    (df["store"] == 1) &
    (df["item"] == 1)
].copy()

df = df.sort_values("date").reset_index(drop=True)

# 2. Create features using past sales only
df["sales_7_days_ago"] = df["sales"].shift(7)
df["sales_14_days_ago"] = df["sales"].shift(14)
df["sales_28_days_ago"] = df["sales"].shift(28)

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

df = df.dropna().reset_index(drop=True)

features = [
    "sales_7_days_ago",
    "sales_14_days_ago",
    "sales_28_days_ago",
    "day_of_week",
    "month",
    "day_of_year"
]

X = df[features]
y = df["sales"]

# 3. Split chronologically: first 80% for training,
#    final 20% for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Baseline: predict the same sales as seven days ago
baseline_predictions = X_test["sales_7_days_ago"]

# 5. Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

model_predictions = model.predict(X_test)

# 6. Calculate evaluation metrics
baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5

model_mae = mean_absolute_error(
    y_test, model_predictions
)

model_rmse = mean_squared_error(
    y_test, model_predictions
) ** 0.5

# 7. Display results
print("\n--- MODEL EVALUATION ---")
print("Store: 1 | Item: 1")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nBaseline MAE:", round(baseline_mae, 2))
print("Baseline RMSE:", round(baseline_rmse, 2))

print("\nRandom Forest MAE:", round(model_mae, 2))
print("Random Forest RMSE:", round(model_rmse, 2))

if model_mae < baseline_mae:
    print("\nRandom Forest has lower MAE than baseline.")
else:
    print("\nRandom Forest did not beat baseline MAE.")
