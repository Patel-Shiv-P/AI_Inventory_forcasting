
import math

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Inventory Forecasting",
    page_icon="📦",
    layout="wide"
)

st.title("📦 AI Inventory Forecasting System")
st.write(
    "Forecast product demand and calculate inventory "
    "replenishment recommendations."
)

st.info(
    "This version uses historical sales data ending in 2017. "
    "Forecast dates are a demonstration, not current 2026 demand."
)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(
        "data/clean_sales.csv",
        parse_dates=["date"]
    )

    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Make sure "
        "data/clean_sales.csv exists."
    )
    st.stop()


# --------------------------------------------------
# 3. SIDEBAR: PRODUCT SELECTION
# --------------------------------------------------

st.sidebar.header("Forecast Settings")

store_ids = sorted(df["store"].unique().tolist())
item_ids = sorted(df["item"].unique().tolist())

store_id = st.sidebar.selectbox(
    "Select Store",
    store_ids
)

item_id = st.sidebar.selectbox(
    "Select Product",
    item_ids
)

st.sidebar.caption(
    "Choose a store and product to forecast."
)


# --------------------------------------------------
# 4. SIDEBAR: INVENTORY INPUTS
# --------------------------------------------------

st.sidebar.header("Inventory Details")

current_stock = st.sidebar.number_input(
    "Current Stock (units)",
    min_value=0,
    value=100,
    step=1
)

incoming_stock = st.sidebar.number_input(
    "Incoming Stock (units)",
    min_value=0,
    value=0,
    step=1
)

committed_stock = st.sidebar.number_input(
    "Committed Stock (units)",
    min_value=0,
    value=0,
    step=1
)

lead_time_days = st.sidebar.number_input(
    "Supplier Lead Time (days)",
    min_value=1,
    max_value=7,
    value=2,
    step=1
)

safety_stock = st.sidebar.number_input(
    "Safety Stock (units)",
    min_value=0,
    value=10,
    step=1
)



# --------------------------------------------------
# 4.5 MODEL EVALUATION
# --------------------------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error


@st.cache_data
def evaluate_model(data, selected_store, selected_item):

    product_data = data[
        (data["store"] == selected_store)
        & (data["item"] == selected_item)
    ].copy()

    product_data = product_data.sort_values(
        "date"
    ).reset_index(drop=True)

    # Create historical features
    product_data["sales_7_days_ago"] = (
        product_data["sales"].shift(7)
    )
    product_data["sales_14_days_ago"] = (
        product_data["sales"].shift(14)
    )
    product_data["sales_28_days_ago"] = (
        product_data["sales"].shift(28)
    )

    product_data["day_of_week"] = (
        product_data["date"].dt.dayofweek
    )
    product_data["month"] = (
        product_data["date"].dt.month
    )
    product_data["day_of_year"] = (
        product_data["date"].dt.dayofyear
    )

    product_data = product_data.dropna().reset_index(
        drop=True
    )

    features = [
        "sales_7_days_ago",
        "sales_14_days_ago",
        "sales_28_days_ago",
        "day_of_week",
        "month",
        "day_of_year"
    ]

    X = product_data[features]
    y = product_data["sales"]

    # Chronological train/test split
    split_index = int(len(product_data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Not enough data to evaluate this product."
        )

    # Baseline: use sales from seven days earlier
    baseline_predictions = X_test["sales_7_days_ago"]

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    model_predictions = model.predict(X_test)

    # Calculate evaluation metrics
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

    improvement = (
        (baseline_mae - model_mae) / baseline_mae * 100
        if baseline_mae > 0
        else 0
    )

    return {
        "Baseline MAE": baseline_mae,
        "Baseline RMSE": baseline_rmse,
        "Random Forest MAE": model_mae,
        "Random Forest RMSE": model_rmse,
        "MAE Improvement (%)": improvement,
        "Test Rows": len(X_test)
    }


# --------------------------------------------------
# 5. FORECAST FUNCTION
# --------------------------------------------------

def generate_forecast(data, selected_store, selected_item):

    product_data = data[
        (data["store"] == selected_store)
        & (data["item"] == selected_item)
    ].copy()

    product_data = product_data.sort_values(
        "date"
    ).reset_index(drop=True)

    if len(product_data) < 60:
        raise ValueError(
            "Not enough historical data to forecast "
            "this product reliably."
        )

    sales_history = product_data["sales"].tolist()
    last_date = product_data["date"].max()

    # Create historical training features
    product_data["sales_7_days_ago"] = (
        product_data["sales"].shift(7)
    )

    product_data["sales_14_days_ago"] = (
        product_data["sales"].shift(14)
    )

    product_data["sales_28_days_ago"] = (
        product_data["sales"].shift(28)
    )

    product_data["day_of_week"] = (
        product_data["date"].dt.dayofweek
    )

    product_data["month"] = (
        product_data["date"].dt.month
    )

    product_data["day_of_year"] = (
        product_data["date"].dt.dayofyear
    )

    product_data = product_data.dropna().reset_index(
        drop=True
    )

    features = [
        "sales_7_days_ago",
        "sales_14_days_ago",
        "sales_28_days_ago",
        "day_of_week",
        "month",
        "day_of_year"
    ]

    X_train = product_data[features]
    y_train = product_data["sales"]

    # Train the model on all available historical records
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Generate a recursive seven-day forecast
    results = []

    for day in range(1, 8):

        future_date = last_date + pd.Timedelta(
            days=day
        )

        input_data = pd.DataFrame([{
            "sales_7_days_ago": sales_history[-7],
            "sales_14_days_ago": sales_history[-14],
            "sales_28_days_ago": sales_history[-28],
            "day_of_week": future_date.dayofweek,
            "month": future_date.month,
            "day_of_year": future_date.dayofyear
        }])

        prediction = float(
            model.predict(input_data)[0]
        )

        prediction = max(0, prediction)

        results.append({
            "Date": future_date,
            "Predicted Sales": round(prediction, 2)
        })

        # Use this prediction when forecasting later days
        sales_history.append(prediction)

    return pd.DataFrame(results), last_date


# --------------------------------------------------
# 6. GENERATE FORECAST
# --------------------------------------------------

if st.button(
    "Generate Forecast",
    type="primary",
    use_container_width=True
):
    try:
        forecast, last_date = generate_forecast(
            df,
            store_id,
            item_id
        )
        # Store only the forecast. Inventory recommendations are recalculated
        # on every rerun so edits to sidebar inputs are reflected immediately.
        st.session_state["forecast"] = forecast
        st.session_state["last_date"] = last_date
        st.session_state["forecast_store_id"] = store_id
        st.session_state["forecast_item_id"] = item_id
    except Exception as error:
        st.error(f"Could not generate forecast: {error}")


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

has_forecast = "forecast" in st.session_state
forecast_matches_selection = (
    has_forecast
    and st.session_state.get("forecast_store_id") == store_id
    and st.session_state.get("forecast_item_id") == item_id
)

if forecast_matches_selection:
    forecast = st.session_state["forecast"]

    # Recalculate inventory recommendations from the latest sidebar inputs.
    lead_time_demand = float(
        forecast["Predicted Sales"].iloc[:int(lead_time_days)].sum()
    )
    weekly_demand = float(forecast["Predicted Sales"].sum())
    inventory_position = int(current_stock + incoming_stock - committed_stock)
    reorder_point = math.ceil(lead_time_demand + safety_stock)
    target_stock = math.ceil(lead_time_demand + weekly_demand + safety_stock)
    order_quantity = max(0, target_stock - inventory_position)
    reorder_needed = inventory_position <= reorder_point

    st.subheader(f"Forecast: Store {store_id}, Product {item_id}")
    st.caption(f"Last historical date: {st.session_state['last_date'].date()}")

    col1, col2, col3 = st.columns(3)
    col1.metric("7-Day Forecast Demand", f"{weekly_demand:.1f} units")
    col2.metric("Reorder Point", f"{reorder_point} units")
    col3.metric("Suggested Order", f"{order_quantity} units")

    st.subheader("Seven-Day Sales Forecast")
    chart_data = forecast.set_index("Date")[["Predicted Sales"]]
    st.line_chart(chart_data)

    st.subheader("Daily Forecast")
    st.dataframe(forecast, use_container_width=True, hide_index=True)

    st.subheader("Inventory Recommendation")
    col1, col2, col3 = st.columns(3)
    col1.metric("Inventory Position", f"{inventory_position} units")
    col2.metric("Lead-Time Demand", f"{lead_time_demand:.2f} units")
    col3.metric("Target Stock", f"{target_stock} units")

    if reorder_needed:
        st.warning("Reorder trigger reached. Review the suggested order quantity.")
    else:
        st.success("Reorder trigger not reached. Monitor inventory and plan replenishment.")

    st.caption(
        "The suggested quantity targets stock for the supplier lead time plus "
        "a seven-day review period and safety stock. It is a planning estimate, "
        "not an instruction to order immediately."
    )

    st.subheader("Model Performance")
    try:
        metrics = evaluate_model(df, store_id, item_id)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Random Forest MAE",
            f"{metrics['Random Forest MAE']:.2f}",
            help="Lower is better. Average absolute error in units."
        )
        col2.metric(
            "Random Forest RMSE",
            f"{metrics['Random Forest RMSE']:.2f}",
            help="Lower is better. Penalizes larger errors."
        )
        col3.metric("Baseline MAE", f"{metrics['Baseline MAE']:.2f}")
        col4.metric("MAE Improvement", f"{metrics['MAE Improvement (%)']:.1f}%")

        if metrics["Random Forest MAE"] < metrics["Baseline MAE"]:
            st.success("Random Forest outperforms the baseline on MAE for this test period.")
        else:
            st.warning("Random Forest does not outperform the baseline on MAE for this test period.")

        st.caption(
            f"Chronological holdout evaluation: {metrics['Test Rows']} test rows. "
            "These are one-day-ahead metrics, not seven-day recursive forecast accuracy."
        )
    except (ValueError, KeyError) as error:
        st.warning(f"Model evaluation unavailable: {error}")

    csv_data = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecast CSV",
        data=csv_data,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )
elif has_forecast:
    st.info(
        "The selected store or product changed. Click Generate Forecast to "
        "create a forecast for the current selection."
    )
else:
    st.info(
        "Select your store, product, and inventory details in the sidebar, "
        "then click Generate Forecast."
    )
