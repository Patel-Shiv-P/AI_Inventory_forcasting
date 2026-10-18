
import pandas as pd
import math

# 1. Load the seven-day forecast
forecast = pd.read_csv(
    "data/forecast_next_7_days.csv",
    parse_dates=["date"]
)

# 2. Enter inventory details for this example
current_stock = 100
incoming_stock = 0
committed_stock = 0
lead_time_days = 2
safety_stock = 10

# 3. Calculate forecast demand during supplier lead time
lead_time_demand = forecast["predicted_sales"].iloc[
    :lead_time_days
].sum()

# 4. Calculate total demand over the seven-day review period
review_period_demand = forecast["predicted_sales"].sum()

# 5. Calculate reorder point
reorder_point = lead_time_demand + safety_stock

# 6. Calculate current inventory position
inventory_position = (
    current_stock
    + incoming_stock
    - committed_stock
)

# 7. Calculate target stock for lead time + review period
target_stock = (
    lead_time_demand
    + review_period_demand
    + safety_stock
)

# 8. Calculate recommended order quantity
order_quantity = max(
    0,
    math.ceil(target_stock - inventory_position)
)

# 9. Determine whether a reorder is needed
reorder_needed = inventory_position <= reorder_point

# 10. Display the recommendation
print("\n--- INVENTORY RECOMMENDATION ---")
print("Store: 1 | Item: 1")
print("Current stock:", current_stock)
print("Incoming stock:", incoming_stock)
print("Committed stock:", committed_stock)
print("Supplier lead time:", lead_time_days, "days")
print("Safety stock:", safety_stock)

print("\nLead-time demand:", round(lead_time_demand, 2))
print("Reorder point:", math.ceil(reorder_point))
print("Inventory position:", inventory_position)
print("Target stock:", math.ceil(target_stock))

print("\nReorder needed:", "YES" if reorder_needed else "NO")
print("Suggested order quantity:", order_quantity, "units")
