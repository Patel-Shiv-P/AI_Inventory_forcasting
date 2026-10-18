
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("data/train.csv")

# 2. Convert date column
df["date"] = pd.to_datetime(df["date"])

# 3. Check unique store-product-date records
duplicate_keys = df.duplicated(
    subset=["date", "store", "item"]
).sum()

print("Duplicate store-product-date records:", duplicate_keys)

# 4. Sort data by date, store, and item
df = df.sort_values(
    by=["date", "store", "item"]
).reset_index(drop=True)

# 5. Create a year-month column for analysis
df["year_month"] = df["date"].dt.to_period("M")

# 6. Calculate total monthly sales
monthly_sales = df.groupby(
    "year_month"
)["sales"].sum()

print("\n--- FIRST 12 MONTHLY TOTALS ---")
print(monthly_sales.head(12))

# 7. Plot monthly sales trend
monthly_sales.plot(
    figsize=(12, 5),
    title="Monthly Sales Trend",
    xlabel="Month",
    ylabel="Units Sold"
)

plt.tight_layout()
plt.show()

# 8. Remove the temporary analysis column
df = df.drop(columns=["year_month"])

# 9. Save prepared data
df.to_csv("data/clean_sales.csv", index=False)

print("\nPrepared dataset saved to data/clean_sales.csv")
print("Final shape:", df.shape)
print("Date data type:", df["date"].dtype)
