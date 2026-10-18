
import pandas as pd

# 1. Load the sales dataset
df = pd.read_csv("data/train.csv")

# 2. Display the first five records
print("\n--- FIRST FIVE RECORDS ---")
print(df.head())

# 3. Check dataset size
print("\n--- DATASET SHAPE ---")
print(df.shape)

# 4. Check column names and data types
print("\n--- DATASET INFORMATION ---")
df.info()

# 5. Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 6. Check duplicate records
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 7. Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 8. Check date range
print("\n--- DATE RANGE ---")
print("Starting date:", df["date"].min())
print("Ending date:", df["date"].max())

# 9. Count unique stores and products
print("\n--- STORES AND PRODUCTS ---")
print("Number of stores:", df["store"].nunique())
print("Number of products:", df["item"].nunique())

# 10. Analyze sales statistics
print("\n--- SALES STATISTICS ---")
print(df["sales"].describe())

# 11. Check invalid values
print("\n--- INVALID VALUES ---")
print("Invalid dates:", df["date"].isnull().sum())
print("Negative sales:", (df["sales"] < 0).sum())

print("\nDataset inspection completed.")


# Total units sold for each product
product_sales = (
    df.groupby("item")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n--- TOP 10 PRODUCTS BY TOTAL SALES ---")
print(product_sales.head(10))