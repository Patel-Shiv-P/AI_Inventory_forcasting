"""Evaluate recursive 7-day forecasts on a reproducible sample of store-item series.

Usage:
  python backtest_7day_multi.py
  python backtest_7day_multi.py --sample-size 20 --trees 50

Each forecast block is up to 7 days. The model only sees history available
before the block; within a block, predictions are used for lag features.
Actual sales are added to history only after the block is complete.
"""
import argparse
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    "sales_7_days_ago", "sales_14_days_ago", "sales_28_days_ago",
    "day_of_week", "month", "day_of_year"
]

def feature_frame(history):
    result = history.copy()
    result["sales_7_days_ago"] = result["sales"].shift(7)
    result["sales_14_days_ago"] = result["sales"].shift(14)
    result["sales_28_days_ago"] = result["sales"].shift(28)
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["day_of_year"] = result["date"].dt.dayofyear
    return result

def evaluate_series(group, trees=50):
    group = group.sort_values("date").reset_index(drop=True)
    if len(group) < 100:
        return None, None

    split = int(len(group) * 0.8)
    history = group.iloc[:split][["date", "sales"]].copy()
    test = group.iloc[split:].copy()
    actuals, baseline_preds, model_preds, dates = [], [], [], []

    for start in range(0, len(test), 7):
        block = test.iloc[start:start + 7].copy()
        train = feature_frame(history).dropna()
        if train.empty:
            continue
        model = RandomForestRegressor(
            n_estimators=trees, random_state=42, n_jobs=-1,
            min_samples_leaf=2
        )
        model.fit(train[FEATURES], train["sales"])
        working = history[["date", "sales"]].copy()

        for _, row in block.iterrows():
            day = row["date"]
            lags = {}
            for lag in (7, 14, 28):
                prior_date = day - pd.Timedelta(days=lag)
                vals = working.loc[working["date"] == prior_date, "sales"]
                lags[f"sales_{lag}_days_ago"] = (
                    float(vals.iloc[0]) if not vals.empty else np.nan
                )
            if any(pd.isna(v) for v in lags.values()):
                raise ValueError(f"Missing lag for date {day}")
            features = {
                **lags,
                "day_of_week": day.dayofweek,
                "month": day.month,
                "day_of_year": day.dayofyear,
            }
            pred = max(0.0, float(model.predict(pd.DataFrame([features])[FEATURES])[0]))
            baseline = max(0.0, float(lags["sales_7_days_ago"]))
            actuals.append(float(row["sales"]))
            baseline_preds.append(baseline)
            model_preds.append(pred)
            dates.append(day)
            working = pd.concat([
                working,
                pd.DataFrame({"date": [day], "sales": [pred]})
            ], ignore_index=True)

        # Reveal actuals only after the entire forecast block has ended.
        history = pd.concat(
            [history, block[["date", "sales"]]], ignore_index=True
        )

    if not actuals:
        return None, None

    base_mae = mean_absolute_error(actuals, baseline_preds)
    rf_mae = mean_absolute_error(actuals, model_preds)
    base_rmse = mean_squared_error(actuals, baseline_preds) ** 0.5
    rf_rmse = mean_squared_error(actuals, model_preds) ** 0.5
    improvement = ((base_mae - rf_mae) / base_mae * 100) if base_mae else np.nan
    summary = {
        "store": int(group["store"].iloc[0]),
        "item": int(group["item"].iloc[0]),
        "forecast_days": len(actuals),
        "baseline_mae": base_mae,
        "random_forest_mae": rf_mae,
        "baseline_rmse": base_rmse,
        "random_forest_rmse": rf_rmse,
        "mae_improvement_percent": improvement,
        "random_forest_wins": rf_mae < base_mae,
    }
    details = pd.DataFrame({
        "date": dates, "actual_sales": actuals,
        "baseline_prediction": baseline_preds,
        "model_prediction": model_preds,
        "store": int(group["store"].iloc[0]),
        "item": int(group["item"].iloc[0]),
    })
    return summary, details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Number of store-item series to evaluate (default: 10).")
    parser.add_argument("--trees", type=int, default=50,
                        help="Random Forest trees per model (default: 50).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_size < 1 or args.trees < 1:
        parser.error("--sample-size and --trees must be positive integers")

    df = pd.read_csv("data/clean_sales.csv", parse_dates=["date"])
    required = {"date", "store", "item", "sales"}
    if not required.issubset(df.columns):
        raise ValueError(f"clean_sales.csv must contain columns: {sorted(required)}")

    pairs = df[["store", "item"]].drop_duplicates().sort_values(
        ["store", "item"]
    )
    if args.sample_size < len(pairs):
        pairs = pairs.sample(n=args.sample_size, random_state=args.seed).sort_values(
            ["store", "item"]
        )
    pair_set = set(map(tuple, pairs[["store", "item"]].to_numpy()))
    summaries, details = [], []
    total = len(pair_set)
    print(f"Evaluating {total} store-item series with {args.trees} trees each...")
    for idx, ((store, item), group) in enumerate(
        df[df.apply(lambda r: (r["store"], r["item"]) in pair_set, axis=1)]
        .groupby(["store", "item"]), start=1
    ):
        print(f"[{idx}/{total}] Store {store}, Item {item}", flush=True)
        summary, detail = evaluate_series(group, trees=args.trees)
        if summary:
            summaries.append(summary)
            details.append(detail)

    if not summaries:
        raise RuntimeError("No series were evaluated.")
    summary_df = pd.DataFrame(summaries).sort_values(["store", "item"])
    detail_df = pd.concat(details, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    summary_df.to_csv("data/backtest_7day_multi_summary.csv", index=False)
    detail_df.to_csv("data/backtest_7day_multi_predictions.csv", index=False)

    print("\n--- MULTI-SERIES SEVEN-DAY BACKTEST ---")
    print("Series evaluated:", len(summary_df))
    print("Series where Random Forest wins:",
          f"{int(summary_df['random_forest_wins'].sum())}/{len(summary_df)}")
    print("Average baseline MAE:", round(summary_df["baseline_mae"].mean(), 3))
    print("Average Random Forest MAE:", round(summary_df["random_forest_mae"].mean(), 3))
    print("Average baseline RMSE:", round(summary_df["baseline_rmse"].mean(), 3))
    print("Average Random Forest RMSE:", round(summary_df["random_forest_rmse"].mean(), 3))
    print("Mean per-series MAE improvement (%):",
          round(summary_df["mae_improvement_percent"].mean(), 2))
    print("\nSaved: data/backtest_7day_multi_summary.csv")
    print("Saved: data/backtest_7day_multi_predictions.csv")

if __name__ == "__main__":
    import os
    main()
