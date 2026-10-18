# AI Inventory Forecasting System

A machine-learning portfolio project that forecasts daily product demand and uses the forecast to support inventory replenishment decisions. The application is built with Python, pandas, scikit-learn, and Streamlit.

> **Dataset limitation:** The included Store Item Demand Forecasting dataset covers 2013-01-01 through 2017-12-31. Forecasts generated from it are historical demonstrations, not current demand predictions. Do not use the output to make real purchasing decisions without current sales and inventory data and business validation.

## Features

- Data preparation and exploratory analysis
- Random Forest regression using lag and calendar features
- Seasonal-naive baseline comparison
- Recursive seven-day demand forecast
- Inventory position, reorder point, and suggested order calculation
- Streamlit dashboard with store/product selection and forecast CSV export
- Chronological one-day-ahead evaluation across store-item series
- Rolling-origin recursive seven-day backtest on a reproducible sample

## Results so far

On the current 10-series seven-day backtest sample:

- Random Forest had lower MAE than the seasonal baseline on **10 of 10 sampled series**.
- Mean per-series MAE improvement was **18.62%**.
- Average baseline MAE: **10.475** sales units.
- Average Random Forest MAE: **8.537** sales units.
- Average baseline RMSE: **13.396** sales units.
- Average Random Forest RMSE: **10.885** sales units.

These results describe a sample of 10 series, not all 500 series, and do not guarantee future performance. One-day-ahead metrics and recursive seven-day metrics measure different forecasting settings and should be reported separately.

## Requirements

- Python 3.10 or newer recommended
- pip
- Windows, macOS, or Linux

## Setup on Windows (PowerShell)

Open a terminal in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The project archive includes `data/clean_sales.csv`. If you prepare the dataset yourself, make sure this file exists before launching the app.

## Run the dashboard

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Run model evaluation

Evaluate one-day-ahead performance across store-item series:

```powershell
python evaluate_all_product.py
```

Run the recursive seven-day backtest on 10 sampled series:

```powershell
python backtest_7day_multi.py
```

For example, evaluate 20 series using 50 trees per model (if supported by the script version included):

```powershell
python backtest_7day_multi.py --sample-size 20 --trees 50
```

The seven-day backtest uses rolling forecast blocks. Predictions within each block are recursive, so model predictions are used for subsequent lag features until the block ends. Actual sales become available after the block is complete. Runtime depends on hardware and the number of series evaluated.

## Model approach

The Random Forest uses these input features:

- Sales 7 days earlier
- Sales 14 days earlier
- Sales 28 days earlier
- Day of week
- Month
- Day of year

The seasonal-naive baseline predicts sales using sales from seven days earlier. The evaluation uses chronological splits to avoid randomly mixing earlier and later dates.

### Metrics

- **MAE (Mean Absolute Error):** average absolute prediction error in sales units; lower is better.
- **RMSE (Root Mean Squared Error):** penalizes larger errors more heavily; lower is better.
- **MAE improvement:** relative reduction in MAE compared with the baseline. Negative improvement means the model performed worse than the baseline.

## Inventory logic

- **Inventory position** = current stock + incoming stock − committed stock.
- **Reorder point** = forecast demand during supplier lead time + safety stock.
- **Suggested order quantity** aims to raise inventory position to a target covering lead-time demand, the review period, and safety stock.

The app's safety stock is user-entered, not statistically optimized. The calculation does not yet account for pack sizes, minimum order quantities, budget limits, supplier variability, perishability, or stockout/censoring effects. Validate these rules with real operational requirements before using them for purchasing.

## Project structure

```text
ai-inventory-forecasting/
├── app.py
├── data_prepairation.py
├── inspeat_data.py
├── evaluate_model.py
├── evaluate_all_product.py
├── backtest_7day.py
├── backtest_7day_multi.py
├── forcast_next_7_days.py
├── inventory_recommedation.py
├── requirements.txt
└── data/
    ├── clean_sales.csv
    ├── model_evaluation_results.csv
    └── backtest result CSV files
```

Some filenames preserve their original spelling in this beginner project.

## Publish to GitHub

1. Create a new **public** repository on GitHub, for example `ai-inventory-forecasting`. Do not initialize it with another README if you plan to upload this folder as-is.
2. Extract this project ZIP and open the extracted project folder.
3. In the terminal, run:

   ```powershell
   git init
   git add .
   git status
   git commit -m "Complete AI inventory forecasting project"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/ai-inventory-forecasting.git
   git push -u origin main
   ```

4. Replace `YOUR-USERNAME` with your GitHub username. GitHub may ask you to sign in or authenticate.
5. Before running `git add .`, check `git status` and confirm you are not uploading secrets, virtual environments, or unrelated private files.
6. Add a screenshot of the working dashboard to the README after you capture it locally, and cite the dataset source in the repository before sharing it publicly.

## Limitations and future improvements

- Replace the historical dataset with recent business sales for real-world forecasts.
- Expand multi-day backtesting beyond the initial 10-series sample.
- Compare additional models and evaluate performance across products with different demand patterns.
- Tune safety stock using demand and lead-time variability.
- Add business constraints such as minimum order quantities, pack sizes, budgets, and supplier calendars.
- Deploy the Streamlit app only after validating its behavior and data handling.

# Live demo: 
Link: https://aiinventoryforcasting-qmvk6y54gdbekgafh3agmq.streamlit.app/
