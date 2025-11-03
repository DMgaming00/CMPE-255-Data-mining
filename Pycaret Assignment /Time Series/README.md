# ⏳ Time-Series Forecasting

This notebook implements **PyCaret’s time-series module** to forecast future values using historical data.

## 📘 Overview
- Dataset: *Daily Minimum Temperatures in Melbourne*
- Target: `Temp`
- Automatically detected time frequency and forecast horizon
- Compared multiple forecasting models to pick the best

## ⚙️ Steps
1. Loaded and cleaned time-series dataset
2. Identified date and target columns
3. Used `setup()` to initialize PyCaret
4. Compared models using `compare_models()`
5. Generated future forecasts and saved outputs

## 📊 Outcome
- Predicted temperature trends over time
- Exported `forecast_future.csv`, `series_cleaned.csv`, and `best_ts_model.pkl`
