"""
Name: Tejas Kulkarni
Library: PyCaret
URL: https://pycaret.gitbook.io/docs
Description:
This library enables rapid prototyping and full lifecycle management for time series forecasting. It automates model
comparison, tuning, and deployment, saving significant time for Data Scientists in complex forecasting scenarios.
"""
# Imports
from pycaret.datasets import get_data
from pycaret.time_series import (
    TSForecastingExperiment,
    setup,
    compare_models,
    plot_model,
    finalize_model,
    predict_model,
    save_model
)

# [STEP 1] Loading the 'airline' dataset
print("[STEP 1] Loading the 'airline' dataset...")
data = get_data('airline')

target_column = 'Number of airline passengers' # Corrected column name for this environment
forecast_horizon = 12 # Predict the next 12 months (one year)

# --- Code execution starts here ---

# [STEP 2] Initializing Functional API Setup...
s_functional = setup(
    data,
    target=target_column,
    fh=forecast_horizon,
    session_id=7502,
    verbose=True # Ensures setup grid is displayed
)

# [STEP 3] Initializing OOP API Setup...
# The OOP API is preferred for stability and structured coding
exp = TSForecastingExperiment()
exp.setup(
    data,
    target=target_column,
    fh=forecast_horizon,
    session_id=7502,
    verbose=True # Ensures setup grid is displayed
)

# [STEP 4] Compare Models (OOP API)
print("\n[STEP 4] Comparing Models to find the best performing algorithm...")
best_model = exp.compare_models(n_select=1)

print(f"\nBest Model Selected: {type(best_model).__name__}")

# [STEP 5] Analyze Model (Plot Forecast)
print("\n[STEP 5] Analyzing Model: Plotting 12-step forecast...")
# This plot will show the historical trend and the predicted future values
exp.plot_model(best_model, plot='forecast')

# [STEP 6] Predictions (Forecast)
print("\n[STEP 6] Making Predictions on the forecast horizon (12 steps)...")
predictions = exp.predict_model(best_model)
print("Forecasted Values (first 5):")
print(predictions.head())

# [STEP 7] Finalize and Save Model (MLOps Readiness)
print("\n[STEP 7] Finalizing model and saving pipeline...")
# Finalize trains the best model on the entire dataset
final_pipeline = exp.finalize_model(best_model)
exp.save_model(final_pipeline, 'time_series_airline_pipeline')

print("\nPipeline saved successfully!")