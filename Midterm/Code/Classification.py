"""
Name: Tejas Kulkarni
Library: PyCaret
URL: https://pycaret.gitbook.io/docs
Description:
PyCaret is an open-source, low-code Python library designed to automate the end-to-end machine learning workflow.
For classification, it rapidly compares, tunes, and deploys predictive models, significantly increasing a
data scientist's efficiency and experiment velocity by simplifying complex tasks into single-line commands.
"""

# --- Imports for Functional and OOP API Access ---
from pycaret.datasets import get_data
from pycaret.classification import (
    setup,
    compare_models,
    predict_model,
    plot_model,
    save_model
)
from pycaret.classification import ClassificationExperiment # Required for OOP

print("--- Structured Classification Workflow ---")


# 1. LOAD DATASET
data = get_data('bank')
print("\n[STEP 1] Data Loaded (Bank Marketing Data).")

# 2. FUNCTIONAL API DEMO (Quick Check)
print("\n[STEP 2] Functional API Demo: Running Setup for Quick Check...")
# The functional API is fine for a quick benchmark/check
# *** FIXED TARGET: 'deposit' ***
s_functional = setup(
    data=data,
    target='deposit',
    session_id=7502,
    normalize=True,
    verbose=False # Set to False to prevent clash with OOP's setup output
)
print("[STEP 2] Functional Setup Complete. Now Switching to OOP API for Full Workflow.")

# 3. OOP API (SETUP - For the Main Workflow)
# Initialize the OOP Experiment object. This is a clean slate.
exp = ClassificationExperiment()
print("\n[STEP 3] Initializing OOP API Setup (Full Workflow)...")

# We run setup again on the OOP object for the main analysis flow.
exp.setup(
    data=data,
    target='deposit',
    session_id=7502,
    normalize=True,
    verbose=True # Set to True to display the main Setup Grid
)

# 4. COMPARE MODELS
print("\n[STEP 4] Comparing Models (Benchmarking all algorithms)...")
# We use the OOP method: exp.compare_models()
best_model = exp.compare_models(verbose=True)
print(f"\nModel Comparison Complete. Best Model: {type(best_model).__name__}")

# 5. ANALYSE MODEL (Diagnostics)
print("\n[STEP 5] Analyzing Model (Plotting AUC and Confusion Matrix)...")
# Analyze AUC curve using OOP method
exp.plot_model(best_model, plot='auc', save=True)
print("AUC Plot saved to project directory.")

# Analyze Confusion Matrix using OOP method
exp.plot_model(best_model, plot='confusion_matrix', save=True)
print("Confusion Matrix saved to project directory.")

# 6. PREDICTIONS
print("\n[STEP 6] Generating Predictions on Held-Out Test Set...")
# Predict on the held-out test set using OOP method
predictions = exp.predict_model(best_model)
print("\nPredictions on held-out set (first 5 rows):")
# Note: The original 'deposit' column is compared against the new 'prediction_label'
print(predictions[['deposit', 'prediction_label']].head())

# 7. SAVE MODEL
print("\n[STEP 7] Saving Pipeline (MLOps Readiness)...")
# Save the final model and the entire preprocessing pipeline using OOP method
exp.save_model(best_model, 'pycaret_bank_final_pipeline')
print("\nModel saved successfully as 'pycaret_bank_final_pipeline.pkl'.")

print("\n--- Workflow Execution Complete ---")