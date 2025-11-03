"""
Name: Tejas Kulkarni
Library: PyCaret
URL: https://pycaret.gitbook.io/docs
Description:
This library enables rapid prototyping and full lifecycle management for unsupervised anomaly detection. It automates
model creation and analysis, allowing Data Scientists to efficiently identify fraud, errors, or significant outliers
in complex datasets.
"""
# Imports
from pycaret.datasets import get_data
from pycaret.anomaly import (
    AnomalyExperiment,
    setup,
    create_model,
    assign_model,
    plot_model,
    save_model
)

print("--- ANOMALY DETECTION Workflow ---")

# [STEP 1] Loading the 'kiva' dataset
print("[STEP 1] Loading the 'kiva' dataset...")
# Kiva data is often used for detecting unusual/fraudulent loan requests
data = get_data('kiva')

# --- Code execution starts here -

# [STEP 2] Initializing Functional API Setup...
# For Anomaly Detection, no target variable is defined.
print("[STEP 2] Initializing Functional API Setup for Anomaly Detection...")
s_functional = setup(
    data,
    session_id=7502,
    verbose=True # Ensures setup grid is displayed
)

# [STEP 3] Initializing OOP API Setup...
exp = AnomalyExperiment()
exp.setup(
    data,
    session_id=7502,
    verbose=True # Ensures setup grid is displayed
)

# [STEP 4] Create Model (OOP API)
print("\n[STEP 4] Creating Isolation Forest Model (iforest) for anomaly detection...")
# Isolation Forest is a robust, common model for outlier detection
iforest = exp.create_model('iforest', fraction=0.05) # fraction=0.05 tells it to look for 5% anomalies

print(f"\nModel Selected: {type(iforest).__name__}")

# [STEP 5] Analyze Model (Plot Diagnostics)
print("\n[STEP 5] Analyzing Model: Plotting t-SNE projection of anomalies...")
# t-SNE helps visualize the outliers separated from the main cluster
exp.plot_model(iforest, plot='tsne')

# [STEP 6] Assign Anomaly Labels
print("\n[STEP 6] Assigning Anomaly Labels and Scores to the original data...")
# Adds two new columns: 'Anomaly' (1 or 0) and 'Anomaly_Score'
data_with_anomalies = exp.assign_model(iforest)

print("Data with Anomaly Labels (first 5 rows):")
print(data_with_anomalies[['Anomaly', 'Anomaly_Score']].head())
print("\nTotal anomalies found:", data_with_anomalies['Anomaly'].sum())

# [STEP 7] Save Model (MLOps Readiness)
print("\n[STEP 7] Saving anomaly detector and preprocessing pipeline...")
exp.save_model(iforest, 'anomaly_kiva_pipeline')

print("\nPipeline saved successfully!")