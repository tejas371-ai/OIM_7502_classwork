"""
Name: Tejas Kulkarni
Library: PyCaret
URL: https://pycaret.gitbook.io/docs
Description:
This library enables rapid prototyping and full lifecycle management for unsupervised learning tasks like clustering.
It automates model creation, comparison, and analysis, saving significant time for Data Scientists in customer
segmentation scenarios.
"""
# Imports
from pycaret.datasets import get_data
from pycaret.clustering import (
    ClusteringExperiment,
    setup,
    create_model,
    assign_model,
    plot_model,
    save_model
)

print("--- Structured CLUSTERING Workflow ---")

# [STEP 1] Loading the 'jewellery' dataset
print("[STEP 1] Loading the 'jewellery' dataset...")
data = get_data('jewellery')

# --- Code execution starts here ---

# [STEP 2] Initializing Functional API Setup...
# For clustering, no target variable is defined.
s_functional = setup(
    data,
    session_id=7502,
    verbose=True # Ensures setup grid is displayed
)

# [STEP 3] Initializing OOP API Setup...
exp = ClusteringExperiment()
exp.setup(
    data,
    session_id=7502,
    verbose=True # Ensures setup grid is displayed
)

# [STEP 4] Create Model (OOP API)
print("\n[STEP 4] Creating K-Means Model for customer segmentation...")
# We choose K-Means and request 4 clusters (segments)
k_means = exp.create_model('kmeans', num_clusters=4)

print(f"\nModel Selected: {type(k_means).__name__}")

# [STEP 5] Analyze Model (Plot Diagnostics)
print("\n[STEP 5] Analyzing Model: Plotting t-SNE projection of clusters...")
# t-SNE is a common plot to visualize high-dimensional clusters
exp.plot_model(k_means, plot='tsne')

# [STEP 6] Assign Cluster Labels
print("\n[STEP 6] Assigning Cluster Labels back to the original data...")
# Adds a new column ('Cluster') to the data, identifying the segment for each row
data_with_clusters = exp.assign_model(k_means)
print("Data with Clusters (first 5 rows):")
print(data_with_clusters.head())

# [STEP 7] Save Model (MLOps Readiness)
print("\n[STEP 7] Saving clustering model and preprocessing pipeline...")
# Saves the final model and the entire preprocessing pipeline for consistent segmenting of new data
exp.save_model(k_means, 'clustering_jewellery_pipeline')

print("\nPipeline saved successfully!")