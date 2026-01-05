import pandas as pd
import numpy as np
import os

# Define possible paths for the CSV file
csv_paths = [
    "MASTER_RESULTS.csv",                               # Current directory
    "../../MASTER_RESULTS.csv",                         # Two levels up (project root)
    "../MASTER_RESULTS.csv",                            # One level up
    "casanovais/ias_individual/IAS_Individual/MASTER_RESULTS.csv" # Original path
]

df = None
loaded_path = ""

for path in csv_paths:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            loaded_path = path
            print(f"Successfully loaded data from: {path}")
            break
        except Exception as e:
            print(f"Found file at {path} but failed to load: {e}")

if df is None:
    print("Error: MASTER_RESULTS.csv not found in any standard location.")
    print("Please ensure the file is in the project root or the current directory.")
    exit()

# Filter for German dataset
german = df[df['dataset'] == 'german']

print("\n--- GERMAN DATASET ANALYSIS ---")
print(german[['filename', 'test_accuracy', 'test_f1_weighted', 'equalized_odds']].head(10))

# Logic Check:
# If Fairness = 1.0, it usually means TPR (True Positive Rate) is 0 for one group and 1 for another, 
# OR (more likely) the model predicts ONE class for everyone, making denominators weird.
# Since we don't have the raw predictions (y_pred), we infer from metrics.

print("\n--- HYPOTHESIS CHECK ---")
print("If 'equalized_odds' is exactly 1.0, it implies maximum disparity.")
print("This often happens when the model predicts the majority class for everyone.")
print("Check the 'test_accuracy'. If it matches the majority class percentage (~70% for German Credit),")
print("then the model has suffered MODE COLLAPSE.")

majority_class_baseline = 0.70 # German credit is usually 70% Good / 30% Bad
print(f"\nAverage Accuracy: {german['test_accuracy'].mean():.4f}")
print(f"Majority Class Baseline: {majority_class_baseline}")

if abs(german['test_accuracy'].mean() - majority_class_baseline) < 0.05:
    print("\n>>> VERDICT: HIGH LIKELIHOOD OF MODE COLLAPSE (Model predicts 'Good' for everyone).")
else:
    print("\n>>> VERDICT: Model is making decisions, but they are extremely biased.")