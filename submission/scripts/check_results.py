import pandas as pd

# 1. Load Data
try:
    # Use the full path if running in the original environment, or local path
    df = pd.read_csv("casanovais/ias_individual/IAS_Individual-4203380b0d53a6b9681d9aef3ecd41c0727b1ee3/MASTER_RESULTS.csv")
except FileNotFoundError:
    df = pd.read_csv("MASTER_RESULTS.csv")

# 2. Cleanup & Preprocessing
# Ensure 'type' column exists or infer it
if 'type' not in df.columns:
    df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')

# Fill missing Privacy Risk for Original data (Worst case assumption = 100%)
df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)

# Extract 'k' parameter from filename (e.g., 'adult_knn3_per1.csv' -> 3)
def get_k(filename):
    if 'knn1' in str(filename): return 1
    if 'knn3' in str(filename): return 3
    if 'knn5' in str(filename): return 5
    if 'Original' in str(filename) or 'knn' not in str(filename): return 0 # 0 denotes Original
    return -1 # Error case

df['k'] = df['filename'].apply(get_k)

# 3. Group and Aggregate
# We want to see the Mean of these metrics for each (Dataset + k) combination
# We also count 'filename' to ensure we have all runs (per1, per2, per3)
summary = df.groupby(['dataset', 'type', 'k']).agg({
    'privacy_risk': ['mean', 'std'],
    'test_f1_weighted': 'mean',
    'equalized_odds': 'mean',
    'filename': 'count' # Number of runs
}).reset_index()

# 4. Format and Print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

print("\n--- ANALYTICAL RESULTS SUMMARY ---")
print("k=0 represents the Original (Baseline) Dataset")
print("-" * 80)
print(summary)
print("-" * 80)

# 5. Specific Check for your question (Adult vs German sensitivity)
print("\n--- SENSITIVITY CHECK (Delta between k=1 and k=5) ---")
datasets = df['dataset'].unique()
for ds in datasets:
    subset = summary[summary['dataset'] == ds]
    
    # Get values for k=1 and k=5 if they exist
    try:
        risk_k1 = subset.loc[subset['k'] == 1, ('privacy_risk', 'mean')].values[0]
        risk_k5 = subset.loc[subset['k'] == 5, ('privacy_risk', 'mean')].values[0]
        delta = risk_k5 - risk_k1
        print(f"Dataset: {ds:<15} | k=1 Risk: {risk_k1:.2f}% | k=5 Risk: {risk_k5:.2f}% | Change: {delta:.2f}%")
    except IndexError:
        print(f"Dataset: {ds:<15} | (Incomplete k values)")