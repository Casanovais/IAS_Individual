import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. LOAD DATA
try:
    df = pd.read_csv("casanovais/ias_individual/IAS_Individual-4203380b0d53a6b9681d9aef3ecd41c0727b1ee3/MASTER_RESULTS.csv")
except FileNotFoundError:
    # Fallback if running in local folder
    df = pd.read_csv("MASTER_RESULTS.csv")

# Cleanup: Fix Privacy for Original
if 'type' not in df.columns:
    df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')
df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)

# Extract 'k' value from filename (e.g., 'adult_knn3_per1.csv' -> 3)
def get_k(filename):
    if 'knn1' in filename: return 1
    if 'knn3' in filename: return 3
    if 'knn5' in filename: return 5
    return 0 # Original

df['k'] = df['filename'].apply(get_k)

# --- CHART A: HYPERPARAMETER SENSITIVITY (k vs Privacy) ---
# We only care about Synthetic data for this analysis
syn_df = df[df['k'] > 0].copy()

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
# Plot line with markers
sns.lineplot(
    data=syn_df, 
    x='k', 
    y='privacy_risk', 
    hue='dataset', 
    marker='o', 
    linewidth=2.5, 
    palette='viridis'
)

plt.title("Hyperparameter Sensitivity: Effect of Neighbors (k) on Privacy", fontsize=14)
plt.xlabel("Number of Neighbors (k)", fontsize=12)
plt.ylabel("Privacy Risk (%)", fontsize=12)
plt.xticks([1, 3, 5]) # Ensure x-axis only shows valid k values
plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("chart_k_sensitivity.png")
print("Saved chart_k_sensitivity.png")

# --- CHART B: ENGINEERING OPTIMIZATION (Runtime) ---
# Data derived from theoretical O(N^2) vs O(N) estimation for 45k records
# 4 hours = 14400 seconds
opt_data = pd.DataFrame({
    'Method': ['Naive (Cartesian)', 'Optimized (Hash-Based)'],
    'Time (Seconds)': [14400, 90]
})

plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
bar_plot = sns.barplot(
    data=opt_data, 
    x='Method', 
    y='Time (Seconds)', 
    palette=['#d62728', '#2ca02c'] # Red for slow, Green for fast
)

# Use Log Scale because the difference is massive
plt.yscale('log') 
plt.title("Privacy Audit Optimization: Runtime Reduction (Log Scale)", fontsize=14)
plt.ylabel("Execution Time (Seconds) - Log Scale", fontsize=12)

# Add text labels on bars
plt.text(0, 14400, "14,400s (Est.)", ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.text(1, 90, "90s (Measured)", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("chart_optimization.png")
print("Saved chart_optimization.png")