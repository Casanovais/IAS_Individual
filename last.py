import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. LOAD DATA
try:
    df = pd.read_csv("casanovais/ias_individual/IAS_Individual-4203380b0d53a6b9681d9aef3ecd41c0727b1ee3/MASTER_RESULTS.csv")
except FileNotFoundError:
    df = pd.read_csv("MASTER_RESULTS.csv")

# Cleanup
if 'type' not in df.columns:
    df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')
df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)

# Filter for Adult Dataset only
adult_df = df[df['dataset'] == 'adult'].copy()

# --- PLOTTING ---
# Use a cleaner style
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 7))

# 1. Plot Synthetic Points (The Cloud)
# We use a standard scatter plot, mapping Color (c) to Unfairness
points = ax.scatter(
    adult_df[adult_df['type'] == 'Synthetic']['privacy_risk'],
    adult_df[adult_df['type'] == 'Synthetic']['test_f1_weighted'],
    c=adult_df[adult_df['type'] == 'Synthetic']['equalized_odds'],
    s=150, # Uniform size for clarity
    cmap='RdYlGn_r', # Red (Unfair) to Green (Fair)
    alpha=0.8,
    edgecolor='gray',
    linewidth=0.5,
    label='Synthetic Model'
)

# 2. Plot Original Baseline
orig = adult_df[adult_df['type'] == 'Original'].iloc[0]
ax.scatter(
    orig['privacy_risk'], 
    orig['test_f1_weighted'], 
    color='black', 
    marker='*', 
    s=400, 
    label='Original Baseline',
    zorder=10
)

# 3. Add Colorbar (Solves the Legend crowding)
cbar = plt.colorbar(points, ax=ax)
cbar.set_label('Unfairness (Equalized Odds)\nLower is Better (Green)', fontsize=11)

# 4. Labels and Titles
ax.set_title("Pareto Analysis: Privacy vs. Utility (Adult Dataset)", fontsize=16, fontweight='bold')
ax.set_xlabel("Privacy Risk (%) - Lower is Better", fontsize=12)
ax.set_ylabel("F1 Score (Utility) - Higher is Better", fontsize=12)

# Simple Legend for just the markers (not the colors)
ax.legend(loc='lower left', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig("chart_pareto_cloud.png", dpi=300)
print("Saved chart_pareto_cloud.png")