import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# --- 1. DATA PREPARATION ---
df = pd.read_csv("MASTER_RESULTS.csv")

# Ensure 'type' exists
if 'type' not in df.columns:
    df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')

# Fill missing Privacy Risk for Original data (Worst case assumption = 100%)
df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)

# Normalize Fairness Metric (Lower is better usually, let's invert for visualization if needed)
# We will use 'equalized_odds' as the main fairness metric (0 is perfect, 1 is unfair)
# For visualization: "Fairness Score" = 1 - equalized_odds (Higher is Better)
df['Fairness_Score'] = 1 - df['equalized_odds']

# --- VISUALIZATION 1: The "Bubble" Trade-off (Static) ---
# X=Privacy Risk, Y=Accuracy, Size=Fairness
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Plot points
scatter = sns.scatterplot(
    data=df, 
    x="privacy_risk", 
    y="test_accuracy", 
    hue="dataset", 
    style="type",
    size="Fairness_Score", 
    sizes=(50, 400),
    alpha=0.8,
    palette="viridis"
)

# Connect Original to Synthetic
datasets = df['dataset'].unique()
for ds in datasets:
    ds_data = df[df['dataset'] == ds]
    original = ds_data[ds_data['type'] == 'Original']
    if not original.empty:
        start_x, start_y = original.iloc[0]['privacy_risk'], original.iloc[0]['test_accuracy']
        synthetics = ds_data[ds_data['type'] == 'Synthetic']
        for _, row in synthetics.iterrows():
            plt.plot([start_x, row['privacy_risk']], [start_y, row['test_accuracy']], 
                     color='gray', linestyle='--', alpha=0.3, zorder=0)

plt.title("Trade-off Analysis: Privacy vs. Accuracy (Bubble Size = Fairness)", fontsize=15)
plt.xlabel("Privacy Risk (%) (Lower is Better)")
plt.ylabel("Accuracy (Higher is Better)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("chart_bubble_tradeoff.png")
print("Saved chart_bubble_tradeoff.png")

# --- VISUALIZATION 2: 3D Interactive Plot (Plotly) ---
# This allows you to rotate and see the 'Knot' structure
fig = px.scatter_3d(
    df, 
    x='privacy_risk', 
    y='test_accuracy', 
    z='Fairness_Score',
    color='dataset', 
    symbol='type',
    hover_data=['filename'],
    title="The Three-Way Knot: Privacy vs. Accuracy vs. Fairness",
    labels={
        "privacy_risk": "Privacy Risk (Lower=Better)", 
        "test_accuracy": "Accuracy",
        "Fairness_Score": "Fairness (1 - EqOdds)"
    }
)
fig.update_layout(scene = dict(
                    xaxis_title='Privacy Risk (Bad ->)',
                    yaxis_title='Accuracy (Good ^)',
                    zaxis_title='Fairness (Good ^)'),
                    margin=dict(r=20, b=10, l=10, t=40))
fig.write_html("chart_3d_knot.html")
print("Saved chart_3d_knot.html")

# --- VISUALIZATION 3: Parallel Coordinates (The "Price" View) ---
# This shows how optimizing one metric drops the others
fig_par = px.parallel_coordinates(
    df, 
    color="test_accuracy", 
    labels={
                "privacy_risk": "Privacy Risk",
                "equalized_odds": "Unfairness (EqOdds)",
                "test_accuracy": "Accuracy",
    },
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=0.7,
    title="Parallel Coordinates: Tracing the Cost of Privacy"
)
fig_par.write_html("chart_parallel_coords.html")
print("Saved chart_parallel_coords.html")