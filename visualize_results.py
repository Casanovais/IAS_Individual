import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("MASTER_RESULTS.csv")

# Filter out rows where privacy risk failed (N/A)
df = df.dropna(subset=['privacy_risk'])

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Create Scatter Plot
# X-axis: Privacy Risk (Lower is better)
# Y-axis: Accuracy (Higher is better)
# Color: Dataset
sns.scatterplot(
    data=df, 
    x="privacy_risk", 
    y="test_accuracy", 
    hue="dataset", 
    style="dataset", 
    s=100, 
    alpha=0.8
)

# Labels
plt.title("The Three-Way Knot: Utility vs Privacy Trade-off", fontsize=16)
plt.xlabel("Re-identification Risk (%) (Lower is Better)", fontsize=12)
plt.ylabel("Model Accuracy (Higher is Better)", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Save
plt.tight_layout()
plt.savefig("tradeoff_chart.png")
print("[x] Chart saved as tradeoff_chart.png")