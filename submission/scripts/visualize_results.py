import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

def load_robust_csv():
    paths = ["MASTER_RESULTS.csv", "../../MASTER_RESULTS.csv", "../MASTER_RESULTS.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

if __name__ == "__main__":
    df = load_robust_csv()
    if df is None:
        print("Error: CSV not found")
        sys.exit(1)

    df = df.dropna(subset=['privacy_risk'])

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df, x="privacy_risk", y="test_accuracy", 
        hue="dataset", style="dataset", s=100, alpha=0.8
    )

    plt.title("The Three-Way Knot: Utility vs Privacy Trade-off", fontsize=16)
    plt.xlabel("Re-identification Risk (%) (Lower is Better)", fontsize=12)
    plt.ylabel("Model Accuracy (Higher is Better)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("tradeoff_chart.png")
    print("[x] Chart saved as tradeoff_chart.png")