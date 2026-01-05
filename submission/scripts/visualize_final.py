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

    if 'type' not in df.columns:
        df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')

    df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    markers = {"Original": "X", "Synthetic": "o"}

    sns.scatterplot(
        data=df, x="privacy_risk", y="test_accuracy", 
        hue="dataset", style="type", markers=markers, s=150, alpha=0.8, palette="deep"
    )

    datasets = df['dataset'].unique()
    for ds in datasets:
        ds_data = df[df['dataset'] == ds]
        original = ds_data[ds_data['type'] == 'Original']
        
        if not original.empty:
            start_x = original.iloc[0]['privacy_risk']
            start_y = original.iloc[0]['test_accuracy']
            synthetics = ds_data[ds_data['type'] == 'Synthetic']
            for _, row in synthetics.iterrows():
                plt.annotate("", 
                             xy=(row['privacy_risk'], row['test_accuracy']), 
                             xytext=(start_x, start_y),
                             arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=1.5))

    plt.title("The Trade-off: How Privacy & Utility Change", fontsize=16)
    plt.xlabel("Re-identification Risk (%) (Lower is Better)", fontsize=12)
    plt.ylabel("Model Accuracy (Higher is Better)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(-5, 105)

    plt.tight_layout()
    plt.savefig("tradeoff_chart_final.png")
    print("[x] Chart saved as tradeoff_chart_final.png")