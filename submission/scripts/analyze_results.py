import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- CONFIGURATION ---
RESULTS_FILE = "MASTER_RESULTS.csv"
OUTPUT_CHART = "chart_price_of_progress.png"

def load_and_prep_data():
    """Loads results and standardizes the 'method' and 'type' columns."""
    possible_paths = [
        "MASTER_RESULTS.csv",
        "../MASTER_RESULTS.csv",
        "../../MASTER_RESULTS.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            df = pd.read_csv(path)
            break
            
    if df is None:
        print("Error: MASTER_RESULTS.csv not found!")
        sys.exit(1)
    
    if 'type' not in df.columns:
        df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')
    
    df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)

    def get_method_name(filename):
        if 'Original' in filename or '.csv' not in filename:
            return 'Original'
        parts = filename.replace('.csv', '').split('_')
        for p in parts:
            if p.startswith('knn'): return p
        return 'Original'

    df['method'] = df['filename'].apply(get_method_name)
    return df

def calculate_pareto(df_grouped):
    print("\n--- PARETO FRONTIER ANALYSIS ---")
    for ds in df_grouped['dataset'].unique():
        subset = df_grouped[df_grouped['dataset'] == ds].copy()
        costs = subset[['privacy_risk', 'test_accuracy', 'equalized_odds']].values
        costs[:, 1] = -costs[:, 1] 

        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.any(costs[is_efficient] == c, axis=1)
                
        pareto_models = subset[is_efficient]
        print(f"\nDataset: {ds}")
        for _, row in pareto_models.iterrows():
            print(f"  * Optimal Model: {row['method']} (Acc: {row['test_accuracy']:.2f}, Privacy: {row['privacy_risk']:.1f}%, Unfairness: {row['equalized_odds']:.3f})")

def plot_price_of_progress(df):
    baselines = df[df['type'] == 'Original'].groupby('dataset')[['test_accuracy', 'equalized_odds']].mean().reset_index()
    plot_data = []
    synthetics = df[df['type'] == 'Synthetic']
    
    for _, row in synthetics.iterrows():
        base = baselines[baselines['dataset'] == row['dataset']]
        if base.empty: continue
        
        base_acc = base.iloc[0]['test_accuracy']
        base_fair = base.iloc[0]['equalized_odds']
        
        acc_change = row['test_accuracy'] - base_acc 
        fair_change = base_fair - row['equalized_odds']
        
        plot_data.append({'dataset': row['dataset'], 'method': row['method'], 'Metric': 'Accuracy Impact', 'Value': acc_change})
        plot_data.append({'dataset': row['dataset'], 'method': row['method'], 'Metric': 'Fairness Gain', 'Value': fair_change})

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    sns.barplot(
        data=plot_df, x='dataset', y='Value', hue='Metric', 
        palette={'Accuracy Impact': '#d62728', 'Fairness Gain': '#2ca02c'}, 
        errorbar='sd', alpha=0.8
    )
    plt.axhline(0, color='black', linewidth=1)
    plt.title("The Price of Progress: What did we pay (Accuracy) to get Fairness?", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART)
    print(f"\n[x] Chart saved as {OUTPUT_CHART}")

def main():
    df = load_and_prep_data()
    grouped = df.groupby(['dataset', 'method', 'type'])[['test_accuracy', 'equalized_odds', 'privacy_risk']].mean().reset_index()
    calculate_pareto(grouped)
    plot_price_of_progress(df)

if __name__ == "__main__":
    main()