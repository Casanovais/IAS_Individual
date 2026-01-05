import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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

def main():
    df = load_robust_csv()
    if df is None:
        print("Error: CSV not found")
        sys.exit(1)

    # Cleanup
    if 'type' not in df.columns:
        df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')
    df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)

    # FIX 1: Add the logic to create the 'method' column (needed for Chart 5)
    def get_method_name(filename):
        if 'Original' in filename or '.csv' not in filename:
            return 'Original'
        parts = filename.replace('.csv', '').split('_')
        for p in parts:
            if p.startswith('knn'): return p
        return 'Original'

    df['method'] = df['filename'].apply(get_method_name)

    # Helper for Chart 1
    def get_k(filename):
        if 'knn1' in str(filename): return 1
        if 'knn3' in str(filename): return 3
        if 'knn5' in str(filename): return 5
        return 0 

    df['k'] = df['filename'].apply(get_k)

    # ==========================================
    # CHART 1: HYPERPARAMETER SENSITIVITY
    # ==========================================
    syn_df = df[df['k'] > 0].copy()
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=syn_df, x='k', y='privacy_risk', hue='dataset', marker='o', linewidth=2.5, palette='viridis')
    plt.title("Hyperparameter Sensitivity: Effect of Neighbors (k) on Privacy", fontsize=14)
    plt.xlabel("Number of Neighbors (k)", fontsize=12)
    plt.ylabel("Privacy Risk (%)", fontsize=12)
    plt.xticks([1, 3, 5])
    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("chart_k_sensitivity.png")
    print("Saved chart_k_sensitivity.png")

    # ==========================================
    # CHART 2: ENGINEERING OPTIMIZATION
    # ==========================================
    opt_data = pd.DataFrame({
        'Method': ['Naive (Cartesian)', 'Optimized (Hash-Based)'],
        'Time (Seconds)': [14400, 90]
    })
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    # FIX 2: Added hue='Method' and legend=False to fix FutureWarning
    sns.barplot(data=opt_data, x='Method', y='Time (Seconds)', hue='Method', palette=['#d62728', '#2ca02c'], legend=False)
    plt.yscale('log')
    plt.title("Privacy Audit Optimization: Runtime Reduction", fontsize=14)
    plt.ylabel("Execution Time (Seconds) - Log Scale", fontsize=12)
    plt.text(0, 14400, "14,400s (4h)", ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.text(1, 90, "90s", ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("chart_optimization.png")
    print("Saved chart_optimization.png")

    # ==========================================
    # CHART 3: PARETO CLOUD (Fully Restored)
    # ==========================================
    adult_df = df[df['dataset'] == 'adult'].copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Synthetic Points
    points = ax.scatter(
        adult_df[adult_df['type'] == 'Synthetic']['privacy_risk'],
        adult_df[adult_df['type'] == 'Synthetic']['test_f1_weighted'],
        c=adult_df[adult_df['type'] == 'Synthetic']['equalized_odds'],
        s=150, cmap='RdYlGn_r', alpha=0.8, edgecolor='gray', linewidth=0.5, label='Synthetic Model'
    )
    
    # Original Baseline
    orig = adult_df[adult_df['type'] == 'Original'].iloc[0]
    ax.scatter(orig['privacy_risk'], orig['test_f1_weighted'], color='black', marker='*', s=400, label='Original Baseline', zorder=10)
    
    # Colorbar and Labels
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label('Unfairness (Equalized Odds)\nLower is Better (Green)', fontsize=11)
    
    ax.set_title("Pareto Analysis: Privacy vs. Utility (Adult Dataset)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Privacy Risk (%) - Lower is Better", fontsize=12)
    ax.set_ylabel("F1 Score (Utility) - Higher is Better", fontsize=12)
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("chart_pareto_cloud.png", dpi=300)
    print("Saved chart_pareto_cloud.png")

    # ==========================================
    # CHART 4: THE KNOT
    # ==========================================
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    synthetics = df[df['type'] == 'Synthetic'].copy()
    
    sns.scatterplot(
        data=synthetics,
        x='privacy_risk',
        y='equalized_odds',
        hue='dataset',
        style='dataset',
        s=100,
        alpha=0.8,
        palette='bright'
    )
    
    sns.regplot(
        data=synthetics, 
        x='privacy_risk', 
        y='equalized_odds', 
        scatter=False, 
        color='black', 
        line_kws={"linestyle": "--", "alpha": 0.5},
        label='Global Trend'
    )
    
    plt.title("Synergy Analysis: Does Privacy (Risk) Correlate with Unfairness?", fontsize=14)
    plt.xlabel("Privacy Risk % (Lower = More Private)", fontsize=12)
    plt.ylabel("Unfairness (Lower = More Fair)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("chart_bias_privacy_correlation.png")
    print("Saved chart_bias_privacy_correlation.png")
    
    # ==========================================
    # CHART 5: PRICE OF PROGRESS
    # ==========================================
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
        data=plot_df, 
        x='dataset', 
        y='Value', 
        hue='Metric', 
        palette={'Accuracy Impact': '#d62728', 'Fairness Gain': '#2ca02c'}, 
        errorbar='sd', 
        alpha=0.8
    )
    plt.axhline(0, color='black', linewidth=1)
    plt.title("The Price of Progress: What did we pay (Accuracy) to get Fairness?", fontsize=16)
    plt.ylabel("Change relative to Original Model", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.legend(title="Trade-off Metrics")
    plt.tight_layout()
    plt.savefig("chart_price_of_progress.png")
    print("Saved chart_price_of_progress.png")

if __name__ == "__main__":
    main()