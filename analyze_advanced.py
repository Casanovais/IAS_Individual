import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --- CONFIGURATION ---
# UPDATE THIS PATH if needed, just like you did for s.py
RESULTS_PATH = "MASTER_RESULTS.csv"
PARETO_METRIC = 'test_f1_weighted'  # Using F1 instead of Accuracy for better results

def load_data():
    try:
        df = pd.read_csv(RESULTS_PATH)
    except FileNotFoundError:
        # Fallback if file is in current dir
        df = pd.read_csv("MASTER_RESULTS.csv")
        
    # 1. Cleanup
    if 'type' not in df.columns:
        df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')
    
    # Set Original Privacy Risk to 100
    df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)
    
    # Extract Method Name (knn1, knn3...)
    def get_method(fname):
        if 'Original' in fname or '.csv' not in fname: return 'Original'
        parts = fname.replace('.csv','').split('_')
        for p in parts:
            if p.startswith('knn'): return p
        return 'Original'
    
    df['method'] = df['filename'].apply(get_method)
    return df

def plot_bias_privacy_correlation(df):
    """
    Idea B: Does Privacy Help Fairness?
    X = Privacy Risk (Lower is Better Privacy)
    Y = Equalized Odds (Lower is Better Fairness)
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # We remove 'Originals' from correlation check to see the effect of privacy mechanisms specifically
    synthetics = df[df['type'] == 'Synthetic'].copy()
    
    # Scatter plot
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
    
    # Add a global trend line
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
    print("[x] Saved chart_bias_privacy_correlation.png")

def calculate_advanced_pareto(df):
    """
    Idea A: Pareto Frontier using F1-Score
    """
    print(f"\n--- PARETO FRONTIER (Optimizing: Privacy, {PARETO_METRIC}, Fairness) ---")
    
    # Average across runs (per1, per2, per3) first
    grouped = df.groupby(['dataset', 'method', 'type'])[[PARETO_METRIC, 'equalized_odds', 'privacy_risk']].mean().reset_index()
    
    pareto_rows = []
    
    for ds in grouped['dataset'].unique():
        subset = grouped[grouped['dataset'] == ds].copy()
        
        # Costs Matrix:
        # 1. Privacy (Minimize) -> value
        # 2. Performance (Maximize) -> -value
        # 3. Unfairness (Minimize) -> value
        costs = subset[['privacy_risk', PARETO_METRIC, 'equalized_odds']].values
        costs[:, 1] = -costs[:, 1]
        
        # Identify Pareto
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # If i is strictly better than j, mark j as False
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.any(costs[is_efficient] == c, axis=1)
                
        optimal = subset[is_efficient]
        pareto_rows.append(optimal)
        
        print(f"\nDataset: {ds}")
        for _, row in optimal.iterrows():
            print(f"  * Best Model: {row['method']:<10} | F1: {row[PARETO_METRIC]:.3f} | Priv: {row['privacy_risk']:.1f}% | Unfair: {row['equalized_odds']:.3f}")

    # Save Pareto Table
    pd.concat(pareto_rows).to_csv("pareto_optimal_models.csv", index=False)
    print("\n[x] Saved Pareto table to 'pareto_optimal_models.csv'")

if __name__ == "__main__":
    df = load_data()
    plot_bias_privacy_correlation(df)
    calculate_advanced_pareto(df)