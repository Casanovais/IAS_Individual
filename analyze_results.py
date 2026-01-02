import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
RESULTS_FILE = "MASTER_RESULTS.csv"
OUTPUT_CHART = "chart_price_of_progress.png"

def load_and_prep_data():
    """Loads results and standardizes the 'method' and 'type' columns."""
    df = pd.read_csv(RESULTS_FILE)
    
    # 1. Ensure 'type' column exists
    if 'type' not in df.columns:
        df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')
    
    # 2. Fix missing privacy for Originals (Default to 100% risk)
    df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)

    # 3. Extract Method Name (aggregating per1, per2, etc.)
    #    e.g., "adult_knn1_per1.csv" -> "knn1"
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
    """Identifies models on the Pareto Frontier (Best Trade-offs)."""
    print("\n--- PARETO FRONTIER ANALYSIS ---")
    print("(Models that are optimal trade-offs: You cannot improve one metric without hurting another)")
    
    for ds in df_grouped['dataset'].unique():
        subset = df_grouped[df_grouped['dataset'] == ds].copy()
        
        # We want to: MINIMIZE Privacy Risk, MAXIMIZE Accuracy, MINIMIZE Unfairness
        # Convert all to "Costs" (lower is better) for calculation
        # 1. Privacy: Lower is better (Use as is)
        # 2. Accuracy: Higher is better (Multiply by -1 to minimize)
        # 3. Unfairness (EqOdds): Lower is better (Use as is)
        costs = subset[['privacy_risk', 'test_accuracy', 'equalized_odds']].values
        costs[:, 1] = -costs[:, 1] 

        # Simple Pareto check
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Keep point if it is strictly better than any other point in at least one way
                # Remove any point 'j' that is dominated by 'i'
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.any(costs[is_efficient] == c, axis=1)
                
        # Filter and Print
        pareto_models = subset[is_efficient]
        print(f"\nDataset: {ds}")
        for _, row in pareto_models.iterrows():
            print(f"  * Optimal Model: {row['method']} (Acc: {row['test_accuracy']:.2f}, Privacy: {row['privacy_risk']:.1f}%, Unfairness: {row['equalized_odds']:.3f})")

def plot_price_of_progress(df):
    """Generates the Delta Chart showing Gain vs Loss."""
    # 1. Calculate Baselines (Originals)
    baselines = df[df['type'] == 'Original'].groupby('dataset')[['test_accuracy', 'equalized_odds']].mean().reset_index()
    
    # 2. Calculate Deltas for Synthetics
    plot_data = []
    synthetics = df[df['type'] == 'Synthetic']
    
    for _, row in synthetics.iterrows():
        # Find matching baseline for this dataset
        base = baselines[baselines['dataset'] == row['dataset']]
        if base.empty: continue
        
        base_acc = base.iloc[0]['test_accuracy']
        base_fair = base.iloc[0]['equalized_odds']
        
        # Calculate Changes
        # Accuracy Loss (Negative is bad)
        acc_change = row['test_accuracy'] - base_acc 
        # Fairness Gain (Positive is good, assuming lower EqOdds is better)
        fair_change = base_fair - row['equalized_odds']
        
        plot_data.append({'dataset': row['dataset'], 'method': row['method'], 'Metric': 'Accuracy Impact', 'Value': acc_change})
        plot_data.append({'dataset': row['dataset'], 'method': row['method'], 'Metric': 'Fairness Gain', 'Value': fair_change})

    plot_df = pd.DataFrame(plot_data)

    # 3. Plot
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Bar chart with error bars (automatically handled by seaborn if multiple points exist)
    sns.barplot(
        data=plot_df, 
        x='dataset', 
        y='Value', 
        hue='Metric', 
        palette={'Accuracy Impact': '#d62728', 'Fairness Gain': '#2ca02c'}, # Red for loss, Green for gain
        errorbar='sd', # Standard Deviation bars
        alpha=0.8
    )
    
    plt.axhline(0, color='black', linewidth=1)
    plt.title("The Price of Progress: What did we pay (Accuracy) to get Fairness?", fontsize=16)
    plt.ylabel("Change relative to Original Model", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.legend(title="Trade-off Metrics")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART)
    print(f"\n[x] Chart saved as {OUTPUT_CHART}")

def main():
    # 1. Load
    df = load_and_prep_data()
    
    # 2. Group by Method to handle stability (per1, per2, per3)
    #    This gives us the MEAN values for Pareto calculation
    grouped = df.groupby(['dataset', 'method', 'type'])[['test_accuracy', 'equalized_odds', 'privacy_risk']].mean().reset_index()
    
    # 3. Analysis
    calculate_pareto(grouped)
    plot_price_of_progress(df) # Pass full df to let seaborn calculate error bars

if __name__ == "__main__":
    main()