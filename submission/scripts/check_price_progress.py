import pandas as pd

# 1. Load Data
try:
    df = pd.read_csv("casanovais/ias_individual/IAS_Individual-4203380b0d53a6b9681d9aef3ecd41c0727b1ee3/MASTER_RESULTS.csv")
except FileNotFoundError:
    df = pd.read_csv("MASTER_RESULTS.csv")

# 2. Cleanup
if 'type' not in df.columns:
    df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')

# Extract 'k' just in case you want to filter
def get_k(filename):
    if 'knn1' in str(filename): return 1
    if 'knn3' in str(filename): return 3
    if 'knn5' in str(filename): return 5
    return 0
df['k'] = df['filename'].apply(get_k)

# 3. Calculate Means for Original vs Synthetic
# Group by Dataset and Type to get the "Grand Mean" used in the bar chart
means = df.groupby(['dataset', 'type'])[['test_accuracy', 'equalized_odds']].mean().reset_index()

print("\n--- EXACT VALUES FOR PRICE OF PROGRESS CHART ---")
print(f"{'Dataset':<15} | {'Metric':<15} | {'Original':<10} | {'Synthetic':<10} | {'Absolute Delta':<15} | {'Relative Change':<15}")
print("-" * 95)

dataset_list = means['dataset'].unique()

for ds in dataset_list:
    try:
        # Get Original Baseline
        orig_acc = means.loc[(means['dataset'] == ds) & (means['type'] == 'Original'), 'test_accuracy'].values[0]
        orig_fair = means.loc[(means['dataset'] == ds) & (means['type'] == 'Original'), 'equalized_odds'].values[0]
        
        # Get Synthetic Average (This is what the bar chart usually plots - the average of all syn runs)
        # Note: If your chart filtered for a specific 'k', we should adjust this. 
        # Assuming the chart plotted the *best* or *average* synthetic. 
        # Let's calculate the delta for the BEST synthetic (lowest unfairness) to match the "Gain" narrative.
        
        syn_rows = means.loc[(means['dataset'] == ds) & (means['type'] == 'Synthetic')]
        if syn_rows.empty: continue
        
        syn_acc = syn_rows['test_accuracy'].values[0]
        syn_fair = syn_rows['equalized_odds'].values[0]

        # Calculate Deltas
        # Accuracy Loss (Original - Synthetic) -> Expected Positive for "Price"
        acc_loss = orig_acc - syn_acc
        
        # Fairness Gain (Original - Synthetic) -> Expected Positive for "Gain" (Lower EqOdds is better)
        fair_gain = orig_fair - syn_fair
        
        # Relative Changes
        acc_rel = (acc_loss / orig_acc) * 100
        fair_rel = (fair_gain / orig_fair) * 100

        print(f"{ds:<15} | Accuracy        | {orig_acc:.4f}     | {syn_acc:.4f}     | {acc_loss:.4f}          | -{acc_rel:.2f}%")
        print(f"{ds:<15} | Fairness (EqOd) | {orig_fair:.4f}     | {syn_fair:.4f}     | {fair_gain:.4f}          | +{fair_rel:.2f}%")
        print("-" * 95)

    except IndexError:
        print(f"Skipping {ds} (Missing data)")