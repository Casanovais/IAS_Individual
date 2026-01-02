import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# FROM visualize_final.py
# ==========================================
def generate_tradeoff_with_arrows():
    print("Generating Trade-off Chart with Arrows...")
    # Load data
    try:
        df = pd.read_csv("MASTER_RESULTS.csv")
    except FileNotFoundError:
        print("Error: MASTER_RESULTS.csv not found.")
        return

    # 1. Fix Missing 'type' column if it doesn't exist
    if 'type' not in df.columns:
        df['type'] = df['filename'].apply(lambda x: 'Synthetic' if '_knn' in str(x) else 'Original')

    # 2. Fix Missing Privacy Risk for Originals (Default to 100%)
    df.loc[df['type'] == 'Original', 'privacy_risk'] = df.loc[df['type'] == 'Original', 'privacy_risk'].fillna(100.0)

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # 3. Create the Chart with Distinct Markers
    markers = {"Original": "X", "Synthetic": "o"}

    sns.scatterplot(
        data=df, 
        x="privacy_risk", 
        y="test_accuracy", 
        hue="dataset", 
        style="type", 
        markers=markers,
        s=150, 
        alpha=0.8,
        palette="deep"
    )

    # 4. Draw Arrows (Vectors) to show the "Change"
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

    # Labels
    plt.title("The Trade-off: How Privacy & Utility Change", fontsize=16)
    plt.xlabel("Re-identification Risk (%) (Lower is Better)", fontsize=12)
    plt.ylabel("Model Accuracy (Higher is Better)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(-5, 105)

    plt.tight_layout()
    plt.savefig("tradeoff_chart_final.png")
    print("[x] Chart saved as tradeoff_chart_final.png")


# ==========================================
# FROM visualize_results.py
# ==========================================
def generate_simple_tradeoff():
    print("Generating Simple Trade-off Chart...")
    # Load data
    try:
        df = pd.read_csv("MASTER_RESULTS.csv")
    except FileNotFoundError:
        print("Error: MASTER_RESULTS.csv not found.")
        return

    # Filter out rows where privacy risk failed (N/A)
    df = df.dropna(subset=['privacy_risk'])

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Create Scatter Plot
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

if __name__ == "__main__":
    generate_tradeoff_with_arrows()
    generate_simple_tradeoff()