import pandas as pd
# Load your results
df = pd.read_csv("MASTER_RESULTS.csv")

# Filter for German
german = df[df['dataset'] == 'german']

# Check columns like 'test_roc_auc' or 'test_f1_weighted'
print(german[['model', 'filename', 'test_accuracy', 'test_f1_weighted', 'equalized_odds']])