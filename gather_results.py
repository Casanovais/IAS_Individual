import pandas as pd
import glob
import os

# --- Configuration ---
MODELING_DIR = "./results/test"
PRIVACY_DIR = "./privacy_results"
OUTPUT_FILE = "MASTER_RESULTS.csv"

def get_privacy_risk(base_filename):
    """
    Looks for the corresponding privacy file.
    If it's an ORIGINAL file (no _knn_ in name), Risk is 100% (Worst).
    """
    if "_knn" not in base_filename:
        return 100.0
        
    privacy_filename = base_filename.replace('.csv', '_privacy.csv')
    privacy_path = os.path.join(PRIVACY_DIR, privacy_filename)
    
    if os.path.exists(privacy_path):
        try:
            df = pd.read_csv(privacy_path)
            return round(df['risk_percentage'].iloc[0], 4)
        except Exception:
            return None
    return None

def main():
    print(f"[*] Scanning {MODELING_DIR} for results...")
    
    all_files = glob.glob(os.path.join(MODELING_DIR, "*")) # Scan all, not just .csv in case of no extension
    combined_data = []

    for filepath in all_files:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath): continue
        
        try:
            model_df = pd.read_csv(filepath)
            row = model_df.iloc[0].to_dict()
            
            # --- Fix Dataset Name Parsing ---
            # Remove extension first
            clean_name = filename.replace('.csv', '')
            # Extract name (everything before the first underscore)
            dataset_name = clean_name.split('_')[0]
            
            # 2. Get Matching Privacy Risk
            risk = get_privacy_risk(filename)
            
            # 3. Enrich the row
            row['filename'] = filename
            row['dataset'] = dataset_name
            row['type'] = 'Original' if "_knn" not in filename else 'Synthetic'
            row['privacy_risk'] = risk if risk is not None else "N/A"
            
            combined_data.append(row)
            
        except Exception as e:
            # Skip non-result files
            pass

    if not combined_data:
        print("[!] No data found. Check your folders.")
        return

    # 4. Create Master DataFrame
    master_df = pd.DataFrame(combined_data)
    
    # 5. Reorder and Save
    cols_order = ['dataset', 'type', 'filename', 'model', 'test_accuracy', 'test_f1_weighted', 'demographic_parity', 'equalized_odds', 'privacy_risk']
    existing_cols = [c for c in cols_order if c in master_df.columns]
    remaining_cols = [c for c in master_df.columns if c not in existing_cols]
    
    master_df = master_df[existing_cols + remaining_cols]
    master_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[x] Success! Master table saved to: {OUTPUT_FILE}")
    print(f"    Total Records: {len(master_df)}")

if __name__ == "__main__":
    main()