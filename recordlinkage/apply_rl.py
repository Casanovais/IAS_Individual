"""Apply Record Linkage
This script measures the privacy risk of the synthetic data using unique-set matching.
"""
import os
import sys
import pandas as pd
import numpy as np

# Import from our fixed preprocessing script
sys.path.append('./')
from dataprep.preprocessing import read_data, quasi_identifiers

def save_results(file, args, results):
    """Save the privacy risk results to CSV"""
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    output_filename = f"{args.output_folder}/{file.replace('.csv', '_privacy.csv')}"
    results.to_csv(output_filename, index=False)

def perform_exact_matching(original_df, synthetic_df, qis):
    """
    Calculates privacy risk using a UNIQUE check.
    This prevents memory explosion by removing duplicates from the original comparison set.
    """
    
    # 1. Standardize types to string to avoid mismatch errors
    for col in qis:
        original_df[col] = original_df[col].astype(str)
        synthetic_df[col] = synthetic_df[col].astype(str)

    # 2. Create a "Lookup Table" of existing real people
    # We drop duplicates here. We don't care if there are 100 real people with these traits
    # or just 1. We just need to know that this combination EXISTS.
    # This prevents the N x M cartesian explosion.
    real_people_lookup = original_df[qis].drop_duplicates()

    # 3. Perform Inner Join against the unique lookup table
    # This will keep every synthetic record that has a match in the lookup table.
    risky_records = pd.merge(synthetic_df, real_people_lookup, on=qis, how='inner')
    
    # Calculate Risk Metrics
    n_synthetic = len(synthetic_df)
    n_matches = len(risky_records)
    
    # Risk = Percentage of synthetic records that match a real trait combination
    risk_percentage = (n_matches / n_synthetic) * 100 if n_synthetic > 0 else 0
    
    return pd.DataFrame({
        'filename': ['current'],
        'n_records': [n_synthetic],
        'n_matches': [n_matches],
        'risk_percentage': [risk_percentage]
    })

def modeling_data(file, args):
    """Main function called by the worker"""
    print(f'Processing Privacy Check (Safe Mode): {file}')
    
    try:
        # 1. Identify Dataset
        ds_name = file.split("_")[0]
        if ds_name == 'bank': ds_name = 'bankmarketing'
        
        all_data = read_data()
        all_qis = quasi_identifiers()
        
        if ds_name not in all_data['name']:
            print(f"Error: Dataset {ds_name} not found.")
            return False
            
        names_index = all_data['name'].index(ds_name)
        original_data = all_data['data'][names_index]
        current_qis = all_qis[names_index]

        # 2. Read Synthetic Data
        sep = ',' if ds_name != 'compas' else ';'
        synthetic_data = pd.read_csv(f'{args.input_folder}/{file}', sep=sep)

        # 3. Cleanup
        if 'single_out' in synthetic_data.columns:
            del synthetic_data['single_out']
            
        # 4. Validate QIs
        valid_qis = [c for c in current_qis if c in original_data.columns and c in synthetic_data.columns]
        
        if not valid_qis:
            print("Error: No common Quasi-Identifiers found.")
            return False

        # 5. Run Safe Attack
        results = perform_exact_matching(original_data, synthetic_data, valid_qis)
        
        results['dataset'] = ds_name
        results['file'] = file
        save_results(file, args, results)
        
        print(f"[x] Privacy Check Complete for {file}. Risk: {results['risk_percentage'][0]:.2f}%")
        return True

    except Exception as e:
        print(f"Error checking privacy for {file}: {e}")
        import traceback
        traceback.print_exc()
        return False