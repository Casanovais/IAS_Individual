"""This script will modeling the data variants
"""
# pylint: disable=import-error
# pylint: disable=wrong-import-position
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from models import evaluate_model
sys.path.append('./')
from dataprep.preprocessing import read_data, get_indexes, sensitive_attributes
from fairmodels.fairlearn import evaluate_fairlearn
from fairmodels.fairmask import evaluate_fairmask

# --- 1. CONFIGURATION ---
TARGET_MAP = {
    'adult': 'income',
    'german': 'status',
    'bankmarketing': 'y',
    'diabets': 'readmitted',
    'heart': 'num',
    'students': 'G3'
}

def save_results(file, args, results):
    """Create a folder if dooes't exist and save results"""
    if args.fairtype!='fairgbm':
        output_folder_val = (f'{args.output_folder}/validation')
        if not os.path.exists(output_folder_val):
            os.makedirs(output_folder_val)
        results[0].to_csv(f'{output_folder_val}/{file}', index=False)
    
    output_folder_test = (f'{args.output_folder}/test')
    if not os.path.exists(output_folder_test):
        os.makedirs(output_folder_test)
    results[1].to_csv(f'{output_folder_test}/{file}', index=False)

def priv_groups(x_train, x_test, file):
    protected_classes = {'ds_name': ['adult','german','bankmarketing'],
                                'privileged_range_min': [25, 25, 25, 17],
                                'privileged_range_max': [60, 26, 60, 18]}

    if 'age' in x_train.columns and file in protected_classes['ds_name']:
        pc_idx = protected_classes['ds_name'].index(file)
        x_train.loc[:, 'age'] = x_train.apply(lambda x: int(protected_classes['privileged_range_min'][pc_idx] <= x['age'] <= protected_classes['privileged_range_max'][pc_idx]), axis=1)
        x_test.loc[:, 'age'] = x_test.apply(lambda x: int(protected_classes['privileged_range_min'][pc_idx] <= x['age'] <= protected_classes['privileged_range_max'][pc_idx]), axis=1)
    
    # Handle race/marital status encoding
    if file in ['adult', 'compas', 'diabets', 'lawschool']:
        if 'race' in x_train.columns:
            if x_train['race'].dtype == 'object':
                value = Counter(x_train.race).most_common()[0]
                x_train.loc[:, 'race'] = np.where(x_train['race']==value[0], 1, 0)
                x_test.loc[:, 'race'] = np.where(x_test['race']==value[0], 1, 0)

    if 'bankmarketing' in file:
        if 'marital' in x_train.columns:
            if x_train['marital'].dtype == 'object':
                value = Counter(x_train.marital).most_common()[0]
                x_train.loc[:, 'marital'] = np.where(x_train['marital']==value[0], 1, 0)
                x_test.loc[:, 'marital'] = np.where(x_test['marital']==value[0], 1, 0)

    return x_train, x_test

def binarize_target(df, target_col, ds_name):
    """Forces the target column to be strictly 0 and 1."""
    y = df[target_col].copy()
    
    if ds_name == 'heart':
        y = y.apply(lambda x: 1 if float(x) > 0 else 0)
    elif ds_name == 'students':
        y = y.apply(lambda x: 1 if float(x) >= 10 else 0)
    elif ds_name == 'diabets':
        y = y.apply(lambda x: 0 if str(x).upper() == 'NO' else 1)
    elif ds_name == 'german':
        y = y.apply(lambda x: 1 if int(x) == 1 else 0)
    else:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        
    return y.astype(int)

def modeling_data(file, args):
    """Apply predictive performance."""
    print(f'Processing: {file}')
    
    # 1. SETUP
    indexes = get_indexes()
    all_data = read_data()
    
    # --- FIX: Handle filenames with extensions (adult.csv -> adult) ---
    clean_name = file.replace('.csv', '').replace('.data', '')
    ds_name = clean_name.split("_")[0]
    
    if ds_name == 'bank': ds_name = 'bankmarketing'
    # -----------------------------------------------------------------
    
    target_col = TARGET_MAP.get(ds_name)
    if not target_col:
        print(f"Skipping unsupported file: {file}")
        return False

    names_index = all_data['name'].index(ds_name)
    index = indexes[names_index]
    test_data = all_data['data'][names_index]
    
    # Read Data
    # For Original, we just copy the test_data (which holds the full DF in preprocessing.py)
    # But we need to handle if it's reading from file for synthetic
    if args.datatype == 'Original':
        data = test_data.copy()
    else:
        sep = ',' if ds_name != 'compas' else ';'
        data = pd.read_csv(f'{args.input_folder}/{file}', sep=sep)

    # CLEANUP
    if 'single_out' in data.columns:
        del data['single_out']
    
    # Ensure target exists
    if target_col not in data.columns:
        print(f"Warning: Target '{target_col}' not found. Using last column.")
        target_col = data.columns[-1]

    # --- CRITICAL FIX: BINARIZE TARGETS ---
    data[target_col] = binarize_target(data, target_col, ds_name)
    test_data[target_col] = binarize_target(test_data, target_col, ds_name)
    # --------------------------------------

    # Encode categorical features
    le = LabelEncoder()
    for col in data.columns:
        if col != target_col and data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))
            
    for col in test_data.columns:
        if col != target_col and test_data[col].dtype == 'object':
            test_data[col] = le.fit_transform(test_data[col].astype(str))

    # Split X/Y
    if args.datatype == 'Original':
        idx = list(set(data.index.tolist()) - set(index))
        train_df = data.iloc[idx]
    else:
        train_df = data

    y_train = train_df[target_col]
    x_train = train_df.drop(columns=[target_col])
    
    y_test = test_data.iloc[index][target_col]
    x_test = test_data.iloc[index].drop(columns=[target_col])

    # Run Model
    all_sa = sensitive_attributes()
    set_sa = all_sa[names_index]

    try:
        if args.fairtype == 'fairlearn':
            if ds_name not in ['dutch', 'heart']:
                x_train, x_test = priv_groups(x_train, x_test, ds_name)
            results = evaluate_fairlearn(x_train, x_test, y_train, y_test, set_sa)
        
        elif args.fairtype == 'fairmask':
            if ds_name not in ['dutch', 'heart']:
                x_train, x_test = priv_groups(x_train, x_test, ds_name)
            results = evaluate_fairmask(x_train, x_test, y_train, y_test, set_sa)
        else:
            # Standard model (No Fairlearn mitigation)
            _, x_test_sa = priv_groups(x_train.copy(), x_test.copy(), ds_name)
            results = evaluate_model(x_train, x_test, y_train, y_test, x_test_sa[set_sa])
            
        save_results(file, args, results)
        print(f"[x] Successfully processed {file}")
        return True
        
    except Exception as e:
        print(f"Error modeling {file}: {e}")
        import traceback
        traceback.print_exc()
        return False