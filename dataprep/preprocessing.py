import os
import random
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

AVAILABLE_DATASETS = [
    "adult", 
    "german", 
    "bankmarketing", 
    "diabets", 
    "heart", 
    "students"
]

def read_data() -> dict:
    """Read only the available data with correct headers and separators."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(os.path.dirname(script_dir), "input")
    data_sets = {"name": [], "data": []}

    # --- 1. ADULT ---
    if "adult" in AVAILABLE_DATASETS and os.path.exists(f"{input_path}/adult.csv"):
        headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                   'marital-status', 'occupation', 'relationship', 'race', 'gender', 
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        df = pd.read_csv(f"{input_path}/adult.csv", sep=",", names=headers, skipinitialspace=True)
        data_sets['data'].append(df)
        data_sets['name'].append("adult")

    # --- 2. GERMAN ---
    if "german" in AVAILABLE_DATASETS and os.path.exists(f"{input_path}/german.csv"):
        headers = ['checkin_acc', 'duration', 'credit_history', 'purpose', 'amount', 
                   'saving_acc', 'employment-since', 'inst_rate', 'personal_status', 
                   'other_debtors', 'residence-since', 'property', 'age', 'inst_plans', 
                   'housing', 'num_credits', 'job', 'dependents', 'telephone', 'foreign-worker', 'status']
        df = pd.read_csv(f"{input_path}/german.csv", sep=" ", names=headers)
        df['sex'] = df['personal_status'] 
        
        # CRITICAL FIX: Move 'status' (the target) to the very end
        cols = [c for c in df.columns if c != 'status'] + ['status']
        df = df[cols]
        
        data_sets['data'].append(df)
        data_sets['name'].append("german")

    # --- 3. BANK MARKETING ---
    if "bankmarketing" in AVAILABLE_DATASETS and os.path.exists(f"{input_path}/bankmarketing.csv"):
        try:
            df = pd.read_csv(f"{input_path}/bankmarketing.csv", sep=";")
            if 'age' not in df.columns: df = pd.read_csv(f"{input_path}/bankmarketing.csv", sep=",")
            if 'id' in df.columns: del df['id']
            data_sets['data'].append(df)
            data_sets['name'].append("bankmarketing")
        except: pass

    # --- 4. DIABETES ---
    if "diabets" in AVAILABLE_DATASETS and os.path.exists(f"{input_path}/diabets.csv"):
        df = pd.read_csv(f"{input_path}/diabets.csv", sep=",")
        for col in ['encounter_id', 'patient_nbr']:
            if col in df.columns: del df[col]
        data_sets['data'].append(df)
        data_sets['name'].append("diabets")

    # --- 5. HEART DISEASE ---
    if "heart" in AVAILABLE_DATASETS and os.path.exists(f"{input_path}/heart.csv"):
        headers = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
        df = pd.read_csv(f"{input_path}/heart.csv", sep=",", names=headers)
        data_sets['data'].append(df)
        data_sets['name'].append("heart")

    # --- 6. STUDENTS ---
    if "students" in AVAILABLE_DATASETS and os.path.exists(f"{input_path}/students.csv"):
        df = pd.read_csv(f"{input_path}/students.csv", sep=";")
        if 'age' not in df.columns: df = pd.read_csv(f"{input_path}/students.csv", sep=",")
        data_sets['data'].append(df)
        data_sets['name'].append("students")

    return data_sets


def quasi_identifiers() -> list:
    qi_map = {
        "adult": ["age", "gender", "race", "occupation", "native-country"],
        "german": ["age", "purpose", "employment-since", "residence-since", "job", "sex", "foreign-worker"],
        "bankmarketing": ["age", "job", "marital", "education"],
        "diabets": ["age", "gender", "race", "time_in_hospital"],
        "heart": ["age", "thalach", "sex", "cp"],
        "students": ["age", "school", "sex", "address", "famsize", "Mjob", "Fjob"]
    }
    return [qi_map[name] for name in AVAILABLE_DATASETS]


def sensitive_attributes() -> list:
    sa_map = {
        "adult": ["age", "gender", "race"],
        "german": ["age", "sex"],
        "bankmarketing": ["age", "marital"],
        "diabets": ["gender", "race"],
        "heart": ["sex"],
        "students": ["sex", "age"]
    }
    return [sa_map[name] for name in AVAILABLE_DATASETS]


def get_indexes() -> list:
    random.seed(42)
    indexes = []
    data = read_data()
    for each_data in data["data"]:
        random_idx = random.sample(list(each_data.index), k=int(0.2*len(each_data)))
        indexes.append(random_idx)
    return indexes