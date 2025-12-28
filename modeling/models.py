"""Predictive performance
This script will test the predictive performance of the data sets.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference)

# %% evaluate a model

def evaluate_model(x_train, x_test, y_train, y_test, x_test_sa):
    """Evaluation without Fairlearn mitigation

    Args:
        x_train (pd.DataFrame): dataframe for train
        x_test (pd.DataFrame): dataframe for test
        y_train (np.int64): target variable for train
        y_test (np.int64): target variable for test
        x_test_sa (pd.DataFrame): sensitive attributes for test
    Returns:
        tuple: dictionary with validation, train and test results
    """

    validation_all = pd.DataFrame()
    seed = np.random.seed(1234)

    # initiate models
    rf = RandomForestClassifier(random_state=seed, n_jobs=1)
    booster = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=seed,
        n_jobs=1)
    reg = LogisticRegression(random_state=seed, n_jobs=1)

    pipe_rf = make_pipeline(rf)
    pipe_booster = make_pipeline(booster)
    pipe_reg = make_pipeline(reg)
    pipeline = [pipe_rf, pipe_booster, pipe_reg]

    # set parameterisation (Lightweight version)
    param_grid_rf = {
        'randomforestclassifier__n_estimators': [50, 100],
        'randomforestclassifier__max_depth': [4, 10]
    }
    param_grid_booster = {
        'xgbclassifier__n_estimators': [50, 100],
        'xgbclassifier__max_depth': [4, 10],
        'xgbclassifier__learning_rate': [0.1]
    }
    param_grid_reg = {
        'logisticregression__C': [1.0, 0.1],
        'logisticregression__max_iter': [1000]
    }
    param_grids = [param_grid_rf, param_grid_booster, param_grid_reg]
    
    # FIXED: Updated for sklearn 1.6+
    scoring = {
        'gmean': make_scorer(geometric_mean_score),
        'acc': 'accuracy',
        'bal_acc': 'balanced_accuracy',
        'f1': 'f1',
        'f1_weighted': 'f1_weighted',
        'roc_auc_curve': make_scorer(roc_auc_score, max_fpr=0.001, response_method="predict_proba")
    }
    
    model = ['Random Forest', 'XGBoost', 'Logistic Regression']
    
    print("Start modeling with CV")
    for idx, pipe in enumerate(pipeline):
        grid_search = GridSearchCV(
            pipe,
            param_grids[idx],
            cv=StratifiedKFold(n_splits=3),
            scoring=scoring,
            return_train_score=True,
            refit='roc_auc_curve',
            n_jobs=1).fit(x_train, y_train)
            
        validation = pd.DataFrame(grid_search.cv_results_)
        validation['model'] = model[idx]
        
        validation_all = validation if validation_all.empty else pd.concat(
                [validation_all, validation])
    
    best_idx = validation_all['mean_test_acc'].idxmax()
    best_model_row = validation_all.iloc[[best_idx]].reset_index()

    best_model_name = best_model_row['model'][0]
    
    # FIXED: Cast parameters to int
    if best_model_name == 'Random Forest':
        best_clf = RandomForestClassifier(
            n_estimators=int(best_model_row['param_randomforestclassifier__n_estimators'][0]),
            max_depth=int(best_model_row['param_randomforestclassifier__max_depth'][0]),
            random_state=seed, n_jobs=1)
    elif best_model_name == 'XGBoost':
        best_clf = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=int(best_model_row['param_xgbclassifier__n_estimators'][0]),
            max_depth=int(best_model_row['param_xgbclassifier__max_depth'][0]),
            learning_rate=best_model_row['param_xgbclassifier__learning_rate'][0],
            random_state=seed, n_jobs=1)
    else:
        best_clf = LogisticRegression(
            C=best_model_row['param_logisticregression__C'][0],
            max_iter=int(best_model_row['param_logisticregression__max_iter'][0]),
            random_state=seed, n_jobs=1)
            
    print(best_clf)
    
    # retrain the grid search model
    best_clf.fit(x_train, y_train)

    score_cv = {
        'params': [], 'model': [],
        'test_accuracy': [], 'test_f1_weighted': [], 'test_gmean': [], 'test_roc_auc': [],
        'demographic_parity': [], 'equalized_odds': []
    }

    print("Predict in out of sample")
    clf = best_clf.predict(x_test)
    
    score_cv['params'].append(best_model_row['params'][0])
    score_cv['model'].append(best_model_row['model'][0])
    score_cv['test_accuracy'].append(accuracy_score(y_test, clf))
    score_cv['test_f1_weighted'].append(f1_score(y_test, clf, average='weighted'))
    score_cv['test_gmean'].append(geometric_mean_score(y_test, clf))
    score_cv['test_roc_auc'].append(roc_auc_score(y_test, clf))
    
    # Fairness metrics still apply even on standard models
    score_cv['demographic_parity'].append(demographic_parity_difference(
        y_test, clf, sensitive_features=x_test_sa))
    score_cv['equalized_odds'].append(equalized_odds_difference(
        y_test, clf, sensitive_features=x_test_sa))

    score_cv = pd.DataFrame.from_dict(score_cv, orient='index')
    score_cv = score_cv.transpose()

    return [validation, score_cv]