"""Predictive performance
This script will test the predictive performance of the data sets.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Changed to simpler KFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference)

def evaluate_fairlearn(x_train, x_test, y_train, y_test, set_sa):
    """Evaluation"""

    validation_all = pd.DataFrame()
    seed = np.random.seed(1234)

    # drop sensitive attributes for bias mitigation
    train_X = x_train.drop(columns=set_sa, axis=1)
    test_X = x_test.drop(columns=set_sa, axis=1)

    # initiate models
    rf = RandomForestClassifier(random_state=seed, n_jobs=1) # Force single thread
    booster = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=seed,
        n_jobs=1) # Force single thread
    reg = LogisticRegression(random_state=seed, n_jobs=1)

    pipe_rf = make_pipeline(rf)
    pipe_booster = make_pipeline(booster)
    pipe_reg = make_pipeline(reg)
    pipeline = [pipe_rf, pipe_booster, pipe_reg]

    # set parameterisation (REDUCED FOR STABILITY)
    param_grid_rf = {
        'randomforestclassifier__n_estimators': [50, 100], # Removed 250, 500
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
        # CHANGED: n_jobs=1 to save RAM, StratifiedKFold for speed (vs Repeated)
        grid_search = GridSearchCV(
            pipe,
            param_grids[idx],
            cv=StratifiedKFold(n_splits=3), # Reduced splits to 3
            scoring=scoring,
            return_train_score=True,
            refit='roc_auc_curve',
            n_jobs=1).fit(train_X, y_train) 
            
        validation = pd.DataFrame(grid_search.cv_results_)
        validation['model'] = model[idx]
        
        validation_all = validation if validation_all.empty else pd.concat(
                [validation_all, validation])
    
    best_idx = validation_all['mean_test_acc'].idxmax()
    best_model_row = validation_all.iloc[[best_idx]].reset_index()

    best_model_name = best_model_row['model'][0]
    
    # Cast params to int for sklearn compatibility
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
    print("Start modeling with ExponentiatedGradient")
    mitigator = ExponentiatedGradient(best_clf, constraints=EqualizedOdds(), max_iter=20) # Reduced max_iter
    
    mitigator.fit(train_X, y_train, sensitive_features=x_train[set_sa])

    score_cv = {
        'params': [], 'model': [],
        'test_accuracy': [], 'test_f1_weighted': [], 'test_gmean': [], 'test_roc_auc': [],
        'demographic_parity': [], 'equalized_odds': []
    }

    print("Predict in out of sample")
    clf = mitigator.predict(test_X)
    score_cv['params'].append(best_model_row['params'][0])
    score_cv['model'].append(best_model_row['model'][0])
    score_cv['test_accuracy'].append(accuracy_score(y_test, clf))
    score_cv['test_f1_weighted'].append(f1_score(y_test, clf, average='weighted'))
    score_cv['test_gmean'].append(geometric_mean_score(y_test, clf))
    score_cv['test_roc_auc'].append(roc_auc_score(y_test, clf))
    
    score_cv['demographic_parity'].append(demographic_parity_difference(
        y_test, clf, sensitive_features=x_test[set_sa]))
    score_cv['equalized_odds'].append(equalized_odds_difference(
        y_test, clf, sensitive_features=x_test[set_sa]))

    score_cv = pd.DataFrame.from_dict(score_cv, orient='index')
    score_cv = score_cv.transpose()

    return [validation, score_cv]