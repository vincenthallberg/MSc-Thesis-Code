# scout_ml/tuning.py
import optuna

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from .utils import make_weights

def lgbm_objective(trial, X, y, preprocessor):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_tr_proc  = preprocessor.fit_transform(X_tr, y_tr)
    X_val_proc = preprocessor.transform(X_val)

    class_weights, sw_tr, _ = make_weights(y_tr)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "class_weight": class_weights,
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 5e-2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }

    model = LGBMClassifier(**params, random_state=42)

    model.fit(
        X_tr_proc, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val_proc, y_val)],
        eval_metric="auc",
    )
    p = model.predict_proba(X_val_proc)[:, 1]
    return roc_auc_score(y_val, p)

def xgb_objective(trial, X, y, preprocessor):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_tr_proc  = preprocessor.fit_transform(X_tr, y_tr)
    X_val_proc = preprocessor.transform(X_val)

    _, sw_tr, spw = make_weights(y_tr)

    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": spw,
        "eta": trial.suggest_float("eta", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 7, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    model = xgb.XGBClassifier(**params, n_estimators=1000, use_label_encoder=False, random_state=42)
    model.fit(
        X_tr_proc, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val_proc, y_val)],
        verbose=False
    )
    p = model.predict_proba(X_val_proc)[:, 1]
    return roc_auc_score(y_val, p)

def catboost_objective(trial, X, y, preprocessor):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_tr_proc  = preprocessor.fit_transform(X_tr, y_tr)
    X_val_proc = preprocessor.transform(X_val)

    class_weights, sw_tr, _ = make_weights(y_tr)

    params = {
        "objective": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1000,
        "class_weights": [class_weights[0], class_weights[1]],
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 9, 15),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }

    model = CatBoostClassifier(**params, random_state=42, verbose=0)
    model.fit(
        X_tr_proc, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val_proc, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    p = model.predict_proba(X_val_proc)[:, 1]
    return roc_auc_score(y_val, p)