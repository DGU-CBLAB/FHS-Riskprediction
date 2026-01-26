#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# In[49]:


# FEATURES = {
#     "af": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","PRSice2"],
#     "chd": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","LDpred"],
#     "chf": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","PRSice2"],
#     "dem": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","LDpred"],
#     "dia": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","LDpred"],
#     "stroke": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","Lasso"]
# }

# FEATURES = {
#     "af": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
#     "chd": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
#     "chf": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
#     "dem": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
#     "dia": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
#     "stroke": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"]
# }

#prs
FEATURES = {
    "af": ["AGE","SEX","PRSice2"],
    "chd": ["AGE","SEX","LDpred"],
    "chf": ["AGE","SEX","PRSice2"],
    "dem": ["AGE","SEX","LDpred"],
    "dia": ["AGE","SEX","LDpred"],
    "stroke": ["AGE","SEX","Lasso"]
}

FILES = {
    "dia":    "diabet_data_path.csv",
    "chf":    "chf_data_path.csv",
    "chd":    "chd_data_path.csv",
    "stroke": "stroke_data_path.csv",
    "af":     "af_data_path.csv",
    "dem":    "dem_data_path.csv"
}

TARGET_NAME = {k: "Disease_status" for k in FILES.keys()}



# In[47]:


import itertools
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
def get_metrics(y_true, y_pred, y_proba):
    auc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    lr_plus = sensitivity / (1 - specificity + 1e-8)
    lr_minus = (1 - sensitivity) / (specificity + 1e-8)
    dor = lr_plus / (lr_minus + 1e-8)

    return auc, auprc, sensitivity, specificity, dor, lr_plus, lr_minus
# --------------------------
# Hyperparameters for RF
# --------------------------
rf_params = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy"]
}

# --------------------------
# Sampling
# --------------------------
sampling_methods = ["none", "undersample", "smote", "class_weight"]
sampling_ratios = [0.6, 0.8, 1.0]   # none 제외

# --------------------------
# Outer fold
# --------------------------
def run_outer_fold(outer_fold, train_idx, test_idx, X, y,
                   rf_params, sampling_methods, sampling_ratios):
    
    X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
    y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]

    # Scaling
    scaler = StandardScaler()
    cols_to_scale = [c for c in X.columns if c not in ["AGE", "SEX", "DLVH"]]
    X_outer_train_scaled = X_outer_train.copy()
    X_outer_test_scaled = X_outer_test.copy()
    X_outer_train_scaled[cols_to_scale] = scaler.fit_transform(X_outer_train[cols_to_scale])
    X_outer_test_scaled[cols_to_scale] = scaler.transform(X_outer_test[cols_to_scale])

    # Inner CV
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    inner_log = []
    avg_auc_list = []

    for sampling_method in sampling_methods:
        for ratio in sampling_ratios if sampling_method in ["undersample", "smote"] else [None]:
            for (n_est, max_d, min_split, min_leaf, max_feat, crit) in itertools.product(
                rf_params["n_estimators"],
                rf_params["max_depth"],
                rf_params["min_samples_split"],
                rf_params["min_samples_leaf"],
                rf_params["max_features"],
                rf_params["criterion"]
            ):
                auc_values = []

                for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner.split(X_outer_train_scaled, y_outer_train), 1):
                    X_inner_train = X_outer_train_scaled.iloc[inner_train_idx]
                    y_inner_train = y_outer_train.iloc[inner_train_idx]
                    X_inner_val = X_outer_train_scaled.iloc[inner_val_idx]
                    y_inner_val = y_outer_train.iloc[inner_val_idx]

                    if sampling_method == "undersample":
                        sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
                        X_res, y_res = sampler.fit_resample(X_inner_train, y_inner_train)
                        class_weight = None
                    elif sampling_method == "smote":
                        sampler = SMOTE(sampling_strategy=ratio, random_state=42)
                        X_res, y_res = sampler.fit_resample(X_inner_train, y_inner_train)
                        class_weight = None
                    elif sampling_method == "class_weight":
                        X_res, y_res = X_inner_train, y_inner_train
                        class_weight = "balanced_subsample"
                    else:  # none
                        X_res, y_res = X_inner_train, y_inner_train
                        class_weight = None

                    model = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        max_features=max_feat,
                        criterion=crit,
                        class_weight=class_weight,
                        n_jobs=8, 
                        random_state=42
                    )
                    model.fit(X_res, y_res)

                    val_proba = model.predict_proba(X_inner_val)[:, 1]
                    val_pred = (val_proba > 0.5).astype(int)
                    auc, auprc, sen, spe, dor, lr_p, lr_m = get_metrics(y_inner_val, val_pred, val_proba)
                    auc_values.append(auc)

                    inner_log.append({
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "sampling_method": sampling_method,
                        "sampling_ratio": ratio,
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "min_samples_split": min_split,
                        "min_samples_leaf": min_leaf,
                        "max_features": max_feat,
                        "criterion": crit,
                        "class_weight": class_weight,
                        "AUC": auc,
                        "AUPRC" : auprc,
                        "sensitivity": sen,
                        "specificity": spe,
                        "DOR": dor,
                        "LR+": lr_p,
                        "LR-": lr_m
                    })

                avg_auc_list.append({
                    "sampling_method": sampling_method,
                    "sampling_ratio": ratio,
                    "n_estimators": n_est,
                    "max_depth": max_d,
                    "min_samples_split": min_split,
                    "min_samples_leaf": min_leaf,
                    "max_features": max_feat,
                    "criterion": crit,
                    "class_weight": class_weight,
                    "mean_AUC": np.mean(auc_values)
                })

    # --------------------------
    # Best model
    # --------------------------
    best_model_info = max(avg_auc_list, key=lambda x: x["mean_AUC"])
    best_model_info["outer_fold"] = outer_fold

    # --------------------------
    # Outer test
    # --------------------------
    sm = best_model_info["sampling_method"]
    r = best_model_info["sampling_ratio"]
    cw = best_model_info["class_weight"]

    if sm == "undersample":
        sampler = RandomUnderSampler(sampling_strategy=r, random_state=42)
        X_res, y_res = sampler.fit_resample(X_outer_train_scaled, y_outer_train)
    elif sm == "smote":
        sampler = SMOTE(sampling_strategy=r, random_state=42)
        X_res, y_res = sampler.fit_resample(X_outer_train_scaled, y_outer_train)
    else:
        X_res, y_res = X_outer_train_scaled, y_outer_train

    model = RandomForestClassifier(
        n_estimators=best_model_info["n_estimators"],
        max_depth=best_model_info["max_depth"],
        min_samples_split=best_model_info["min_samples_split"],
        min_samples_leaf=best_model_info["min_samples_leaf"],
        max_features=best_model_info["max_features"],
        criterion=best_model_info["criterion"],
        class_weight=cw,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_res, y_res)
    test_proba = model.predict_proba(X_outer_test_scaled)[:, 1]
    test_pred = (test_proba > 0.5).astype(int)
    auc, auprc, sen, spe, dor, lr_p, lr_m = get_metrics(y_outer_test, test_pred, test_proba)

    outer_test_metrics = {
        "outer_fold": outer_fold,
        "AUC": auc,
        "AUPRC": auprc,
        "sensitivity": sen,
        "specificity": spe,
        "DOR": dor,
        "LR+": lr_p,
        "LR-": lr_m
    }

    return {
        "inner_log": inner_log,
        "best_model": best_model_info,
        "outer_test": outer_test_metrics
    }



# In[50]:


disease_list = ["dia", "chf", "chd", "stroke", "af", "dem"]
#disease_list = ["chf", "chd"]

for disease in disease_list:    
    print(f"\n\n\n==============================")
    print(f"### disease: {disease} ###")
    print("==============================")
    
    df = pd.read_csv(FILES[disease])
    features = FEATURES[disease]
    target = TARGET_NAME[disease]
    
    df_sub = df[features + [target]].dropna()
    
    X = df_sub[features]
    y = df_sub[target]
    # --------------------------
    # Outer fold 병렬 실행
    # --------------------------
    outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    results = Parallel(n_jobs=4, verbose=10, backend="loky")(
        delayed(run_outer_fold)(
            outer_fold,
            train_idx,
            test_idx,
            X, y,
            rf_params, sampling_methods, sampling_ratios
        )
        for outer_fold, (train_idx, test_idx) in enumerate(outer.split(X, y), 1)
    )
    
    inner_log = []
    outer_best_log = []
    outer_test_log = []
    
    for r in results:
        inner_log.extend(r["inner_log"])
        outer_best_log.append(r["best_model"])
        outer_test_log.append(r["outer_test"])
    
    SAVE_DIR = f"/save_path/{disease}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    pd.DataFrame(inner_log).to_csv(f"{SAVE_DIR}/{disease}_rf_nested_cv_all_results.csv", index=False)
    pd.DataFrame(outer_best_log).to_csv(f"{SAVE_DIR}/{disease}_rf_nested_cv_best_per_fold.csv", index=False)
    pd.DataFrame(outer_test_log).to_csv(f"{SAVE_DIR}/{disease}_rf_nested_cv_outer_test_results.csv", index=False)
    
    print("\n=== randomforest Nested CV complete ===")
    print("save path:", SAVE_DIR)

