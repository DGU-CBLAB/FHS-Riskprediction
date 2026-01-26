#!/usr/bin/env python
# coding: utf-8

# In[4]:


import itertools
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from catboost import CatBoostClassifier
from joblib import Parallel, delayed

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


# In[2]:


import itertools
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from catboost import CatBoostClassifier
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
# Hyperparameters for CatBoost
# --------------------------
cat_params = {
    "iterations": [100, 300, 500],
    "depth": [5, 10],
    "learning_rate": [0.1, 0.3],
    "l2_leaf_reg": [1, 3],
}

sampling_methods = ["none", "undersample", "smote", "class_weight"]
sampling_ratios = [0.6, 0.8, 1.0]

def run_outer_fold_cat(outer_fold, train_idx, test_idx, X, y,
                       cat_params, sampling_methods, sampling_ratios, gpu_id):
    
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
            for (iters, depth, lr, l2) in itertools.product(
                cat_params["iterations"],
                cat_params["depth"],
                cat_params["learning_rate"],
                cat_params["l2_leaf_reg"]
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
                        auto_class_weights = None
                    elif sampling_method == "smote":
                        sampler = SMOTE(sampling_strategy=ratio, random_state=42)
                        X_res, y_res = sampler.fit_resample(X_inner_train, y_inner_train)
                        auto_class_weights = None
                    elif sampling_method == "class_weight":
                        X_res, y_res = X_inner_train, y_inner_train
                        auto_class_weights = "Balanced"
                    else:
                        X_res, y_res = X_inner_train, y_inner_train
                        auto_class_weights = None

                    model = CatBoostClassifier(
                        iterations=iters,
                        depth=depth,
                        learning_rate=lr,
                        l2_leaf_reg=l2,
                        task_type="CPU",
                        auto_class_weights=auto_class_weights,
                        verbose=0,
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
                        "iterations": iters,
                        "depth": depth,
                        "learning_rate": lr,
                        "l2_leaf_reg": l2,
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
                    "iterations": iters,
                    "depth": depth,
                    "learning_rate": lr,
                    "l2_leaf_reg": l2,
                    "mean_AUC": np.mean(auc_values)
                })

    best_model_info = max(avg_auc_list, key=lambda x: x["mean_AUC"])
    best_model_info["outer_fold"] = outer_fold

    sm = best_model_info["sampling_method"]
    r = best_model_info["sampling_ratio"]

    if sm == "undersample":
        sampler = RandomUnderSampler(sampling_strategy=r, random_state=42)
        X_res, y_res = sampler.fit_resample(X_outer_train_scaled, y_outer_train)
        auto_class_weights = None
    elif sm == "smote":
        sampler = SMOTE(sampling_strategy=r, random_state=42)
        X_res, y_res = sampler.fit_resample(X_outer_train_scaled, y_outer_train)
        auto_class_weights = None

    elif sm == "class_weight":
        X_res, y_res = X_inner_train, y_inner_train
        auto_class_weights = "Balanced"
    else:
        X_res, y_res = X_outer_train_scaled, y_outer_train
        auto_class_weights = None

    model = CatBoostClassifier(
        iterations=best_model_info["iterations"],
        depth=best_model_info["depth"],
        learning_rate=best_model_info["learning_rate"],
        l2_leaf_reg=best_model_info["l2_leaf_reg"],
        task_type="GPU",
        devices=str(gpu_id),
        auto_class_weights=auto_class_weights,
        verbose=0,
        random_state=42
    )
    model.fit(X_res, y_res)
    test_proba = model.predict_proba(X_outer_test_scaled)[:, 1]
    test_pred = (test_proba > 0.5).astype(int)
    auc, auprc, sen, spe, dor, lr_p, lr_m = get_metrics(y_outer_test, test_pred, test_proba)

    outer_test_metrics = {
        "outer_fold": outer_fold,
        "AUC": auc,
        "AUPRC" : auprc,
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


# In[5]:


disease_list = ["dia", "chf", "chd", "stroke", "af", "dem"]

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
    
    outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    results = Parallel(n_jobs=1, verbose=10, backend="loky")(
        delayed(run_outer_fold_cat)(
            outer_fold,
            train_idx,
            test_idx,
            X, y,
            cat_params, sampling_methods, sampling_ratios,
            gpu_id=i
        )
        for i, (outer_fold, (train_idx, test_idx)) in enumerate(zip(range(1,5), outer.split(X, y)))
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
    
    pd.DataFrame(inner_log).to_csv(f"{SAVE_DIR}/{disease}_catboost_nested_cv_all_results.csv", index=False)
    pd.DataFrame(outer_best_log).to_csv(f"{SAVE_DIR}/{disease}_catboost_nested_cv_best_per_fold.csv", index=False)
    pd.DataFrame(outer_test_log).to_csv(f"{SAVE_DIR}/{disease}_catboost_nested_cv_outer_test_results.csv", index=False)
    
    print("\n=== CatBoost Nested CV complete ===")
    print("Save Path:", SAVE_DIR)


# In[28]:


# --------------------------
# 결과 합치기
# --------------------------


# In[ ]:





# In[ ]:




