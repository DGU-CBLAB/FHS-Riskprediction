#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
from xgboost import XGBClassifier
from joblib import Parallel, delayed
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

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


# In[5]:


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
    

def run_xgb_nested_cv_cpu(
    X,
    y,
    sampling_methods=["none", "undersample", "smote", "class_weight"],
    sampling_ratios=[0.6, 0.8, 1.0],
):

    xgb_params = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 10],
        "learning_rate": [0.1, 0.3],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.7, 1.0]
    }

    outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    all_inner_logs = []
    all_best_models = []
    all_outer_tests = []

    # ============================================================
    # OUTER LOOP
    # ============================================================
    for outer_fold, (train_idx, test_idx) in enumerate(outer.split(X, y), 1):

        X_train_outer = X.iloc[train_idx].copy()
        X_test_outer = X.iloc[test_idx].copy()
        y_train_outer = y.iloc[train_idx]
        y_test_outer = y.iloc[test_idx]

        # Scaling
        scaler = StandardScaler()
        cols_to_scale = [c for c in X.columns if c not in ["AGE", "SEX", "DLVH"]]
        X_train_outer[cols_to_scale] = scaler.fit_transform(X_train_outer[cols_to_scale])
        X_test_outer[cols_to_scale] = scaler.transform(X_test_outer[cols_to_scale])

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        inner_log = []
        avg_auc_list = []

        # ============================================================
        # INNER LOOP 
        # ============================================================
        for sampling_method in sampling_methods:

            ratios = sampling_ratios if sampling_method in ["undersample", "smote"] else [None]

            for ratio in ratios:
                for (n_est, max_d, lr, subs, col) in itertools.product(
                    xgb_params["n_estimators"],
                    xgb_params["max_depth"],
                    xgb_params["learning_rate"],
                    xgb_params["subsample"],
                    xgb_params["colsample_bytree"],
                ):

                    inner_auc_values = []

                    # INNER FOLD
                    for inner_fold, (tr_idx, val_idx) in enumerate(inner.split(X_train_outer, y_train_outer), 1):

                        X_tr = X_train_outer.iloc[tr_idx]
                        y_tr = y_train_outer.iloc[tr_idx]
                        X_val = X_train_outer.iloc[val_idx]
                        y_val = y_train_outer.iloc[val_idx]

                        if sampling_method == "undersample":
                            sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
                            X_res, y_res = sampler.fit_resample(X_tr, y_tr)

                        elif sampling_method == "smote":
                            sampler = SMOTE(sampling_strategy=ratio, random_state=42)
                            X_res, y_res = sampler.fit_resample(X_tr, y_tr)

                        else:  # none or class_weight
                            X_res, y_res = X_tr, y_tr

                        model = XGBClassifier(
                            n_estimators=n_est,
                            max_depth=max_d,
                            learning_rate=lr,
                            subsample=subs,
                            colsample_bytree=col,
                            tree_method="hist",
                            device="cpu",
                            eval_metric="logloss",
                            random_state=42,
                        )

                        # class_weight
                        if sampling_method == "class_weight":
                            model.set_params(scale_pos_weight=(len(y_res) - sum(y_res)) / sum(y_res))

                        model.fit(X_res, y_res)

                        val_proba = model.predict_proba(X_val)[:, 1]
                        val_pred = (val_proba > 0.5).astype(int)
                        auc, auprc, sen, spe, dor, lr_p, lr_m = get_metrics(y_val, val_pred, val_proba)

                        inner_auc_values.append(auc)

                        inner_log.append({
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "sampling_method": sampling_method,
                            "sampling_ratio": ratio,
                            "n_estimators": n_est,
                            "max_depth": max_d,
                            "learning_rate": lr,
                            "subsample": subs,
                            "colsample_bytree": col,
                            "AUC": auc,
                            "AUPRC" : auprc,
                            "sensitivity": sen,
                            "specificity": spe,
                            "DOR": dor,
                            "LR+": lr_p,
                            "LR-": lr_m
                        })

                    # Mean AUC for this setting
                    avg_auc_list.append({
                        "outer_fold": outer_fold,
                        "sampling_method": sampling_method,
                        "sampling_ratio": ratio,
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "learning_rate": lr,
                        "subsample": subs,
                        "colsample_bytree": col,
                        "mean_AUC": np.mean(inner_auc_values),
                    })

        # ============================================================
        # BEST MODEL SELECTED
        # ============================================================
        best_info = max(avg_auc_list, key=lambda x: x["mean_AUC"])
        all_best_models.append(best_info)

        # ============================================================
        # OUTER TEST ON BEST MODEL
        # ============================================================
        sm = best_info["sampling_method"]
        r = best_info["sampling_ratio"]

        # sampling 
        if sm == "undersample":
            sampler = RandomUnderSampler(sampling_strategy=r, random_state=42)
            X_res, y_res = sampler.fit_resample(X_train_outer, y_train_outer)
        elif sm == "smote":
            sampler = SMOTE(sampling_strategy=r, random_state=42)
            X_res, y_res = sampler.fit_resample(X_train_outer, y_train_outer)
        else:
            X_res, y_res = X_train_outer, y_train_outer

        final_model = XGBClassifier(
            n_estimators=best_info["n_estimators"],
            max_depth=best_info["max_depth"],
            learning_rate=best_info["learning_rate"],
            subsample=best_info["subsample"],
            colsample_bytree=best_info["colsample_bytree"],
            tree_method="hist",
            device="cpu",
            eval_metric="logloss",
            random_state=42,
        )

        if sm == "class_weight":
            final_model.set_params(scale_pos_weight=(len(y_res) - sum(y_res)) / sum(y_res))

        final_model.fit(X_res, y_res)

        test_proba = final_model.predict_proba(X_test_outer)[:, 1]
        test_pred = (test_proba > 0.5).astype(int)
        auc, auprc, sen, spe, dor, lr_p, lr_m = get_metrics(y_test_outer, test_pred, test_proba)

        all_outer_tests.append({
            "outer_fold": outer_fold,
            "AUC": auc,
            "AUPRC" : auprc,
            "sensitivity": sen,
            "specificity": spe,
            "DOR": dor,
            "LR+": lr_p,
            "LR-": lr_m
        })

        all_inner_logs.extend(inner_log)

    # ------------------------------------
    # RETURN STRUCTURE
    # ------------------------------------
    return {
        "inner_log": pd.DataFrame(all_inner_logs),
        "best_model": pd.DataFrame(all_best_models),
        "outer_test": pd.DataFrame(all_outer_tests)
    }


# In[8]:


import os
import time

if __name__ == "__main__":
    disease_list = ["dia", "chf", "chd", "stroke", "af", "dem"]
#    disease_list = ["chf", "chd"]

    for disease in disease_list:
        start_time = time.time()

        df = pd.read_csv(FILES[disease])
        features = FEATURES[disease]
        target = TARGET_NAME[disease]
        
        df_sub = df[features + [target]].dropna()
        
        X = df_sub[features]
        y = df_sub[target]
        
        print(">>> XGBoost Nested CV ")
    
        res = run_xgb_nested_cv_cpu(
            X=X,
            y=y
            # sampling_methods=sampling_methods,
            # sampling_ratios=sampling_ratios
        )
    
        df_inner = res["inner_log"]
        df_best = res["best_model"]
        df_outer = res["outer_test"]
    
        SAVE_DIR = f"/save_path/{disease}"
        os.makedirs(SAVE_DIR, exist_ok=True)
    
        df_inner.to_csv(f"{SAVE_DIR}/{disease}_xgb_nested_cv_all_results.csv", index=False)
        df_best.to_csv(f"{SAVE_DIR}/{disease}_xgb_nested_cv_best_per_fold.csv", index=False)
        df_outer.to_csv(f"{SAVE_DIR}/{disease}_xgb_nested_cv_outer_test_results.csv", index=False)
    
        print(">>> save complete:")
        print(f" - {SAVE_DIR}/{disease}_xgb_nested_cv_all_results.csv")
        print(f" - {SAVE_DIR}/{disease}_xgb_nested_cv_best_per_fold.csv")
        print(f" - {SAVE_DIR}/{disease}_xgb_nested_cv_outer_test_results.csv")
    
        end_time = time.time()
        elapsed = end_time - start_time
    
        print(f"\n>>> total time: {elapsed:.2f}초 ({elapsed/60:.2f}분)")


# In[ ]:




