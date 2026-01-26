#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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


# In[7]:


import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
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


# ============================================================
# Logistic Regression Nested CV (CPU)
# ============================================================
def run_logreg_nested_cv_cpu(
    X,
    y,
    sampling_methods=["none", "undersample", "smote", "class_weight"],
    sampling_ratios=[0.6, 0.8, 1.0],
):

    # --------------------------
    # Hyperparameter Grid
    # --------------------------
    logreg_params = {
        "C": [0.01, 0.1, 1.0, 10],
        "penalty": ["l2"],
        "solver": ["liblinear"],
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

        # --------------------------
        # Scaling (outer train 기준)
        # --------------------------
        scaler = StandardScaler()
        cols_to_scale = [c for c in X.columns if c not in ["AGE", "SEX", "CURRSMK"]]

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
                for C, penalty, solver in itertools.product(
                    logreg_params["C"],
                    logreg_params["penalty"],
                    logreg_params["solver"],
                ):

                    inner_auc_values = []

                    for inner_fold, (tr_idx, val_idx) in enumerate(
                        inner.split(X_train_outer, y_train_outer), 1
                    ):
                        X_tr = X_train_outer.iloc[tr_idx]
                        y_tr = y_train_outer.iloc[tr_idx]
                        X_val = X_train_outer.iloc[val_idx]
                        y_val = y_train_outer.iloc[val_idx]

                        # --------------------------
                        # Sampling
                        # --------------------------
                        if sampling_method == "undersample":
                            sampler = RandomUnderSampler(
                                sampling_strategy=ratio, random_state=42
                            )
                            X_res, y_res = sampler.fit_resample(X_tr, y_tr)
                            class_weight = None

                        elif sampling_method == "smote":
                            sampler = SMOTE(
                                sampling_strategy=ratio, random_state=42
                            )
                            X_res, y_res = sampler.fit_resample(X_tr, y_tr)
                            class_weight = None

                        elif sampling_method == "class_weight":
                            X_res, y_res = X_tr, y_tr
                            class_weight = "balanced"

                        else:  # none
                            X_res, y_res = X_tr, y_tr
                            class_weight = None

                        # --------------------------
                        # Model
                        # --------------------------
                        model = LogisticRegression(
                            C=C,
                            penalty=penalty,
                            solver=solver,
                            class_weight=class_weight,
                            max_iter=500,
                        )
                        model.fit(X_res, y_res)

                        val_proba = model.predict_proba(X_val)[:, 1]
                        val_pred = (val_proba > 0.5).astype(int)
                        auc, auprc, sen, spe, dor, lr_p, lr_m = get_metrics(
                            y_val, val_pred, val_proba
                        )

                        inner_auc_values.append(auc)

                        inner_log.append({
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "sampling_method": sampling_method,
                            "sampling_ratio": ratio,
                            "C": C,
                            "penalty": penalty,
                            "solver": solver,
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
                        "outer_fold": outer_fold,
                        "sampling_method": sampling_method,
                        "sampling_ratio": ratio,
                        "C": C,
                        "penalty": penalty,
                        "solver": solver,
                        "class_weight": class_weight,
                        "mean_AUC": np.mean(inner_auc_values),
                    })

        # ============================================================
        # BEST MODEL
        # ============================================================
        best_info = max(avg_auc_list, key=lambda x: x["mean_AUC"])
        all_best_models.append(best_info)

        # ============================================================
        # OUTER TEST
        # ============================================================
        sm = best_info["sampling_method"]
        r = best_info["sampling_ratio"]

        if sm == "undersample":
            sampler = RandomUnderSampler(sampling_strategy=r, random_state=42)
            X_res, y_res = sampler.fit_resample(X_train_outer, y_train_outer)
            cw = None
        elif sm == "smote":
            sampler = SMOTE(sampling_strategy=r, random_state=42)
            X_res, y_res = sampler.fit_resample(X_train_outer, y_train_outer)
            cw = None
        elif sm == "class_weight":
            X_res, y_res = X_train_outer, y_train_outer
            cw = "balanced"
        else:
            X_res, y_res = X_train_outer, y_train_outer
            cw = None

        final_model = LogisticRegression(
            C=best_info["C"],
            penalty=best_info["penalty"],
            solver=best_info["solver"],
            class_weight=cw,
            max_iter=500,
        )
        final_model.fit(X_res, y_res)

        test_proba = final_model.predict_proba(X_test_outer)[:, 1]
        test_pred = (test_proba > 0.5).astype(int)
        auc, auprc, sen, spe, dor, lr_p, lr_m = get_metrics(
            y_test_outer, test_pred, test_proba
        )

        all_outer_tests.append({
            "outer_fold": outer_fold,
            "AUC": auc,
            "AUPRC": auprc,
            "sensitivity": sen,
            "specificity": spe,
            "DOR": dor,
            "LR+": lr_p,
            "LR-": lr_m
        })

        all_inner_logs.extend(inner_log)

    return {
        "inner_log": pd.DataFrame(all_inner_logs),
        "best_model": pd.DataFrame(all_best_models),
        "outer_test": pd.DataFrame(all_outer_tests)
    }


# In[12]:


if __name__ == "__main__":

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

        print(">>> Logistic Regression Nested CV")

        # --------------------------
        # Nested CV
        # --------------------------
        res = run_logreg_nested_cv_cpu(
            X=X,
            y=y
            # sampling_methods, sampling_ratios
        )

        # return: {"inner_log": df, "best_model": df, "outer_test": df}
        df_inner = res["inner_log"]
        df_best = res["best_model"]
        df_outer = res["outer_test"]

        # --------------------------
        # save result
        # --------------------------
        SAVE_DIR = f"/save_path/{disease}"
        os.makedirs(SAVE_DIR, exist_ok=True)

        df_inner.to_csv(
            f"{SAVE_DIR}/{disease}_logreg_nested_cv_all_results.csv",
            index=False
        )
        df_best.to_csv(
            f"{SAVE_DIR}/{disease}_logreg_nested_cv_best_per_fold.csv",
            index=False
        )
        df_outer.to_csv(
            f"{SAVE_DIR}/{disease}_logreg_nested_cv_outer_test_results.csv",
            index=False
        )

        print(">>> complete")
        print(f" - {SAVE_DIR}/{disease}_logreg_nested_cv_all_results.csv")
        print(f" - {SAVE_DIR}/{disease}_logreg_nested_cv_best_per_fold.csv")
        print(f" - {SAVE_DIR}/{disease}_logreg_nested_cv_outer_test_results.csv")

