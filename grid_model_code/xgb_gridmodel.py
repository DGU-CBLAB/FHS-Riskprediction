#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import make_scorer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier

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


def calc_metrics(y_true, y_pred, y_proba):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    auc = roc_auc_score(y_true, y_proba)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    lr_plus = sensitivity / (1 - specificity + 1e-8)
    lr_minus = (1 - sensitivity) / (specificity + 1e-8)
    dor = lr_plus / (lr_minus + 1e-8)

    return auc, sensitivity, specificity, dor, lr_plus, lr_minus

def apply_sampling(X, y, method, ratio, random_state=42):

    if method == "undersample":
        sampler = RandomUnderSampler(
            sampling_strategy=ratio,
            random_state=random_state
        )
        return sampler.fit_resample(X, y)

    elif method == "smote":
        sampler = SMOTE(
            sampling_strategy=ratio,
            random_state=random_state
        )
        return sampler.fit_resample(X, y)

    else:
        return X.copy(), y.copy()

def xgb_auc_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)


# In[3]:


def run_xgb_final_pipeline(
    X,
    y,
    disease,
    save_dir,
    sampling_methods=["none", "undersample", "smote", "class_weight"],
    sampling_ratios=[0.6, 0.8, 1.0],
):


    os.makedirs(save_dir, exist_ok=True)

    # ======================================================
    # 1. Train / Validation split
    # ======================================================
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ======================================================
    # 2. Scaling
    # ======================================================
    cols_to_scale = [c for c in X.columns if c not in ["AGE", "SEX", "DLVH"]]

    scaler = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), cols_to_scale)
        ],
        remainder="passthrough"
    )

    # ======================================================
    # 3. XGBoost base model
    # ======================================================
    xgb = XGBClassifier(
        tree_method="hist",
        device="cpu",
        eval_metric="logloss",
        random_state=42
    )

    # ======================================================
    # 4. Hyperparameter grid
    # ======================================================
    param_grid = {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [3, 5, 10],
        "model__learning_rate": [0.1, 0.3],
        "model__subsample": [0.6, 1.0],
        "model__colsample_bytree": [0.7, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_logs = []
    final_logs = []

    # ======================================================
    # 5. Sampling loop + GridSearch
    # ======================================================
    for sm in sampling_methods:

        ratios = sampling_ratios if sm in ["undersample", "smote"] else [None]

        for ratio in ratios:

            print(f"\n>>> [{disease}] Sampling={sm}, Ratio={ratio}")

            # --------------------------
            # Sampling (Train only)
            # --------------------------
            X_res, y_res = apply_sampling(X_train, y_train, sm, ratio)

            # --------------------------
            # Pipeline
            # --------------------------
            pipe = Pipeline([
                ("scaler", scaler),
                ("model", xgb)
            ])

            if sm == "class_weight":
                scale_pos_weight = (len(y_res) - sum(y_res)) / sum(y_res)
                pipe.set_params(model__scale_pos_weight=scale_pos_weight)

            # --------------------------
            # GridSearchCV
            # --------------------------
            for i, (tr, te) in enumerate(cv.split(X_res, y_res)):
                uniq, cnt = np.unique(y_res.iloc[te], return_counts=True)
                print(i, dict(zip(uniq, cnt)))
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=xgb_auc_scorer,
                cv=cv,
                n_jobs=-1,
                verbose=1,
                return_train_score=True,
                error_score="raise"   # 🔥 이거
            )
            grid.fit(X_res, y_res)

            # --------------------------
            # Grid 
            # --------------------------
            df_grid = pd.DataFrame(grid.cv_results_)
            df_grid["sampling_method"] = sm
            df_grid["sampling_ratio"] = ratio
            grid_logs.append(df_grid)

            # ==================================================
            # 6. best model → Validation
            # ==================================================
            best_model = grid.best_estimator_

            y_val_proba = best_model.predict_proba(X_val)[:, 1]
            y_val_pred = (y_val_proba > 0.5).astype(int)

            auc, sen, spe, dor, lr_p, lr_m = calc_metrics(
                y_val, y_val_pred, y_val_proba
            )

            final_logs.append({
                "disease": disease,
                "sampling_method": sm,
                "sampling_ratio": ratio,
                "AUC": auc,
                "sensitivity": sen,
                "specificity": spe,
                "DOR": dor,
                "LR+": lr_p,
                "LR-": lr_m,
                **grid.best_params_
            })

    # ======================================================
    # 7. save result
    # ======================================================
    df_grid_all = pd.concat(grid_logs, ignore_index=True)
    df_final = pd.DataFrame(final_logs)

    best_row = df_final.loc[df_final["AUC"].idxmax()]

    df_grid_all.to_csv(
        f"{save_dir}/{disease}_xgb_gridsearch_results.csv",
        index=False
    )
    df_final.to_csv(
        f"{save_dir}/{disease}_xgb_validation_results.csv",
        index=False
    )
    pd.DataFrame([best_row]).to_csv(
        f"{save_dir}/{disease}_xgb_best_model.csv",
        index=False
    )

    print(f"\n>>> [{disease}] result complete")

    return {
        "grid_results": df_grid_all,
        "validation_results": df_final,
        "best_model": best_row
    }


# In[6]:


if __name__ == "__main__":

    disease_list = ["dia", "chf", "chd", "stroke", "af", "dem"]

    for disease in disease_list:

        print("\n===================================")
        print(f"### GridSearch: {disease.upper()} ###")
        print("===================================")

        df = pd.read_csv(FILES[disease])
        features = FEATURES[disease]
        target = TARGET_NAME[disease]

        df_sub = df[features + [target]].dropna()
        X = df_sub[features]
        y = df_sub[target]
        
        SAVE_DIR = f"/save_path/{disease}"
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        res = run_xgb_final_pipeline(X, y, disease, SAVE_DIR)


# In[ ]:




