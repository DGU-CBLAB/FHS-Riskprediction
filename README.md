# Risk prediction for cardiovascular-related diseases using PRS and EHR in the Framingham Heart Study

This repository contains the analysis code used in the study:

**Risk prediction for cardiovascular-related diseases using polygenic risk scores (PRS) and electronic health records (EHR) in the Framingham Heart Study (FHS).**

The code implements model development, hyperparameter tuning, nested cross-validation, and evaluation for cardiovascular risk prediction using PRS and EHR-derived features.

---

## 📁 Repository Structure

```
├── grid_model_code/
│   ├── rf/
│   ├── xgboost/
│   ├── catboost/
│   ├── lightgbm/
│   └── logistic_regression/
│
├── nestedCV/
│   ├── rf/
│   ├── xgboost/
│   ├── catboost/
│   ├── lightgbm/
│   └── logistic_regression/
│
├── risk_prediction_model/
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── model/
│       ├── rf.py
│       ├── xgboost.py
│       ├── catboost.py
│       ├── lightgbm.py
│       └── logistic_regression.py
│
└── README.md
```

---

## 🧠 Models Implemented

- Logistic Regression  
- Random Forest  
- XGBoost  
- CatBoost  
- LightGBM  

Hyperparameter tuning was performed using grid search, and model performance was evaluated using a **nested cross-validation framework**.

---

## 🔒 Data Availability

The individual-level data used in this study were obtained from the Framingham Heart Study (FHS) through the NIH Database of Genotypes and Phenotypes (dbGaP; accession: phs000007.v32.p13, Framingham Cohort). These data are available to qualified researchers upon approval of data access requests through dbGaP (https://www.ncbi.nlm.nih.gov/projects/gap/).

GWAS summary statistics used for polygenic risk score (PRS) estimation were obtained from the GWAS Catalog and are publicly available. The following datasets were used: GCST006414 (Atrial fibrillation), GCST90473543 (Myocardial ischemia), GCST90480183 (Diastolic heart failure), GCST007320 (Alzheimer’s disease), GCST90267278 (Diabetes), and GCST90044350 (Stroke). These summary statistics can be downloaded directly from the GWAS Catalog (https://www.ebi.ac.uk/gwas/).


---

## 🔁 Reproducibility

All analyses are **reproducible in principle**, provided that authorized access to the Framingham Heart Study data is obtained through dbGaP.

Users must supply their own approved datasets and adapt file paths and data-loading scripts as needed.

---

## ⚙️ Software and Implementation

Machine learning model training and evaluation were performed in Python using commonly used libraries.

Key software and packages include:
- R (v3.6.0)
- Python (v3.10)
- scikit-learn (version 1.7.2)
- XGBoost (version 2.0.3)
- LightGBM (version 4.6.0)
- CatBoost (version 1.2.8)
- imbalanced-learn (version 0.14.0)

Random seeds were fixed within each cross-validation procedure to ensure reproducibility.  
SHAP was used for model interpretation, with explanations computed on held-out test data only.


## 📜 Ethics Statement

This study is a secondary analysis of data from the Framingham Heart Study.  
All participants provided informed consent, and data access was approved through dbGaP.

---

## 📌 Citation

If you use this code, please cite:

> *Risk prediction for cardiovascular-related diseases using PRS and EHR in the Framingham Heart Study*. (Under review)

---

## 📬 Contact

For questions regarding this repository, contact taegun89@gmail.com.
