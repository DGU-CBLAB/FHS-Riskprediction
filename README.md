# Risk prediction for cardiovascular-related diseases using PRS and EHR in the Framingham Heart Study

This repository contains the analysis code used in the study:

**Risk prediction for cardiovascular-related diseases using polygenic risk scores (PRS) and electronic health records (EHR) in the Framingham Heart Study (FHS).**

The code implements model development, hyperparameter tuning, nested cross-validation, and evaluation for cardiovascular risk prediction using PRS and EHR-derived features.

---

## 📁 Repository Structure

├── grid_model_code/
│ ├── rf/
│ ├── xgboost/
│ ├── catboost/
│ ├── lightgbm/
│ └── logistic_regression/
│
├── nestedCV/
│ ├── rf/
│ ├── xgboost/
│ ├── catboost/
│ ├── lightgbm/
│ └── logistic_regression/
│
├── risk_prediction_model/
│ ├── train.py
│ ├── evaluate.py
│ ├── utils.py
│ └── model/
│ ├── rf.py
│ ├── xgboost.py
│ ├── catboost.py
│ ├── lightgbm.py
│ └── logistic_regression.py
│
└── README.md


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

The individual-level genotype and phenotype data used in this study are **not publicly available** due to controlled-access restrictions.

Framingham Heart Study (FHS) data were obtained through the NIH database of Genotypes and Phenotypes (**dbGaP**).  
Access to these data requires prior dbGaP approval and compliance with all relevant data use agreements.

This repository does not include individual-level genotype or EHR data.  
Reproduction of the analyses requires independent authorization from dbGaP.

---

## 🔁 Reproducibility

All analyses are **reproducible in principle**, provided that authorized access to the Framingham Heart Study data is obtained through dbGaP.

Users must supply their own approved datasets and adapt file paths and data-loading scripts as needed.

---

## ⚙️ Software Requirements

- Python (≥ 3.x)
- scikit-learn  
- XGBoost  
- CatBoost  
- LightGBM  
- NumPy, pandas

---

## 📜 Ethics Statement

This study is a secondary analysis of data from the Framingham Heart Study.  
All participants provided informed consent, and data access was approved through dbGaP.

---

## 📌 Citation

If you use this code, please cite:

> *Risk prediction for cardiovascular-related diseases using PRS and EHR in the Framingham Heart Study*. (Under review)

---

## 📬 Contact

For questions regarding this repository, please contact the corresponding author.
