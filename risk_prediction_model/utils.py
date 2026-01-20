# utils.py

import pandas as pd
##! check BG diabets
#prs_ehr_model
FEATURES_prs_ehr = {
    "af": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","PRSice2"],
    "chd": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","LDpred"],
    "chf": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","PRSice2"],
    "dem": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","LDpred"],
    "dia": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","LDpred"],
    "stroke": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP","Lasso"]
}
#    "dia": ["AGE","SEX","TRIG","VENT_RT","HIP","CREAT","alcohol","BG","CALC_LDL","DBP","SBP","DLVH","HDL","TC","CURRSMK","BMI","HGT","LDpred"],

#ehr_model
FEATURES_ehr = {
    "af": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
    "chd": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
    "chf": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
    "dem": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
    "dia": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"],
    "stroke": ["AGE","SEX","VENT_RT","alcohol","TRIG","BG","CREAT","SBP","DBP","HGT","DLVH","CALC_LDL","HDL","CPD","HIP"]
}

#prs_model
FEATURES_prs = {
    "af": ["AGE","SEX","PRSice2"],
    "chd": ["AGE","SEX","LDpred"],
    "chf": ["AGE","SEX","PRSice2"],
    "dem": ["AGE","SEX","LDpred"],
    "dia": ["AGE","SEX","LDpred"],
    "stroke": ["AGE","SEX","Lasso"]
}

# FILES = {
#     "dia":    "/Data/taegun/prs_revision/data/df_diabet_phenotype_final.csv",
#     "chf":    "/Data/taegun/prs_revision/data/df_chf_phenotype_final.csv",
#     "chd":    "/Data/taegun/prs_revision/data/df_chd_phenotype_final.csv",
#     "stroke": "/Data/taegun/prs_revision/data/df_stroke_phenotype_final.csv",
#     "af":     "/Data/taegun/prs_revision/data/df_af_phenotype_final.csv",
#     "dem":    "/Data/taegun/prs_revision/data/df_dem_phenotype_final.csv"
# }
FILES = {
    "dia":    "/Data/taegun/prs_revision/data/df_diabet_match_pcr_final2.csv",
    "chf":    "/Data/taegun/prs_revision/data/df_chf_match_pcr_final2.csv",
    "chd":    "/Data/taegun/prs_revision/data/df_chd_match_pcr_final2.csv",
    "stroke": "/Data/taegun/prs_revision/data/df_stroke_match_pcr_final2.csv",
    "af":     "/Data/taegun/prs_revision/data/df_af_match_pcr_final2.csv",
    "dem":    "/Data/taegun/prs_revision/data/df_dem_match_pcr_final2.csv"
}

TARGET_NAME = {k: "Disease_status" for k in FILES.keys()}

FEATURE_SETS = {
    "prs_ehr": FEATURES_prs_ehr,
    "ehr": FEATURES_ehr,
    "prs": FEATURES_prs,
}

def load_data(disease, model_type="prs_ehr"):
    """
    model_type: 'prs_ehr', 'ehr', 'prs'
    """

    df = pd.read_csv(FILES[disease])
    if model_type not in FEATURE_SETS:
        raise ValueError(f"Unknown model_type: {model_type}")
    features = FEATURE_SETS[model_type][disease]
    target = TARGET_NAME[disease]
    df_sub = df[features + [target]].dropna()
    return df_sub[features], df_sub[target]

