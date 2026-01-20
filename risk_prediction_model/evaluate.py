# evaluate.py

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from scipy.stats import norm
from sklearn.utils import resample

def bootstrap_sens_spec_ci(
    y_true,
    y_pred,
    n_bootstrap=2000,
    alpha=0.95,
    random_state=0
):
    rng = np.random.RandomState(random_state)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    sens_list = []
    spec_list = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)

        yt = y_true[idx]
        yp = y_pred[idx]

        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        sens_list.append(sens)
        spec_list.append(spec)

    lower = (1 - alpha) / 2 * 100
    upper = (1 + alpha) / 2 * 100

    return {
        "sensitivity_ci_lower": np.nanpercentile(sens_list, lower),
        "sensitivity_ci_upper": np.nanpercentile(sens_list, upper),
        "specificity_ci_lower": np.nanpercentile(spec_list, lower),
        "specificity_ci_upper": np.nanpercentile(spec_list, upper),
    }

    
def delong_auc_ci(y_true, y_scores, alpha=0.95):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]

    m, n = len(pos_scores), len(neg_scores)

    v10 = np.array([
        np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)
        for ps in pos_scores
    ])
    v01 = np.array([
        np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)
        for ns in neg_scores
    ])

    auc = v10.mean()
    var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
    se = np.sqrt(var)

    z = norm.ppf(1 - (1 - alpha) / 2)
    ci_lower = auc - z * se
    ci_upper = auc + z * se

    return auc, ci_lower, ci_upper, se

def evaluate_binary(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    DOR = (tp * tn) / (fp * fn) if fp > 0 and fn > 0 else np.nan
    LR_pos = sensitivity / (1 - specificity) if specificity < 1 else np.nan
    LR_neg = (1 - sensitivity) / specificity if specificity > 0 else np.nan

    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # DeLong AUC CI
    auc_d, auc_lo, auc_hi, auc_se = delong_auc_ci(y_true, y_prob)

    # ⭐ Sens / Spec bootstrap CI
    sens_spec_ci = bootstrap_sens_spec_ci(
        y_true,
        y_pred,
        n_bootstrap=2000,
        alpha=0.95
    )

    return {
        "AUC": auc,
        "AUC_delong": auc_d,
        "AUC_delong_ci_lower": auc_lo,
        "AUC_delong_ci_upper": auc_hi,
        "AUC_delong_se": auc_se,
        "AUPRC": auprc,
        "Brier": brier,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "sensitivity_ci_lower": sens_spec_ci["sensitivity_ci_lower"],
        "sensitivity_ci_upper": sens_spec_ci["sensitivity_ci_upper"],
        "specificity_ci_lower": sens_spec_ci["specificity_ci_lower"],
        "specificity_ci_upper": sens_spec_ci["specificity_ci_upper"],
        "DOR": DOR,
        "LR+": LR_pos,
        "LR-": LR_neg
    }

# def evaluate_binary(y_true, y_pred, y_prob):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

#     DOR = (tp * tn) / (fp * fn) if fp > 0 and fn > 0 else np.nan
#     LR_pos = sensitivity / (1 - specificity) if specificity < 1 else np.nan
#     LR_neg = (1 - sensitivity) / specificity if specificity > 0 else np.nan

#     return {
#         "AUC": roc_auc_score(y_true, y_prob),
#         "AUPRC": average_precision_score(y_true, y_prob),
#         "Brier": brier_score_loss(y_true, y_prob),
#         "sensitivity": sensitivity,
#         "specificity": specificity,
#         "DOR": DOR,
#         "LR+": LR_pos,
#         "LR-": LR_neg
#     }

