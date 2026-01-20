# models/xgb.py
from xgboost import XGBClassifier
from .base import BaseModel


class XGBModel(BaseModel):

    def build(self, params, class_weight=None):

        model_params = {
            "colsample_bytree": params["colsample_bytree"],
            "learning_rate": params["learning_rate"],
            "max_depth": int(params["max_depth"]),
            "n_estimators": int(params["n_estimators"]),
            "subsample": params["subsample"],
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }

        # scale_pos_weight는 SHAP-safe
        if class_weight:
            model_params["scale_pos_weight"] = float(class_weight)

        return XGBClassifier(**model_params)
