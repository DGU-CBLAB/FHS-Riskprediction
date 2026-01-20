# models/lgb.py
from lightgbm import LGBMClassifier
from .base import BaseModel


class LGBModel(BaseModel):

    def build(self, params, class_weight=None):
        model_params = {
            "n_estimators": int(params["n_estimators"]),
            "learning_rate": params["learning_rate"],
            "max_depth": int(params["max_depth"]),
            "num_leaves": int(params["num_leaves"]),
            "colsample_bytree": params["colsample_bytree"],
            "subsample": params["subsample"],
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,      # ⭐ 핵심
            "verbose": -1         # ⭐ 핵심
        }

        if class_weight == "balanced":
            model_params["class_weight"] = "balanced"

        return LGBMClassifier(**model_params)

