# models/cat.py
from catboost import CatBoostClassifier
from .base import BaseModel


class CatModel(BaseModel):

    def build(self, params, class_weight=None):

        model_params = {
            "iterations": int(params["n_estimators"]),
            "depth": int(params["max_depth"]),
            "learning_rate": params["learning_rate"],
            "l2_leaf_reg": params["l2_leaf_reg"],
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": 0
        }

        if class_weight == "balanced":
            model_params["auto_class_weights"] = "Balanced"

        return CatBoostClassifier(**model_params)

