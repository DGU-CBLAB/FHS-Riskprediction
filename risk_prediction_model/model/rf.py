from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel


class RFModel(BaseModel):

    def build(self, params, class_weight=None):

        model_params = {
            "n_estimators": int(params["n_estimators"]),
            "random_state": 42,
            "n_jobs": -1,
            "bootstrap": True
        }

        # max_depth
        if int(params["max_depth"]) > 0:
            model_params["max_depth"] = int(params["max_depth"])

        # split / leaf
        model_params["min_samples_split"] = int(params["min_samples_split"])
        model_params["min_samples_leaf"] = int(params["min_samples_leaf"])

        # criterion (gini / entropy / log_loss)
        model_params["criterion"] = params["criterion"]

        # max_features (SHAP 안정성 ↑)
        mf = params["max_features"]
        if mf in ["sqrt", "log2"]:
            model_params["max_features"] = mf
        else:
            model_params["max_features"] = float(mf)

        # class weight (balanced or dict)
        if class_weight is not None:
            model_params["class_weight"] = class_weight

        return RandomForestClassifier(**model_params)