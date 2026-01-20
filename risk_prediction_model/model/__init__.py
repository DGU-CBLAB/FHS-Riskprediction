from .xgb import XGBModel
from .lgb import LGBModel
from .cat import CatModel
from .rf import RFModel
from .log import LogisticModel

MODEL_FACTORY = {
    "xgboost": XGBModel(),
    "lightgbm": LGBModel(),
    "catboost": CatModel(),
    "randomforest": RFModel(),
    "logistic": LogisticModel()
}

