# models/log.py
from sklearn.linear_model import LogisticRegression
from .base import BaseModel


class LogisticModel(BaseModel):

    def build(self, params, class_weight=None):

        return LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            solver=params["solver"],
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        )

