# models/base.py
from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def build(self, params, class_weight=None):
        pass

