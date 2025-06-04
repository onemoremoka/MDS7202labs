import os
import pickle
from typing import Any

from apilab.models.ml import MLModel, MLModelIn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")


class MLModelService:
    _instance = None

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model()

    @classmethod
    def get_instance(cls) -> "MLModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> Any:
        print(f"Loading model from {self.model_path}: {os.getcwd()}")
        with open(os.path.join(os.getcwd(), self.model_path), "rb") as f:
            model = pickle.load(f)
        return model

    async def predict(self, ml_model_in: MLModelIn) -> MLModel:
        features = [
            [
                ml_model_in.ph,
                ml_model_in.hardness,
                ml_model_in.solids,
                ml_model_in.chloramines,
                ml_model_in.sulfate,
                ml_model_in.conductivity,
                ml_model_in.organic_carbon,
                ml_model_in.trihalomethanes,
                ml_model_in.turbidity,
            ]
        ]
        prediction = int(self.model.predict(features)[0])
        return MLModel(prediction=prediction, id=1)
