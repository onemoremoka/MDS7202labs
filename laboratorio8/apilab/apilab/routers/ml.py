import logging

from fastapi import APIRouter, Depends, HTTPException

from apilab.ml_models.model import MLModelService
from apilab.models.ml import MLModel, MLModelIn

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/potabilidad", response_model=MLModel, status_code=200)
async def predict_potability(
    ml_model_input: MLModelIn,
):
    model_ = MLModelService("apilab/best_model.pkl")
    prediction = await model_.predict(ml_model_input)
    return MLModel(
        prediction=prediction.prediction,
        id=prediction.id,
        message="Prediction successful",
    )
    