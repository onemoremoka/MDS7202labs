import logging

from fastapi import APIRouter, Depends, HTTPException

from apilab.ml_models.model import MLModelService
from apilab.models.ml import MLModel, MLModelIn

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/potabilidad", response_model=MLModel, status_code=201)
async def predict_potability(
    ml_model: MLModelIn,
    ml_service: MLModelService = Depends(MLModelService.get_instance),
):
    """
    Predict the potability of water based on input features.
    """
    try:
        prediction = await ml_service.predict(ml_model)
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
