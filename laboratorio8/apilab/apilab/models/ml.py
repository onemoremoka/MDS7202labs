from pydantic import BaseModel, ConfigDict


class MLModelIn(BaseModel):
    ph: float
    hardness: float
    solids: float
    chloramines: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float


class MLModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    prediction: int
    id: int
