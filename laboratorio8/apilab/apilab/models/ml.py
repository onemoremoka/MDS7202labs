from pydantic import BaseModel, ConfigDict
from typing import Optional

from pydantic import BaseModel, Field

class MLModelIn(BaseModel):
    ph: float
    hardness: float = Field(..., alias="Hardness")
    solids: float = Field(..., alias="Solids")
    chloramines: float = Field(..., alias="Chloramines")
    sulfate: float = Field(..., alias="Sulfate")
    conductivity: float = Field(..., alias="Conductivity")
    organic_carbon: float = Field(..., alias="Organic_carbon")
    trihalomethanes: float = Field(..., alias="Trihalomethanes")
    turbidity: float = Field(..., alias="Turbidity")



class MLModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    prediction: int
    id: int
    message: Optional[str] = None