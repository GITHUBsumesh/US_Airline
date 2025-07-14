from pydantic import BaseModel
from typing import Any, Dict
from datetime import datetime

class PredictionCreate(BaseModel):
    predicted_value: int
    input_features: Dict[str, Any]

class PredictionResponse(PredictionCreate):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
