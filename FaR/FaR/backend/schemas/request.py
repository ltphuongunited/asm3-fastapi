
from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    feature_values: List[float]
