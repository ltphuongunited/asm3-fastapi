from pydantic import BaseModel

class RegressionInput(BaseModel):
    feature: float

class RegressionOutput(BaseModel):
    prediction: float
