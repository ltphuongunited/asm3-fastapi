from fastapi import APIRouter
from app.models.regression_model import predict
from app.schemas.request import RegressionInput, RegressionOutput

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "Server is running"}

@router.post("/predict", response_model=RegressionOutput)
async def get_prediction(input_data: RegressionInput):
    result = predict(input_data)
    return {"prediction": result}
