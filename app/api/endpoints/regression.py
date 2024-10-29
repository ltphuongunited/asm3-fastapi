from fastapi import APIRouter
from app.models.regression_model import predict
from app.schemas.request import RegressionInput, RegressionOutput
from datetime import datetime
import pandas as pd
router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "Server is running"}

@router.post("/predict", response_model=RegressionOutput)
async def get_prediction(input_data: RegressionInput):
    result = predict(input_data)

    prediction_data = {
        "timestamp": [datetime.now()],
        "feature": [input_data.feature],
        "prediction": [result]
    }
    df = pd.DataFrame(prediction_data)

    df.to_csv("data/predictions.csv", mode='a', header=not pd.io.common.file_exists("data/predictions.csv"), index=False)

    return {"prediction": result}
