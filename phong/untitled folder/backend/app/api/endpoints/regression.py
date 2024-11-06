from fastapi import APIRouter, HTTPException, Request
from typing import List
import os
import pandas as pd

from app.api.schemas.request import ValidationInput, PredictionOutput
from app.api.models.regression_model import (
    validate_data,
    predict_burden,
    read_countries_from_csv,
)

router = APIRouter()

# Path to the CSV data file
CSV_FILE_PATH = "./data/air_quality_health.csv"

# show current working directory

@router.get("/pollutants")
def get_pollutants():
    if not os.path.exists(CSV_FILE_PATH):
        raise HTTPException(status_code=404, detail="Data file not found.")
    
    try:
        data = pd.read_csv(CSV_FILE_PATH)
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=500, detail=f"CSV Parsing Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data file: {str(e)}")
    
    pollutants = data['Pollutant'].unique().tolist()
    pollutant_mapping = {
        "no2": "Nitrogen Oxide (NO2)",
        "pm25": "Particulate Matter 2.5 (PM)",
        "ozone": "Ozone (O3)",
        "hap": "Hazardous Air Pollutants (HAP)"
    }
    pollutants_with_display = [{"key": key, "display": pollutant_mapping.get(key, key)} for key in pollutants]
    return {"pollutants": pollutants_with_display}


@router.post("/predict", response_model=PredictionOutput)
async def predict(input_data: ValidationInput):
    try:
        # Validate input data
        validate_data(input_data)

        # Generate prediction
        result = predict_burden(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
