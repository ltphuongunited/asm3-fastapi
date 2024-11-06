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
CSV_FILE_PATH = "./data/air_quality_health_2.csv"

# show current working directory

@router.get("/")
def read_root():
    if not os.path.exists(CSV_FILE_PATH):
        raise HTTPException(status_code=404, detail="Data file not found")

    try:
        data = pd.read_csv(CSV_FILE_PATH)
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=500, detail=f"CSV Parsing Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data file: {str(e)}")

    return data.head().to_dict(orient='records')


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
        "hap": "Hazardous Air Pollutants (HAP)"
    }
    pollutants_with_display = [{"key": key, "display": pollutant_mapping.get(key, key)} for key in pollutants]
    return {"pollutants": pollutants_with_display}


# @router.post("/validate", response_model=dict)
# async def validate(input_data: ValidationInput):
#     return validate_data(input_data)


@router.post("/predict", response_model=PredictionOutput)
async def predict(input_data: ValidationInput):
    validate_data(input_data)
    result = predict_burden(input_data)
    return result
