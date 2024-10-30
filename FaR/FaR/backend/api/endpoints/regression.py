import os

from fastapi import APIRouter, HTTPException
from FaR.backend.schemas.request import PredictionInput
from FaR.backend.models import regression_model

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "Server is running"}

@router.get("/current-directory")
async def get_current_directory():
    return {"directory": os.getcwd()}

@router.get("/csv-head")
async def get_csv_head():
    data = regression_model.load_data()
    # Convert DataFrame to a JSON-serializable format
    return {"directory": data.head().to_dict(orient="records")}

@router.post("/train/")
async def train():
    try:
        # Load and preprocess data
        data = regression_model.load_data()
        X, y = regression_model.preprocess_data(data)
        X_train, X_test, y_train, y_test = regression_model.split_data(X, y)

        # Train linear and polynomial models
        linear_model = regression_model.train_linear_model(X_train, y_train)
        poly_model = regression_model.train_polynomial_model(X_train, y_train, degree=2)

        # Save models to disk
        regression_model.save_model(linear_model, path='FaR/model/linear_model.pkl')
        regression_model.save_model(poly_model, path='FaR/model/poly_model.pkl')

        return {"message": "Models trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@router.post("/predict/")
async def predict(input_data: PredictionInput):
    try:
        # Load models
        linear_model = regression_model.load_model(path='FaR/model/linear_model.pkl')
        poly_model = regression_model.load_model(path='FaR/model/poly_model.pkl')

        if not linear_model or not poly_model:
            raise HTTPException(status_code=400, detail="Models not trained. Please train the models first.")

        # Make predictions with both models
        linear_prediction = regression_model.make_prediction(linear_model, input_data.feature_values)
        poly_prediction = regression_model.make_prediction(poly_model, input_data.feature_values)

        return {
            "linear_prediction": linear_prediction,
            "polynomial_prediction": poly_prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
