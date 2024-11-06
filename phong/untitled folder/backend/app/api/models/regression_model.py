import os
import pandas as pd
import numpy as np
from fastapi import HTTPException
from typing import List, Tuple, Optional

from app.api.schemas.request import ValidationInput, PredictionOutput
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def read_countries_from_csv(iso3_code: str, csv_file_path: str) -> str:
    """
    Retrieve the country name based on the ISO3 code from the CSV file.

    Args:
        iso3_code (str): The 3-letter ISO country code.
        csv_file_path (str): Path to the CSV data file.

    Returns:
        str: The country name if found, else "Unknown Country".
    """
    data = pd.read_csv(csv_file_path)
    country_row = data[data['ISO3'] == iso3_code]
    if not country_row.empty:
        return country_row['Country'].iloc[0]
    return "Unknown Country"


def validate_data(input_data: ValidationInput):
    """
    Validate the input data against the CSV dataset.

    Args:
        input_data (ValidationInput): The input data containing iso3, exposure_mean, and pollutant.

    Raises:
        HTTPException: If validation fails due to missing data or columns.
    """
    errors: List[str] = []

    iso3 = input_data.iso3
    exposure_mean = input_data.exposure_mean
    pollutant = input_data.pollutant

    # Check if the CSV file exists
    csv_file_path = "./data/air_quality_health_2.csv"
    if not os.path.exists(csv_file_path):
        raise HTTPException(status_code=404, detail=["Data file not found at", csv_file_path])

    try:
        data = pd.read_csv(csv_file_path)
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=500, detail=[f"CSV Parsing Error: {str(e)}"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=[f"Error reading data file: {str(e)}"])

    # Normalize column names to lowercase for consistency
    data.columns = [col.strip().lower() for col in data.columns]

    # Ensure that 'iso3' and 'pollutant' columns exist
    required_columns = ['iso3', 'pollutant']
    for col in required_columns:
        if col not in data.columns:
            errors.append(f"Missing required column in CSV: {col.upper()}.")

    if errors:
        raise HTTPException(status_code=500, detail=errors)

    # Filter data based on input ISO3 code and pollutant
    iso3_data = data[data['iso3'] == iso3]
    if iso3_data.empty:
        errors.append(f"No data found for the specified ISO3 country code: {iso3}.")

    pollutant_data = iso3_data[iso3_data['pollutant'] == pollutant]
    if pollutant_data.empty:
        country_name = read_countries_from_csv(iso3, csv_file_path)
        errors.append(f"No data found for the specified pollutant: {pollutant} in country: {country_name}.")

    if errors:
        raise HTTPException(status_code=400, detail=errors)

    return {"message": "Validation successful", "status_code": 200}


def load_and_filter_data(input_data: ValidationInput, csv_file_path: str) -> pd.DataFrame:
    """
    Load the dataset and filter it based on ISO3 code and pollutant.

    Args:
        input_data (ValidationInput): The input data containing iso3 and pollutant.
        csv_file_path (str): Path to the CSV data file.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        HTTPException: If no data is found after filtering.
    """
    dataset = pd.read_csv(csv_file_path)

    # Filter dataset based on country and pollutant
    filtered_df = dataset[
        (dataset['ISO3'] == input_data.iso3) &
        (dataset['Pollutant'] == input_data.pollutant)
    ]

    if filtered_df.empty:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "NoDataFound",
                "message": f"No data available for the specified criteria: ISO3={input_data.iso3}, Pollutant={input_data.pollutant}."
            }
        )

    return filtered_df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the DataFrame using the IQR method for 'Exposure Mean' and 'Burden Mean'.

    Args:
        df (pd.DataFrame): The filtered DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame without outliers.

    Raises:
        HTTPException: If no data remains after outlier removal.
    """
    # Outlier removal for 'Exposure Mean'
    Q1_exposure = df['Exposure Mean'].quantile(0.25)
    Q3_exposure = df['Exposure Mean'].quantile(0.75)
    IQR_exposure = Q3_exposure - Q1_exposure
    lower_bound_exposure = Q1_exposure - 1.5 * IQR_exposure
    upper_bound_exposure = Q3_exposure + 1.5 * IQR_exposure

    # Outlier removal for 'Burden Mean'
    Q1_burden = df['Burden Mean'].quantile(0.25)
    Q3_burden = df['Burden Mean'].quantile(0.75)
    IQR_burden = Q3_burden - Q1_burden
    lower_bound_burden = Q1_burden - 1.5 * IQR_burden
    upper_bound_burden = Q3_burden + 1.5 * IQR_burden

    # Clean dataset by removing outliers
    dataset_cleaned = df[
        (df['Exposure Mean'] >= lower_bound_exposure) &
        (df['Exposure Mean'] <= upper_bound_exposure) &
        (df['Burden Mean'] >= lower_bound_burden) &
        (df['Burden Mean'] <= upper_bound_burden)
    ]

    if dataset_cleaned.empty:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "NoDataFound",
                "message": "No data available after outlier removal."
            }
        )

    return dataset_cleaned


def scale_features_targets(X: pd.DataFrame, y: pd.Series) -> Tuple[MinMaxScaler, MinMaxScaler, pd.DataFrame, pd.Series]:
    """
    Scale the feature and target variables using Min-Max Scaler.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.

    Returns:
        Tuple containing scaler_X, scaler_y, X_scaled, y_scaled.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X.values)  # Use NumPy array)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    return scaler_X, scaler_y, X_scaled, y_scaled


def split_data(X_scaled, y_scaled, test_size=0.2, random_state=42) -> Tuple:
    """
    Split the scaled data into training and testing sets.

    Args:
        X_scaled (np.ndarray): Scaled feature array.
        y_scaled (np.ndarray): Scaled target array.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.

    Returns:
        Tuple: Split data (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """
    Train a Linear Regression model.

    Args:
        X_train (np.ndarray): Training feature array.
        y_train (np.ndarray): Training target array.

    Returns:
        LinearRegression: Trained Linear Regression model.
    """
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)
    return model_linear


def train_polynomial_regression(X_train, y_train, degree=4) -> Tuple[LinearRegression, PolynomialFeatures]:
    """
    Train a Polynomial Regression model of specified degree.

    Args:
        X_train (np.ndarray): Training feature array.
        y_train (np.ndarray): Training target array.
        degree (int, optional): Degree of the polynomial features. Defaults to 4.

    Returns:
        Tuple containing the trained Polynomial Regression model and the PolynomialFeatures instance.
    """
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)
    return model_poly, poly


def evaluate_model(model, X_test, y_test) -> Tuple[float, float]:
    """
    Evaluate the model using Mean Squared Error and R^2 Score.

    Args:
        model: Trained regression model.
        X_test (np.ndarray): Testing feature array.
        y_test (np.ndarray): Testing target array.

    Returns:
        Tuple containing MSE and R^2 Score.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def make_prediction(model: LinearRegression, scaler_X: MinMaxScaler, scaler_y: MinMaxScaler, poly: Optional[PolynomialFeatures], exposure_mean: float) -> float:
    """
    Make a prediction for the given exposure_mean using the specified model.

    Args:
        model (LinearRegression): Trained regression model.
        scaler_X (MinMaxScaler): Fitted scaler for features.
        scaler_y (MinMaxScaler): Fitted scaler for targets.
        poly (Optional[PolynomialFeatures]): PolynomialFeatures instance if using Polynomial Regression.
        exposure_mean (float): The exposure mean value for prediction.

    Returns:
        float: The predicted burden mean, rounded to two decimal places.
    """
    input_array = np.array([[exposure_mean]])
    X_scaled = scaler_X.transform(input_array)
    if poly:
        exposure_transformed = poly.transform(X_scaled)
        prediction_scaled = model.predict(exposure_transformed)
    else:
        prediction_scaled = model.predict(X_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    return round(prediction[0][0], 2)


def select_best_model(metrics_linear: Tuple[float, float], metrics_poly: Tuple[float, float]) -> str:
    """
    Select the best model based on MSE and R^2 Score.

    Args:
        metrics_linear (Tuple[float, float]): MSE and R^2 for Linear Regression.
        metrics_poly (Tuple[float, float]): MSE and R^2 for Polynomial Regression.

    Returns:
        str: 'linear' or 'poly' indicating the better model.
    """
    mse_linear, r2_linear = metrics_linear
    mse_poly, r2_poly = metrics_poly

    if mse_linear < mse_poly and r2_linear > r2_poly:
        return 'linear'
    elif mse_poly < mse_linear and r2_poly > r2_linear:
        return 'poly'
    else:
        return 'linear'  # Default to linear if no clear winner


def predict_burden(input_data: ValidationInput) -> PredictionOutput:
    """
    Predict the burden mean based on the input data.

    Args:
        input_data (ValidationInput): The input data containing iso3, exposure_mean, and pollutant.

    Returns:
        PredictionOutput: The predicted burden mean.
    """
    csv_file_path = "./data/air_quality_health_2.csv"

    # Load and filter data
    filtered_df = load_and_filter_data(input_data, csv_file_path)

    # Remove outliers
    dataset_cleaned = remove_outliers(filtered_df)

    # Feature and target variables
    X = dataset_cleaned[['Exposure Mean']]
    y = dataset_cleaned['Burden Mean']

    # Scale features and target
    scaler_X, scaler_y, X_scaled, y_scaled = scale_features_targets(X, y)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)

    # Train models
    model_linear = train_linear_regression(X_train, y_train)
    model_poly, poly = train_polynomial_regression(X_train, y_train, degree=4)

    # Evaluate models
    mse_linear, r2_linear = evaluate_model(model_linear, X_test, y_test)
    mse_poly, r2_poly = evaluate_model(model_poly, poly.transform(X_test), y_test)

    # Select best model
    best_model = select_best_model((mse_linear, r2_linear), (mse_poly, r2_poly))

    # Make prediction
    exposure_mean = input_data.exposure_mean
    if best_model == 'linear':
        prediction = make_prediction(model_linear, scaler_X, scaler_y, poly=None, exposure_mean=exposure_mean)
    else:
        prediction = make_prediction(model_poly, scaler_X, scaler_y, poly=poly, exposure_mean=exposure_mean)

    return PredictionOutput(predicted_burden_mean=float(prediction))
