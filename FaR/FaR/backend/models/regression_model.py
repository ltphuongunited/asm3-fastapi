# backend/api/models/regression_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os

# Paths for data and model storage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_FILE_PATH = os.path.join(BASE_DIR, 'data', 'air_quality_health_2.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'regression_model.pkl')

def load_data():
    # Load dataset from CSV
    return pd.read_csv(CSV_FILE_PATH)


def preprocess_data(data):
    # Drop non-numeric columns that are irrelevant for training
    data = data.select_dtypes(include=[float, int])

    # Separate features and target
    X = data.iloc[:, :-1]  # Assuming all but the last column are features
    y = data.iloc[:, -1]  # Assuming the last column is the target variable

    # Normalize numeric features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    return X_normalized, y

def split_data(X, y):
    # Split into training and test sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_model(X_train, y_train):
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_polynomial_model(X_train, y_train, degree=2):
    # Train a polynomial regression model
    poly_features = PolynomialFeatures(degree=degree)
    model = make_pipeline(poly_features, LinearRegression())
    model.fit(X_train, y_train)
    return model

def save_model(model, path=MODEL_PATH):
    # Save model to disk
    dump(model, path)

def load_model(path=MODEL_PATH):
    # Load model from disk
    return load(path) if os.path.exists(path) else None

def make_prediction(model, features):
    # Make predictions
    return model.predict([features])[0]
