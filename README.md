# FastAPI Regression Model Server

This project is a **FastAPI** server designed to perform predictions using a regression model with one input feature and one output. The server includes two endpoints: **GET** to check the server status and **POST** to make predictions.

## Project Structure

```plaintext
project_root/
├── app/
│   ├── main.py                # Entry point for FastAPI
│   ├── api/
│   │   └── endpoints/
│   │       └── regression.py   # Routes for GET status and POST predict
│   ├── models/
│   │   └── regression_model.py # Model loading and prediction logic
│   └── schemas/
│       └── request.py          # Request and response schemas
├── model/
│   └── regression_model.pkl    # Serialized regression model
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

## Features

1. **GET /status** - Checks if the server is running.
2. **POST /predict** - Accepts a JSON input with a single feature to predict an output.

## Getting Started

### Prerequisites


### Installation
   ```bash
   pip install -r requirements.txt
   ```

### Run the Server

To start the FastAPI server, use:

```bash
uvicorn app.main:app --reload
```

The server should now be running on `http://127.0.0.1:8000`.

## API Endpoints

### 1. GET `/status`

- **Description**: Checks if the server is running.
- **Response**:
  - **Status Code**: `200 OK`
  - **Response Body**:
    ```json
    {
      "status": "Server is running"
    }
    ```

### 2. POST `/predict`

- **Description**: Predicts an output based on a single input feature using the regression model.
- **Request Body**:
  - **Format**: JSON
  - **Example**:
    ```json
    {
      "feature": 1.5
    }
    ```
- **Response**:
  - **Status Code**: `200 OK`
  - **Response Body**:
    ```json
    {
      "prediction": <predicted_value>
    }
    ```

## Example Requests

1. **Check server status**:

   ```bash
   curl -X GET "http://127.0.0.1:8000/status"
   ```

2. **Make a prediction**:

   ```bash
   curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"feature": 1.5}'
   ```
