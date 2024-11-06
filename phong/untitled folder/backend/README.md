# Air Quality Health API

A FastAPI application for predicting air quality health burdens based on exposure data.

## Project Structure

```plaintext
project_root/
├── app/
│   ├── main.py                # Entry point for FastAPI
│   ├── api/
│   │   └── endpoints/
│   │       └── regression.py   # Routes for GET status, POST predict, and POST query
│   ├── models/
│   │   └── regression_model.py # Model loading and prediction logic
│   ├── schemas/
│   │   └── request.py          # Request and response schemas
│   └── database.py             # Database connection and setup
├── model/
│   └── regression_model.pkl    # Serialized regression model
├── data/
│   └── predictions.csv         # Saved predictions (if using CSV)
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```


## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Phonginhere/cos30049.git
   cd air-quality-health-api
   ```

### Create a virtual environment
```plaintext
Use Anaconda Navigator for your own os
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the application
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## API Documentation
FastAPI provides interactive API documentation at:

* Swagger UI: http://127.0.0.1:8000/docs
* ReDoc: http://127.0.0.1:8000/redoc
## Endpoints
* GET /: Root endpoint returning a welcome message.
* GET /pollutants: Retrieve a list of pollutants.
* POST /validate: Validate input data.
* POST /predict: Predict the burden mean based on input data.
