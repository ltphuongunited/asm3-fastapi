
from fastapi import FastAPI
from FaR.backend.api.endpoints import regression

app = FastAPI()

# Include regression endpoints
app.include_router(regression.router, prefix="/api")
