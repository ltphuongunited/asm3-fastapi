from fastapi import FastAPI
from app.api.endpoints import regression

app = FastAPI()

app.include_router(regression.router)
