from pydantic import BaseModel, validator
from typing import Union

class ValidationInput(BaseModel):
    iso3: str  # 3-letter country code
    exposure_mean: Union[int, float]
    pollutant: str

    @validator('iso3')
    def iso3_must_be_valid(cls, v):
        if len(v) != 3 or not v.isalpha():
            raise ValueError("`iso3` must be a 3-letter alphabetic country code.")
        return v.upper()

    @validator('exposure_mean', pre=True)
    def validate_exposure_mean(cls, v):
        if isinstance(v, str):
            try:
                float_value = float(v)
                return float_value
            except ValueError:
                raise ValueError("`exposure_mean` must be a number, not a string.")
        elif isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("`exposure_mean` must be a positive number.")
            return float(v)
        else:
            raise ValueError("`exposure_mean` must be a number.")


class PredictionOutput(BaseModel):
    predicted_burden_mean: float
