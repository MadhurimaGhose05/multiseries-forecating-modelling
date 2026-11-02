"""Response models (the public API contract). Request is multipart/form-data:

    file       .csv in the training schema (required)
    horizon    int 1..48 hours ahead            (default 24)
    interval   50 or 90 (percent)               (default 90)
    model      "global" | "chronos"             (default "global")
    use_exog   bool: use aux_*/cov_*/cov_cat    (default true)
    series_id  optional, only needed if the file contains several series
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    timestamp: str
    point: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    series_id: str
    model: str
    use_exog: bool
    horizon: int
    interval: int
    origin: str = Field(description="Last timestamp of the uploaded history; forecasts start one hour later.")
    forecast: list[ForecastPoint]
    mean_interval_width: float
    warnings: list[str] = []
    model_version: dict = {}


class BacktestResponse(BaseModel):
    series_id: str
    model: str
    use_exog: bool
    horizon: int
    interval: int
    n_origins: int
    mae: float
    rmse: float
    smape: float
    coverage: float
    nominal_coverage: float
    mean_interval_width: float
    naive_mae: float
    n_points: int
    warnings: list[str] = []


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = {}


class ErrorResponse(BaseModel):
    error: ErrorBody
