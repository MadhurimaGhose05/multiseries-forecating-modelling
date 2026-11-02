"""FastAPI service.  uvicorn app.main:app --port 8000

POST /forecast   multipart CSV + horizon + interval + model + use_exog -> JSON
POST /backtest   same inputs; rolling-origin MAE / coverage on the uploaded series
GET  /health     liveness
GET  /models     artifact versions (hash, calibration window, foundation-model revision)
"""
from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import forecasting
from app.schemas import BacktestResponse, ErrorResponse, ForecastResponse
from app.validation import MAX_BYTES, ApiError, parse_csv, validate_params
from models.preprocessing import SEED

log = logging.getLogger("forecast-api")
random.seed(SEED)
np.random.seed(SEED)

app = FastAPI(title="Multi-series forecasting service", version="1.0.0")
store = forecasting.ModelStore()

ERRORS = {400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
          503: {"model": ErrorResponse}}


@app.exception_handler(ApiError)
async def _api_error(_: Request, e: ApiError):
    return JSONResponse(status_code=e.status, content=e.to_dict())


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, e: RequestValidationError):
    fields = [{"field": ".".join(str(x) for x in err["loc"]), "problem": err["msg"]} for err in e.errors()]
    return JSONResponse(status_code=422, content={"error": {"code": "invalid_request",
                        "message": "Request parameters are missing or malformed.", "details": {"errors": fields}}})


@app.exception_handler(Exception)
async def _unhandled(_: Request, e: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": {"code": "internal_error",
                        "message": "Unexpected server error.", "details": {}}})


def _read(file: UploadFile) -> bytes:
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ApiError(413, "file_too_large", f"File exceeds {MAX_BYTES // (1024 * 1024)} MB.")
    return raw


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return store.describe()


@app.post("/forecast", response_model=ForecastResponse, responses=ERRORS)
def forecast(file: UploadFile = File(..., description="CSV in the training schema"),
             horizon: int = Form(24), interval: int = Form(90), model: str = Form("global"),
             use_exog: bool = Form(True), series_id: Optional[str] = Form(None)):
    horizon, interval, model, use_exog = validate_params(horizon, interval, model, use_exog)
    hist, info = parse_csv(_read(file), series_id)
    out = forecasting.forecast(store, hist, horizon, interval, model, use_exog)
    cal = store.calibration(model, use_exog)
    return {**out, "series_id": info["series_id"], "warnings": info["warnings"],
            "model_version": {"calibrated_on": cal.get("window"), "calibration_n": cal.get("n"),
                              "foundation_model": store.chronos.info() if model == "chronos" else None}}


@app.post("/backtest", response_model=BacktestResponse, responses=ERRORS)
def backtest(file: UploadFile = File(...), horizon: int = Form(24), interval: int = Form(90),
             model: str = Form("global"), use_exog: bool = Form(True), series_id: Optional[str] = Form(None),
             n_origins: int = Form(14)):
    horizon, interval, model, use_exog = validate_params(horizon, interval, model, use_exog)
    if not 1 <= n_origins <= 60:
        raise ApiError(422, "invalid_n_origins", "n_origins must be between 1 and 60.", {"min": 1, "max": 60})
    hist, info = parse_csv(_read(file), series_id)
    out = forecasting.backtest(store, hist, horizon, interval, model, use_exog, n_origins)
    return {**out, "series_id": info["series_id"], "warnings": info["warnings"]}
