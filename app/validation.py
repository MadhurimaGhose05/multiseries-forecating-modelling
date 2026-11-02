"""Input validation for the API. Framework-free so it is unit-testable.

Every failure raises ApiError(status, code, message, details) which main.py
renders as {"error": {"code", "message", "details"}} -- one error envelope for
malformed files, bad parameters, unknown models and server faults.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from models.preprocessing import (AUX, COV, MAX_HORIZON, NUMERIC, SCHEMA, normalise_cat, to_hourly)

MAX_BYTES = 25 * 1024 * 1024
MIN_ROWS = 24 * 14            # two weeks of hourly history
MIN_OBS_LAST_168 = 48         # at least 48 observed targets in the last week
MAX_STALE_HOURS = 72          # target must have been observed within 72h of the last timestamp
VALID_INTERVALS = (50, 90)
VALID_MODELS = ("global", "chronos")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details or {}

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


def validate_params(horizon, interval, model, use_exog=True) -> tuple[int, int, str, bool]:
    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        raise ApiError(422, "invalid_horizon", f"horizon must be an integer, got {horizon!r}")
    if not 1 <= horizon <= MAX_HORIZON:
        raise ApiError(422, "invalid_horizon", f"horizon must be between 1 and {MAX_HORIZON} hours, got {horizon}",
                       {"min": 1, "max": MAX_HORIZON})
    try:
        interval = int(interval)
    except (TypeError, ValueError):
        raise ApiError(422, "invalid_interval", f"interval must be one of {VALID_INTERVALS}, got {interval!r}")
    if interval not in VALID_INTERVALS:
        raise ApiError(422, "invalid_interval", f"interval must be one of {VALID_INTERVALS}, got {interval}",
                       {"allowed": list(VALID_INTERVALS)})
    model = str(model).strip().lower()
    if model not in VALID_MODELS:
        raise ApiError(422, "unknown_model", f"model must be one of {VALID_MODELS}, got {model!r}",
                       {"allowed": list(VALID_MODELS)})
    return horizon, interval, model, bool(use_exog)


def parse_csv(raw: bytes, series_id: str | None = None) -> tuple[pd.DataFrame, dict]:
    """-> (regular hourly frame indexed by timestamp, info). Raises ApiError."""
    if not raw:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    if len(raw) > MAX_BYTES:
        raise ApiError(413, "file_too_large", f"File exceeds {MAX_BYTES // (1024 * 1024)} MB.")
    if b"\x00" in raw[:4096]:
        raise ApiError(400, "malformed_file", "File looks binary, expected a UTF-8 CSV.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ApiError(400, "malformed_file", f"Could not parse file as CSV: {e}")
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ApiError(422, "schema_mismatch", f"Missing required columns: {missing}",
                       {"missing": missing, "expected": SCHEMA})
    df = df[SCHEMA].copy()

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    bad = int(ts.isna().sum())
    if bad:
        raise ApiError(422, "invalid_timestamp", f"{bad} timestamp value(s) could not be parsed.", {"bad_rows": bad})
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df["timestamp"] = ts

    ids = df["series_id"].dropna().astype(str).unique().tolist()
    if not ids:
        raise ApiError(422, "missing_series_id", "series_id column is empty.")
    if len(ids) > 1:
        if series_id is None:
            raise ApiError(422, "multiple_series", "File holds several series; pass `series_id` to choose one.",
                           {"series_ids": ids[:20]})
        if series_id not in ids:
            raise ApiError(422, "unknown_series_id", f"series_id {series_id!r} not in file.", {"series_ids": ids[:20]})
        df = df[df["series_id"].astype(str) == series_id]
    elif series_id is not None and series_id != ids[0]:
        raise ApiError(422, "unknown_series_id", f"series_id {series_id!r} not in file.", {"series_ids": ids})
    sid = series_id or ids[0]

    for c in NUMERIC:
        coerced = pd.to_numeric(df[c], errors="coerce")
        newly_bad = int((coerced.isna() & df[c].notna() & (df[c].astype(str).str.strip() != "")).sum())
        if newly_bad:
            raise ApiError(422, "non_numeric_values", f"Column {c!r} has {newly_bad} non-numeric value(s).",
                           {"column": c, "bad_rows": newly_bad})
        if np.isinf(coerced).any():
            raise ApiError(422, "non_finite_values", f"Column {c!r} contains infinite values.", {"column": c})
        df[c] = coerced

    if df["timestamp"].duplicated().any():
        raise ApiError(422, "duplicate_timestamps", "Duplicate timestamps found.",
                       {"n_duplicates": int(df["timestamp"].duplicated().sum())})
    df = df.sort_values("timestamp")
    steps = df["timestamp"].diff().dropna()
    if len(steps) == 0:
        raise ApiError(422, "insufficient_history", "Only one row provided.")
    if (steps % pd.Timedelta(hours=1)).ne(pd.Timedelta(0)).any():
        raise ApiError(422, "irregular_frequency", "Timestamps must lie on a regular hourly grid.",
                       {"observed_min_step": str(steps.min())})
    hourly = to_hourly(df)
    inserted = len(hourly) - len(df)

    if len(hourly) < MIN_ROWS:
        raise ApiError(422, "insufficient_history", f"Need at least {MIN_ROWS} hourly rows, got {len(hourly)}.",
                       {"rows": len(hourly), "required": MIN_ROWS})
    tgt = hourly["target"]
    n_obs_week = int(tgt.tail(168).notna().sum())
    if n_obs_week < MIN_OBS_LAST_168:
        raise ApiError(422, "insufficient_recent_data",
                       f"Only {n_obs_week} observed target values in the last 168h; need {MIN_OBS_LAST_168}.",
                       {"observed": n_obs_week, "required": MIN_OBS_LAST_168})
    last_obs = tgt.last_valid_index()
    stale = (hourly.index[-1] - last_obs) / pd.Timedelta(hours=1)
    if stale > MAX_STALE_HOURS:
        raise ApiError(422, "stale_target", f"Last observed target is {int(stale)}h before the end of the file "
                                            f"(max {MAX_STALE_HOURS}h).", {"hours_stale": int(stale)})

    warnings = []
    _, n_unknown = normalise_cat(hourly["cov_cat"])
    if n_unknown:
        warnings.append(f"{n_unknown} cov_cat value(s) outside the known 16 buckets were mapped to 'other'.")
    if inserted:
        warnings.append(f"{inserted} missing hourly timestamp(s) were inserted as missing values (not interpolated).")
    miss = {c: round(float(hourly[c].tail(168).isna().mean()), 3) for c in [*AUX, *COV]
            if hourly[c].tail(168).isna().mean() > 0.2}
    if miss:
        warnings.append(f"High missingness in the last 168h: {miss}.")
    if hourly["target"].tail(1).isna().any():
        warnings.append("Target is missing at the final timestamp; forecasting from the last observed history.")
    info = {"series_id": sid, "n_rows": int(len(hourly)), "start": str(hourly.index[0]), "end": str(hourly.index[-1]),
            "warnings": warnings}
    return hourly, info
