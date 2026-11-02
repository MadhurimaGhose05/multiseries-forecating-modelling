"""Single source of truth for schema, cleaning and feature engineering.

Both training (training/*.py) and serving (app/forecasting.py) import
`build_origin_features` and `expand_horizons` from here. There is no second
implementation of any feature, which is how train-serve skew is avoided.

Every feature at forecast origin t uses data with timestamp <= t only
(rolling windows and shifts are strictly causal). tests/test_forecasting.py
asserts this by truncating a series and comparing features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
FREQ = "h"
MAX_HORIZON = 48

AUX = [f"aux_{i}" for i in range(1, 6)]
COV = [f"cov_{i}" for i in range(1, 6)]
SCHEMA = ["timestamp", "series_id", "target", *AUX, *COV, "cov_cat"]
NUMERIC = ["target", *AUX, *COV]

# Fixed vocabulary => identical category codes in training and serving.
CAT_VOCAB = ["E", "ENE", "ESE", "N", "NE", "NNE", "NNW", "NW",
             "S", "SE", "SSE", "SSW", "SW", "W", "WNW", "WSW"]
OTHER = "other"
CAT_LEVELS = CAT_VOCAB + [OTHER]
_COMPASS = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
            "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
            "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}

# Chronological split applied identically to every series (on *target* timestamps).
T_TRAIN_END = pd.Timestamp("2016-01-01")   # train: targets  < T1
T_VAL_END = pd.Timestamp("2016-09-01")     # val:   T1 <= targets < T2 (calibration)
T_TEST_END = pd.Timestamp("2017-03-01")    # test:  T2 <= targets < T3

# Horizons seen in training (the model interpolates over `h`); serving uses 1..H.
TRAIN_HORIZONS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36, 42, 48]
HORIZON_BUCKETS = [(1, 6), (7, 12), (13, 24), (25, 48)]

Y_LAGS = [0, 1, 2, 3, 6, 12, 23, 24, 25, 47, 48, 72, 168]
AUX_LAGS = [0, 1, 6, 24]
COV_FFILL_LIMIT = 6   # training AND serving: short covariate gaps are forward-filled


def bucket_label(h: int) -> str:
    for lo, hi in HORIZON_BUCKETS:
        if lo <= h <= hi:
            return f"{lo}-{hi}"
    return f"{HORIZON_BUCKETS[-1][0]}-{HORIZON_BUCKETS[-1][1]}"


def normalise_cat(s: pd.Series) -> tuple[pd.Series, int]:
    """Map raw cov_cat strings onto CAT_LEVELS. Unknown labels -> 'other'.
    Missing stays missing. Returns (categorical series, number of unknown rows)."""
    raw = s.astype("object")
    cleaned = raw.where(raw.isna(), raw.astype(str).str.strip().str.upper())
    unknown = cleaned.notna() & ~cleaned.isin(CAT_VOCAB)
    cleaned = cleaned.mask(unknown, OTHER)
    return pd.Series(pd.Categorical(cleaned, categories=CAT_LEVELS), index=s.index), int(unknown.sum())


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, index by timestamp and reindex onto a complete hourly grid.
    Inserted rows are all-NaN ('missing'); nothing is interpolated."""
    d = df.sort_values("timestamp").set_index("timestamp")
    grid = pd.date_range(d.index.min(), d.index.max(), freq=FREQ)
    d = d.reindex(grid)
    d.index.name = "timestamp"
    return d


def compute_scale(y: pd.Series) -> pd.Series:
    """Trailing 168h mean of the target (causal). Used to normalise targets, lags
    and conformal scores so one model / one calibration transfers across series."""
    s = y.rolling(168, min_periods=24).mean()
    s = s.fillna(y.expanding(min_periods=1).mean())
    return s.fillna(1.0).clip(lower=1.0)


def build_origin_features(df: pd.DataFrame, exog: bool = True) -> pd.DataFrame:
    """Features for *every* timestamp t of a regular hourly frame, each using only
    data <= t. Returns a frame with the same index plus a `_scale` column
    (not a model feature) used to de-normalise predictions."""
    d = df.copy()
    d[COV] = d[COV].astype(float).ffill(limit=COV_FFILL_LIMIT)
    y = d["target"].astype(float)
    scale = compute_scale(y)
    f: dict[str, pd.Series] = {}

    for k in Y_LAGS:
        f[f"y_lag{k}"] = y.shift(k) / scale
    for w in (6, 24, 168):
        r = y.rolling(w, min_periods=max(2, w // 4))
        f[f"y_mean{w}"] = r.mean() / scale
        f[f"y_std{w}"] = r.std() / scale
        f[f"y_min{w}"] = r.min() / scale
        f[f"y_max{w}"] = r.max() / scale
    f["log_scale"] = np.log(scale)

    # --- data-quality features: the model sees *how* the history is missing ---
    pos = pd.Series(np.arange(len(d), dtype=float), index=d.index)
    last_obs = pos.where(y.notna()).ffill()
    f["hours_since_obs"] = (pos - last_obs).fillna(168).clip(upper=168)
    obs = y.notna().astype(float)
    f["miss_frac24"] = 1 - obs.rolling(24, min_periods=1).mean()
    f["miss_frac168"] = 1 - obs.rolling(168, min_periods=1).mean()
    f["origin_hour"] = pd.Series(d.index.hour.astype(float), index=d.index)

    if exog:
        # aux_*: observed *with* the target, so unknown in the future -> lags only.
        for j in AUX:
            a = d[j].astype(float)
            base = a.rolling(168, min_periods=24).mean().abs().clip(lower=1e-6)
            for l in AUX_LAGS:
                f[f"{j}_lag{l}"] = a.shift(l) / base
            f[f"{j}_mean24"] = a.rolling(24, min_periods=6).mean() / base
        f["aux_miss_frac24"] = d[AUX].isna().mean(axis=1).rolling(24, min_periods=1).mean()
        # cov_*: physical units shared across series -> raw value at origin + tendencies.
        for c in COV:
            v = d[c]
            f[c] = v
            f[f"{c}_d3"] = v - v.shift(3)
            f[f"{c}_d24"] = v - v.shift(24)
            f[f"{c}_mean24"] = v.rolling(24, min_periods=6).mean()
        cat, _ = normalise_cat(d["cov_cat"].astype("object").ffill(limit=COV_FFILL_LIMIT))
        f["cov_cat"] = cat
        ang = np.deg2rad(cat.astype("object").map(_COMPASS).astype(float))
        f["wind_sin"], f["wind_cos"] = np.sin(ang), np.cos(ang)

    out = pd.DataFrame(f, index=d.index)
    for c in out.columns:
        if c != "cov_cat":
            out[c] = out[c].astype("float32")
    out["_scale"] = scale.astype("float64")
    return out


def expand_horizons(origin_feats: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Cross origin rows with horizons and add calendar features of the *target*
    time (known in advance). Meta columns (`origin_ts`, `target_ts`, `_scale`)
    are carried along; `model_columns` lists the real model inputs."""
    parts = []
    for h in horizons:
        p = origin_feats.copy()
        p.index.name = "origin_ts"
        p = p.reset_index()
        p["h"] = np.float32(h)
        p["target_ts"] = p["origin_ts"] + pd.Timedelta(hours=h)
        t = p["target_ts"]
        p["tgt_hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24).astype("float32")
        p["tgt_hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24).astype("float32")
        p["tgt_dow"] = t.dt.dayofweek.astype("float32")
        p["tgt_month_sin"] = np.sin(2 * np.pi * (t.dt.month - 1) / 12).astype("float32")
        p["tgt_month_cos"] = np.cos(2 * np.pi * (t.dt.month - 1) / 12).astype("float32")
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    if "cov_cat" in out:
        out["cov_cat"] = pd.Categorical(out["cov_cat"].astype("object"), categories=CAT_LEVELS)
    return out


META_COLS = {"origin_ts", "target_ts", "_scale"}


def model_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in META_COLS]
