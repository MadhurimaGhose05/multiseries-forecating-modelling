"""Serving logic: history DataFrame in -> point forecast + calibrated intervals out.

All features are recomputed from the uploaded history by the *same* functions
used in training (models.preprocessing). The model never receives anything the
caller did not upload, and nothing at or after the forecast origin is used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.uncertainty import QUANTILES, calibrated_interval, load_calibration
from app.validation import ApiError
from models.chronos_model import ChronosForecaster, ModelUnavailable
from models.global_model import GlobalQuantileModel
from models.preprocessing import build_origin_features, compute_scale, expand_horizons

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
Q = [f"q{i}" for i in range(len(QUANTILES))]


def _variant(model: str, exog: bool) -> str:
    return f"{model}_{'exog' if exog else 'noexog'}"


class ModelStore:
    """Loads versioned artifacts once; refuses to start on a corrupted artifact (hash check)."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS, chronos: ChronosForecaster | None = None):
        self.dir = Path(artifacts_dir)
        self.chronos = chronos or ChronosForecaster()
        self._global: dict[bool, GlobalQuantileModel] = {}
        self._cal: dict[str, dict] = {}

    def calibration(self, model: str, exog: bool) -> dict:
        v = _variant(model, exog)
        if v not in self._cal:
            p = self.dir / v / "calibration.json"
            if not p.exists():
                raise ApiError(503, "model_unavailable",
                               f"No calibration artifact for '{v}'. Run the training pipeline first.", {"variant": v})
            self._cal[v] = load_calibration(p)
        return self._cal[v]

    def global_model(self, exog: bool) -> GlobalQuantileModel:
        if exog not in self._global:
            d = self.dir / _variant("global", exog)
            if not (d / "model.joblib").exists():
                raise ApiError(503, "model_unavailable", f"No global model artifact at {d.name}.", {"variant": d.name})
            self._global[exog] = GlobalQuantileModel.load(d)
        return self._global[exog]

    def describe(self) -> dict:
        out = {}
        for d in sorted(self.dir.glob("*_*exog")):
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            cal = json.loads((d / "calibration.json").read_text()) if (d / "calibration.json").exists() else {}
            out[d.name] = {"sha256": meta.get("sha256", "")[:16], "train_end": meta.get("train_end"),
                           "calibration_n": cal.get("n"), "calibrated_on": cal.get("window")}
        out["chronos"] = self.chronos.info()
        return out


# --------------------------------------------------------------------------- prediction tables
def predict_table(store: ModelStore, model: str, exog: bool, hist: pd.DataFrame,
                  origins: list[pd.Timestamp], horizon: int) -> pd.DataFrame:
    """Normalised raw quantiles for every (origin, h): columns origin_ts, h, scale, q0..q4."""
    if model == "global":
        m = store.global_model(exog)
        F = build_origin_features(hist, exog=exog)
        X = expand_horizons(F.loc[origins], list(range(1, horizon + 1)))
        P = m.predict(X)
        t = X[["origin_ts", "h", "_scale"]].rename(columns={"_scale": "scale"})
        t["h"] = t["h"].astype(int)
        t[Q] = P
        return t.reset_index(drop=True)
    try:
        arrays = store.chronos.predict([hist.loc[:o] for o in origins], horizon, exog)
    except ModelUnavailable as e:
        raise ApiError(503, "model_unavailable", f"Foundation model unavailable: {e}")
    scale = compute_scale(hist["target"].astype(float))
    parts = []
    for o, a in zip(origins, arrays):
        s = float(scale.loc[o])
        t = pd.DataFrame(a / s, columns=Q)
        t.insert(0, "scale", s); t.insert(0, "h", np.arange(1, horizon + 1)); t.insert(0, "origin_ts", o)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def add_intervals(t: pd.DataFrame, cal: dict, level: int) -> pd.DataFrame:
    """Adds point/lower/upper in original units (conformal-calibrated, floored at 0)."""
    lo, hi = calibrated_interval(t[Q].to_numpy(), t["h"].to_numpy(), cal, level)
    s = t["scale"].to_numpy()
    t = t.copy()
    t["point"] = np.maximum(t["q2"].to_numpy() * s, 0)
    t["lower"] = np.minimum(np.maximum(lo * s, 0), t["point"])
    t["upper"] = np.maximum(hi * s, t["point"])
    return t


# --------------------------------------------------------------------------- public API
def forecast(store: ModelStore, hist: pd.DataFrame, horizon: int, interval: int, model: str, exog: bool) -> dict:
    cal = store.calibration(model, exog)
    origin = hist.index[-1]
    t = add_intervals(predict_table(store, model, exog, hist, [origin], horizon), cal, interval)
    ts = origin + pd.to_timedelta(t["h"].to_numpy(), unit="h")
    return {
        "model": model, "use_exog": exog, "horizon": horizon, "interval": interval, "origin": str(origin),
        "forecast": [{"timestamp": str(a), "point": round(float(p), 3), "lower": round(float(l), 3),
                      "upper": round(float(u), 3)} for a, p, l, u in zip(ts, t["point"], t["lower"], t["upper"])],
        "mean_interval_width": round(float((t["upper"] - t["lower"]).mean()), 3),
    }


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "smape": float("nan"), "n": 0}
    return {"mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "smape": float(100 * np.mean(2 * np.abs(y - p) / np.maximum(np.abs(y) + np.abs(p), 1e-9))), "n": int(len(y))}


def backtest(store: ModelStore, hist: pd.DataFrame, horizon: int, interval: int, model: str, exog: bool,
             n_origins: int = 14) -> dict:
    """Rolling-origin check on the *uploaded* series: daily origins in the last
    `n_origins` days whose full horizon is already observed. Each origin only sees
    history up to itself (features are causal), so this is leak-free."""
    cal = store.calibration(model, exog)
    last = hist.index[-1]
    cand = [last - pd.Timedelta(hours=horizon + 24 * i) for i in range(n_origins)]
    origins = [o for o in cand if o >= hist.index[168]]
    if not origins:
        raise ApiError(422, "insufficient_history", "Not enough history left over to backtest this horizon.")
    t = add_intervals(predict_table(store, model, exog, hist, origins, horizon), cal, interval)
    tt = t["origin_ts"] + pd.to_timedelta(t["h"], unit="h")
    y = hist["target"].reindex(tt).to_numpy()
    naive = hist["target"].ffill(limit=24).reindex(t["origin_ts"]).to_numpy()
    m = point_metrics(y, t["point"].to_numpy())
    ok = np.isfinite(y)
    inside = (y >= t["lower"].to_numpy()) & (y <= t["upper"].to_numpy())
    cov = float(inside[ok].mean()) if ok.any() else float("nan")
    return {"model": model, "use_exog": exog, "horizon": horizon, "interval": interval, "n_origins": len(origins),
            "mae": round(m["mae"], 3), "rmse": round(m["rmse"], 3), "smape": round(m["smape"], 2),
            "coverage": round(cov, 4), "nominal_coverage": interval / 100,
            "mean_interval_width": round(float((t["upper"] - t["lower"]).mean()), 3),
            "naive_mae": round(point_metrics(y, naive)["mae"], 3), "n_points": m["n"]}
