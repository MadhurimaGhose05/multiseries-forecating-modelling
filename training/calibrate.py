"""Evaluation windows, prediction tables and conformal calibration.

Both models are evaluated through the *same* table format, produced from the
*same* origins, so the comparison is apples to apples:

    columns: series_id, origin_ts, target_ts, h, y, scale, naive, snaive, q0..q4
    q0..q4 are the raw quantile forecasts (0.05/0.25/0.5/0.75/0.95) divided by `scale`.

Time-ordering guarantees (no leakage):
  * a row belongs to a window iff its *target* timestamp is inside the window;
  * calibration uses only the validation window, metrics only the test window;
  * every input feature at an origin uses data <= origin.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.uncertainty import fit_calibration, save_calibration
from models.preprocessing import (MAX_HORIZON, T_TEST_END, T_TRAIN_END, T_VAL_END,
                                  build_origin_features, expand_horizons)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
WINDOWS = {"train": (pd.Timestamp("2013-03-08"), T_TRAIN_END), "val": (T_TRAIN_END, T_VAL_END), "test": (T_VAL_END, T_TEST_END)}
HORIZONS = list(range(1, MAX_HORIZON + 1))


def log_run(record: dict) -> None:
    """Plain structured experiment log (one JSON object per line)."""
    ARTIFACTS.mkdir(exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(ARTIFACTS / "runs.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def window_origins(index: pd.DatetimeIndex, window: str, stride_h: int) -> pd.DatetimeIndex:
    """Origins whose entire 1..48h forecast path lies inside the window."""
    start, end = WINDOWS[window]
    keep = (index >= start) & (index <= end - pd.Timedelta(hours=MAX_HORIZON))
    o = index[keep]
    return o[(o - o[0]) // pd.Timedelta(hours=1) % stride_h == 0]


def base_table(frame: pd.DataFrame, sid: str, origin: np.ndarray, h: np.ndarray, scale: pd.Series) -> pd.DataFrame:
    """Truth, scale and trivial baselines for aligned (origin, h) arrays (any row order)."""
    o = pd.DatetimeIndex(origin)
    h = np.asarray(h).astype(int)
    t = o + pd.to_timedelta(h, unit="h")
    y = frame["target"]
    k = np.ceil(h / 24).astype(int)
    return pd.DataFrame({
        "series_id": sid, "origin_ts": o, "target_ts": t, "h": h,
        "y": y.reindex(t).values,
        "scale": scale.reindex(o).values,
        "naive": y.ffill(limit=24).reindex(o).values,                              # persistence
        "snaive": y.reindex(t - pd.to_timedelta(24 * k, unit="h")).values,          # same hour, last day
    })


def global_table(model, frames: dict[str, pd.DataFrame], window: str, stride_h: int,
                 series: list[str] | None = None, feats_cache: dict | None = None) -> pd.DataFrame:
    parts = []
    for sid in series or list(frames):
        fr = frames[sid]
        F = (feats_cache or {}).get(sid)
        if F is None:
            F = build_origin_features(fr, exog=model.exog)
        origins = window_origins(fr.index, window, stride_h)
        X = expand_horizons(F.loc[origins], HORIZONS)
        P = model.predict(X)
        tbl = base_table(fr, sid, X["origin_ts"].values, X["h"].values, F["_scale"])
        tbl[[f"q{i}" for i in range(5)]] = P
        parts.append(tbl)
    return pd.concat(parts, ignore_index=True)


def calibrate_table(tbl: pd.DataFrame, extra: dict | None = None) -> dict:
    y = (tbl["y"] / tbl["scale"]).to_numpy()
    q = tbl[[f"q{i}" for i in range(5)]].to_numpy()
    cal = fit_calibration(y, q, tbl["h"].to_numpy())
    cal["series"] = sorted(tbl["series_id"].unique().tolist())
    cal["window"] = [str(WINDOWS["val"][0]), str(WINDOWS["val"][1])]
    cal.update(extra or {})
    return cal


def save_cal(cal: dict, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    save_calibration(cal, model_dir / "calibration.json")
