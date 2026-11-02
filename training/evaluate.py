"""Evaluate every model on the same time-ordered TEST window (2016-09-01 -> 2017-02-28).

    python -m training.evaluate                 # global models (+ chronos if its calibration exists)
    python -m training.evaluate --loso          # + leave-one-series-out for the global model
    python -m training.evaluate --chronos       # also run Chronos-2 on the test origins (needs weights)

Both models are scored on identical origins (every `--stride` hours, full 48h path inside the window),
with MAE / RMSE / sMAPE on the median forecast, plus empirical coverage of the raw and the
conformal-calibrated 50% / 90% intervals. Offsets were fit on the validation window only.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from app.forecasting import point_metrics
from app.uncertainty import QUANTILES, calibrated_interval, coverage, load_calibration, LEVELS
from models.global_model import GlobalQuantileModel
from models.preprocessing import HORIZON_BUCKETS, bucket_label
from training.calibrate import ARTIFACTS, global_table, log_run
from training.prepare_data import load_processed

Q = [f"q{i}" for i in range(len(QUANTILES))]


def evaluate_table(tbl: pd.DataFrame, cal: dict) -> dict:
    t = tbl.copy()
    s = t["scale"].to_numpy()
    t["point"] = np.maximum(t["q2"].to_numpy() * s, 0)
    q = t[Q].to_numpy()
    h = t["h"].to_numpy()
    y = t["y"].to_numpy()
    for level, (i, j) in LEVELS.items():
        lo, hi = calibrated_interval(q, h, cal, level)
        t[f"lo{level}"], t[f"hi{level}"] = np.maximum(lo * s, 0), hi * s
        t[f"rlo{level}"], t[f"rhi{level}"] = np.maximum(q[:, i] * s, 0), q[:, j] * s

    def block(d: pd.DataFrame) -> dict:
        yy = d["y"].to_numpy()
        m = point_metrics(yy, d["point"].to_numpy())
        ok = np.isfinite(d["y"]) & np.isfinite(d["naive"]) & np.isfinite(d["snaive"])
        out = {"mae": m["mae"], "rmse": m["rmse"], "smape": m["smape"], "n": m["n"],
               "naive_mae": point_metrics(d["y"].to_numpy()[ok], d["naive"].to_numpy()[ok])["mae"],
               "snaive_mae": point_metrics(d["y"].to_numpy()[ok], d["snaive"].to_numpy()[ok])["mae"]}
        for level in LEVELS:
            out[f"cov{level}"] = coverage(yy, d[f"lo{level}"].to_numpy(), d[f"hi{level}"].to_numpy())
            out[f"raw_cov{level}"] = coverage(yy, d[f"rlo{level}"].to_numpy(), d[f"rhi{level}"].to_numpy())
            out[f"width{level}"] = float(np.nanmean(d[f"hi{level}"] - d[f"lo{level}"]))
        return out

    t["bucket"] = [bucket_label(int(x)) for x in t["h"]]
    return {"overall": block(t),
            "by_horizon": {f"{a}-{b}": block(t[t["bucket"] == f"{a}-{b}"]) for a, b in HORIZON_BUCKETS},
            "by_series": {sid: block(g) for sid, g in t.groupby("series_id")}}


def to_markdown(results: dict) -> str:
    rows = ["| Model | Exog | MAE | RMSE | sMAPE % | 50% cov (raw → cal) | 90% cov (raw → cal) | 90% width |",
            "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if name.startswith("_"):
            continue
        o = r["overall"]
        model, _, ex = name.partition("_")
        rows.append(f"| {model} | {'yes' if ex == 'exog' else 'no'} | {o['mae']:.2f} | {o['rmse']:.2f} | {o['smape']:.1f} | "
                    f"{o['raw_cov50']:.3f} → {o['cov50']:.3f} | {o['raw_cov90']:.3f} → {o['cov90']:.3f} | {o['width90']:.1f} |")
    any_r = next(v for k, v in results.items() if not k.startswith("_"))["overall"]
    rows.append(f"| baseline: persistence | – | {any_r['naive_mae']:.2f} | | | | | |")
    rows.append(f"| baseline: seasonal-naive 24h | – | {any_r['snaive_mae']:.2f} | | | | | |")
    return "\n".join(rows)


def run_loso(frames: dict, stride_h: int, train_stride: int, max_iter: int) -> dict:
    """Leave-one-series-out: train+calibrate on 7 series (pre-2016 / 2016-val), score the 8th on test."""
    from training.train_global import train_variant
    per, tables = {}, []
    for sid in sorted(frames):
        others = {k: v for k, v in frames.items() if k != sid}
        model, cal = train_variant(others, True, train_stride, {"max_iter": max_iter}, val_stride_h=24, quiet=True)
        tbl = global_table(model, frames, "test", stride_h, series=[sid])
        per[sid] = evaluate_table(tbl, cal)["overall"]
        tables.append(tbl)
        print(f"  LOSO {sid}: MAE {per[sid]['mae']:.2f}  cov90 {per[sid]['cov90']:.3f}", flush=True)
    return per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12, help="hours between evaluation origins (same for all models)")
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--loso-max-iter", type=int, default=80)
    ap.add_argument("--loso-train-stride", type=int, default=24)
    ap.add_argument("--chronos", action="store_true", help="run Chronos-2 on the test origins (needs weights)")
    a = ap.parse_args()
    frames = load_processed()
    results: dict = {}
    for name, exog in [("global_exog", True), ("global_noexog", False)]:
        model = GlobalQuantileModel.load(ARTIFACTS / name)
        cal = load_calibration(ARTIFACTS / name / "calibration.json")
        results[name] = evaluate_table(global_table(model, frames, "test", a.stride), cal)
    if a.chronos:
        from models.chronos_model import ChronosForecaster
        from training.train_chronos import chronos_table
        fc = ChronosForecaster()
        for name, exog in [("chronos_exog", True), ("chronos_noexog", False)]:
            cal = load_calibration(ARTIFACTS / name / "calibration.json")
            results[name] = evaluate_table(chronos_table(fc, frames, "test", a.stride, exog), cal)
    if a.loso:
        results["_loso_global_exog"] = run_loso(frames, a.stride, a.loso_train_stride, a.loso_max_iter)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    md = to_markdown(results)
    (ARTIFACTS / "results.md").write_text(md + "\n")
    log_run({"event": "evaluate", "stride": a.stride, "models": [k for k in results if not k.startswith("_")]})
    print(md)
