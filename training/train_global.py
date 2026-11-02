"""Train the global quantile model (with and without exogenous features).

    python -m training.train_global [--stride 12] [--max-iter 150]

Split (identical for every series, on *target* timestamps):
    train  targets < 2016-01-01
    val    2016-01-01 <= targets < 2016-09-01   -> conformal calibration only
    test   2016-09-01 <= targets < 2017-03-01   -> never touched here
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from models.global_model import GlobalQuantileModel
from models.preprocessing import (T_TRAIN_END, TRAIN_HORIZONS, build_origin_features,
                                  expand_horizons)
from training.calibrate import (ARTIFACTS, calibrate_table, global_table, log_run, save_cal)
from training.prepare_data import load_processed


def training_frame(frames: dict[str, pd.DataFrame], exog: bool, stride_h: int,
                   train_end: pd.Timestamp = T_TRAIN_END) -> tuple[pd.DataFrame, np.ndarray]:
    Xs, ys = [], []
    for sid, fr in frames.items():
        F = build_origin_features(fr, exog)
        origins = F.index[168::stride_h]                       # skip warm-up, then stride
        X = expand_horizons(F.loc[origins], TRAIN_HORIZONS)
        X = X[X["target_ts"] < train_end]                       # label must be strictly before T1
        y = fr["target"].reindex(X["target_ts"]).to_numpy() / X["_scale"].to_numpy()
        ok = np.isfinite(y)
        Xs.append(X[ok]); ys.append(y[ok])
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def train_variant(frames: dict, exog: bool, stride_h: int, params: dict, val_stride_h: int = 12,
                  out_name: str | None = None, quiet: bool = False) -> tuple[GlobalQuantileModel, dict]:
    t0 = time.time()
    X, y = training_frame(frames, exog, stride_h)
    assert X["target_ts"].max() < T_TRAIN_END, "leakage: training label at/after split boundary"
    model = GlobalQuantileModel(exog, params).fit(X, y)
    val = global_table(model, frames, "val", val_stride_h)
    cal = calibrate_table(val, {"model": "global", "exog": exog})
    if out_name:
        d = ARTIFACTS / out_name
        meta = model.save(d, {"train_rows": int(len(X)), "train_end": str(T_TRAIN_END),
                              "series": sorted(frames), "stride_h": stride_h})
        save_cal(cal, d)
        log_run({"event": "train_global", "variant": out_name, "rows": len(X), "params": model.params,
                 "seconds": round(time.time() - t0, 1), "sha256": meta["sha256"][:12],
                 "val_cov": cal["cal_val_coverage"], "val_cov_raw": cal["raw_val_coverage"]})
    if not quiet:
        print(f"  exog={exog}: {len(X):,} rows, {X.shape[1]} cols, {time.time()-t0:.0f}s | "
              f"val coverage raw {cal['raw_val_coverage']} -> calibrated {cal['cal_val_coverage']}")
    return model, cal


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12, help="hours between training origins")
    ap.add_argument("--max-iter", type=int, default=150)
    a = ap.parse_args()
    frames = load_processed()
    print(f"Training on {len(frames)} series: {sorted(frames)}")
    for exog, name in [(True, "global_exog"), (False, "global_noexog")]:
        train_variant(frames, exog, a.stride, {"max_iter": a.max_iter}, out_name=name)
