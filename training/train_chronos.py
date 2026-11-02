"""Chronos-2 is used zero-shot: nothing is fitted except the conformal offsets.

    python -m training.train_chronos [--stride 12] [--smoke]

For every series and every validation origin the model sees ONLY history up to
that origin (hist.loc[:origin]); the resulting quantiles are normalised by the
same trailing-168h scale as the global model and calibrated with the same CQR
code. Writes artifacts/chronos_{exog,noexog}/{meta.json,calibration.json}.

Requires `chronos-forecasting` + `torch` and the (pinned) weights. NOT run in
the authoring sandbox -- see README "Verification status".
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from models.chronos_model import ChronosForecaster
from models.preprocessing import MAX_HORIZON, compute_scale
from training.calibrate import (ARTIFACTS, HORIZONS, base_table, calibrate_table, log_run,
                                save_cal, window_origins)
from training.prepare_data import load_processed

BATCH = 32


def chronos_table(fc: ChronosForecaster, frames: dict, window: str, stride_h: int, exog: bool,
                  series: list[str] | None = None, max_origins: int | None = None) -> pd.DataFrame:
    parts = []
    for sid in series or list(frames):
        fr = frames[sid]
        origins = window_origins(fr.index, window, stride_h)
        if max_origins:
            origins = origins[:max_origins]
        scale = compute_scale(fr["target"].astype(float))
        preds = []
        for i in range(0, len(origins), BATCH):
            chunk = origins[i:i + BATCH]
            preds += fc.predict([fr.loc[:o] for o in chunk], MAX_HORIZON, exog)   # leak-free: history <= origin
        P = np.concatenate(preds)                                                  # (n_origins*48, 5)
        o = np.repeat(origins.values, MAX_HORIZON)
        h = np.tile(np.array(HORIZONS), len(origins))
        tbl = base_table(fr, sid, o, h, scale)
        tbl[[f"q{k}" for k in range(5)]] = P / tbl["scale"].to_numpy()[:, None]
        parts.append(tbl)
        print(f"  chronos {window} {sid} exog={exog}: {len(origins)} origins", flush=True)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="tiny run (1 series, 4 origins) to verify the wiring")
    a = ap.parse_args()
    frames = load_processed()
    fc = ChronosForecaster()
    if not fc.available():
        raise SystemExit("Chronos-2 is not available (install chronos-forecasting/torch and make weights reachable).")
    series = sorted(frames)[:1] if a.smoke else None
    for exog, name in [(True, "chronos_exog"), (False, "chronos_noexog")]:
        t0 = time.time()
        val = chronos_table(fc, frames, "val", a.stride, exog, series, 4 if a.smoke else None)
        cal = calibrate_table(val, {"model": "chronos", "exog": exog})
        if not a.smoke:
            d = ARTIFACTS / name
            save_cal(cal, d)
            (d / "meta.json").write_text(json.dumps({"kind": "chronos2_zero_shot", **fc.info()}, indent=2))
        log_run({"event": "calibrate_chronos", "variant": name, "smoke": a.smoke, "seconds": round(time.time() - t0, 1),
                 "val_cov_raw": cal["raw_val_coverage"], "val_cov": cal["cal_val_coverage"]})
        print(name, "raw", cal["raw_val_coverage"], "-> calibrated", cal["cal_val_coverage"])
