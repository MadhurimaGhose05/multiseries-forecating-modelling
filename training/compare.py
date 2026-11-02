"""One table, both models, every series.

    python -m training.compare                      # global model only (works anywhere)
    python -m training.compare --chronos            # + Chronos-2 (needs weights; run train_chronos first)

For each (model, exogenous on/off, series) on the TEST window (2016-09-01 -> 2017-03-01) and the
VALIDATION window it reports MAE, RMSE, sMAPE, mean pinball loss and raw -> calibrated coverage of the
50% / 90% intervals, plus an "ALL" row per model over every series.

`seen` = the series was in the global model's training/calibration set. Chronos-2 is zero-shot: it has no
training set, so for it every series counts as unseen and the validation window is only used for the
conformal offsets. Writes artifacts/comparison.md and artifacts/comparison.json.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from app.uncertainty import load_calibration
from models.global_model import GlobalQuantileModel
from training.calibrate import ARTIFACTS, global_table
from training.evaluate import evaluate_table
from training.prepare_data import load_processed
from training.report import pinball


def rows_for(tbl: pd.DataFrame, cal: dict, model: str, exog: bool, window: str, seen: set | None) -> list[dict]:
    res = evaluate_table(tbl, cal)
    out = []
    for sid, o in [*sorted(res["by_series"].items()), ("ALL", res["overall"])]:
        sub = tbl if sid == "ALL" else tbl[tbl["series_id"] == sid]
        out.append({"model": model, "exog": "yes" if exog else "no", "window": window, "series": sid,
                    "seen": "" if sid == "ALL" or seen is None else ("yes" if sid in seen else "no"),
                    "mae": o["mae"], "rmse": o["rmse"], "smape": o["smape"], "pinball": pinball(sub),
                    "cov50_raw": o["raw_cov50"], "cov50": o["cov50"], "cov90_raw": o["raw_cov90"], "cov90": o["cov90"],
                    "persist_mae": o["naive_mae"], "n": o["n"]})
    return out


def to_markdown(df: pd.DataFrame) -> str:
    head = ("| Model | Exog | Window | Series | Seen | MAE | RMSE | sMAPE % | Pinball | 50% cov raw→cal | 90% cov raw→cal "
            "| Persistence MAE |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in df.itertuples():
        b = "**" if r.series == "ALL" else ""
        lines.append(f"| {r.model} | {r.exog} | {r.window} | {b}{r.series}{b} | {r.seen} | {b}{r.mae:.2f}{b} | {r.rmse:.2f} | "
                     f"{r.smape:.1f} | {r.pinball:.2f} | {r.cov50_raw:.3f}→{r.cov50:.3f} | {r.cov90_raw:.3f}→{r.cov90:.3f} | "
                     f"{r.persist_mae:.2f} |")
    return "\n".join(lines)


def collect(frames: dict, stride: int, windows: list[str], chronos: bool, chronos_obj=None) -> pd.DataFrame:
    rows: list[dict] = []
    for name, exog in [("global_exog", True), ("global_noexog", False)]:
        model = GlobalQuantileModel.load(ARTIFACTS / name)
        cal = load_calibration(ARTIFACTS / name / "calibration.json")
        seen = set(json.loads((ARTIFACTS / name / "meta.json").read_text())["series"])
        for w in windows:
            rows += rows_for(global_table(model, frames, w, stride), cal, "global", exog, w, seen)
    if chronos:
        from models.chronos_model import ChronosForecaster
        from training.train_chronos import chronos_table
        fc = chronos_obj or ChronosForecaster()
        if not fc.available():
            raise SystemExit("Chronos-2 weights/libraries unavailable on this machine; cannot score it.")
        for name, exog in [("chronos_exog", True), ("chronos_noexog", False)]:
            p = ARTIFACTS / name / "calibration.json"
            if not p.exists():
                raise SystemExit(f"{p} missing: run `python -m training.train_chronos` first.")
            cal = load_calibration(p)
            for w in windows:
                rows += rows_for(chronos_table(fc, frames, w, stride, exog), cal, "chronos-2", exog, w, None)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--windows", nargs="+", default=["val", "test"])
    ap.add_argument("--chronos", action="store_true")
    a = ap.parse_args()
    df = collect(load_processed(), a.stride, a.windows, a.chronos)
    df.to_json(ARTIFACTS / "comparison.json", orient="records", indent=2)
    md = to_markdown(df)
    (ARTIFACTS / "comparison.md").write_text(md + "\n")
    print(md)
