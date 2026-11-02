"""Shared fixtures. Tests never need the real (non-redistributable) data: they use a
synthetic generator with the same schema, and a tiny model trained on it."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.forecasting import ModelStore  # noqa: E402
from app.uncertainty import QUANTILES  # noqa: E402
from models.chronos_model import ChronosForecaster  # noqa: E402
from models.preprocessing import CAT_VOCAB, HORIZON_BUCKETS, SCHEMA  # noqa: E402


def make_series(sid="series_X", start="2015-01-01", end="2017-03-01", seed=0, level=60.0,
                miss=0.02, unknown_cat=False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(idx)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.97 * ar[i - 1] + rng.normal(0, 0.12)
    daily = 0.25 * np.sin(2 * np.pi * (np.asarray(idx.hour) - 7) / 24)
    y = np.round(level * np.exp(ar + daily)).clip(3)
    df = pd.DataFrame({"timestamp": idx, "series_id": sid, "target": y})
    df["aux_1"] = y + rng.normal(0, 3, n)
    for j in range(2, 6):
        df[f"aux_{j}"] = np.abs(y * rng.uniform(0.2, 2) + rng.normal(0, 10, n))
    df["cov_1"] = 12 + 10 * np.sin(2 * np.pi * (np.asarray(idx.dayofyear) - 100) / 365) + rng.normal(0, 2, n)
    df["cov_2"] = 1010 + rng.normal(0, 5, n)
    df["cov_3"] = df["cov_1"] - 8 + rng.normal(0, 1, n)
    df["cov_4"] = np.where(rng.random(n) < 0.04, rng.exponential(1.0, n), 0.0)
    df["cov_5"] = np.abs(rng.normal(1.8, 1.0, n))
    df["cov_cat"] = rng.choice(CAT_VOCAB, n)
    if unknown_cat:
        df.loc[df.sample(frac=0.01, random_state=1).index, "cov_cat"] = "XYZ"
    for c in [c for c in SCHEMA if c not in ("timestamp", "series_id", "cov_cat")]:
        df.loc[rng.random(n) < miss, c] = np.nan
    return df[SCHEMA]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


class StubPipeline:
    """Mimics Chronos2Pipeline.predict_df: persistence +- spread. Tests our parsing contract only."""

    def predict_df(self, df, prediction_length, quantile_levels, id_column, timestamp_column, target):
        rows = []
        for i, g in df.groupby(id_column, sort=False):
            g = g.sort_values(timestamp_column)
            last = float(g[target].dropna().iloc[-1])
            ts = pd.date_range(g[timestamp_column].iloc[-1], periods=prediction_length + 1, freq="h")[1:]
            z = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
            d = pd.DataFrame({id_column: i, timestamp_column: ts, "target_name": target, "predictions": last})
            for q in quantile_levels:
                d[str(q)] = last + z[q] * 0.2 * last
            rows.append(d)
        return pd.concat(rows, ignore_index=True)


class BrokenPipeline:
    def predict_df(self, *a, **k):
        from models.chronos_model import ModelUnavailable
        raise ModelUnavailable("weights not found")


def _zero_cal() -> dict:
    off = {str(l): {f"{a}-{b}": 0.0 for a, b in HORIZON_BUCKETS} for l in (50, 90)}
    return {"n": 0, "offsets": off, "window": ["synthetic", "synthetic"]}


def build_store(tmp: Path, chronos_pipeline=None) -> ModelStore:
    """Train a *tiny* global model (both variants) on synthetic series into `tmp`."""
    import json
    from training.calibrate import save_cal
    from training.train_global import train_variant

    frames = {}
    for k, sid in enumerate(["series_a", "series_b"]):
        d = make_series(sid, seed=k, level=40 + 30 * k)
        frames[sid] = d.set_index("timestamp").drop(columns="series_id")
    for exog, name in [(True, "global_exog"), (False, "global_noexog")]:
        model, cal = train_variant(frames, exog, stride_h=24, params={"max_iter": 10, "min_samples_leaf": 50},
                                   val_stride_h=48, quiet=True)
        model.save(tmp / name, {"train_end": "2016-01-01"})
        save_cal(cal, tmp / name)
    for name in ("chronos_exog", "chronos_noexog"):
        (tmp / name).mkdir(parents=True, exist_ok=True)
        (tmp / name / "calibration.json").write_text(json.dumps(_zero_cal()))
        (tmp / name / "meta.json").write_text(json.dumps({"kind": "stub"}))
    fc = ChronosForecaster(pipeline=chronos_pipeline) if chronos_pipeline is not None else ChronosForecaster()
    return ModelStore(tmp, chronos=fc)


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("artifacts"), StubPipeline())


@pytest.fixture(scope="session")
def series_df():
    return make_series("series_new", seed=11, level=90.0, start="2016-06-01", end="2016-09-01")


@pytest.fixture(scope="session")
def csv_bytes(series_df):
    return to_csv_bytes(series_df)
