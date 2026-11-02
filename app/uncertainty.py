"""Prediction intervals: raw model quantiles + split-conformal (CQR) correction.

Both models emit raw quantiles at 0.05 / 0.25 / 0.5 / 0.75 / 0.95. The raw
quantiles are only as calibrated as the model's own quantile loss / pre-training,
so each is corrected on a held-out *validation* window with CQR
(Romano et al., 2019):

    score_i = max(lo_i - y_i, y_i - hi_i)          (in target/scale units)
    q_hat   = ceil((n+1)(1-alpha))/n quantile of scores, per horizon bucket
    interval = [lo - q_hat, hi + q_hat]

q_hat can be negative (intervals shrink when the raw band is too wide).
Scores are computed on values divided by the origin's trailing scale so one
set of offsets transfers across series of different magnitude (and to unseen
series). Coverage is then *verified* on the untouched test window.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from models.preprocessing import HORIZON_BUCKETS, bucket_label

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# interval width (%) -> (lower quantile idx, upper quantile idx) in QUANTILES
LEVELS = {50: (1, 3), 90: (0, 4)}


def _bucket_array(h: np.ndarray) -> np.ndarray:
    return np.array([bucket_label(int(x)) for x in h])


def fit_cqr(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, h: np.ndarray, level: int) -> dict:
    """Per-horizon-bucket conformal offsets. Inputs are in normalised units."""
    alpha = 1 - level / 100
    scores = np.maximum(lo - y, y - hi)
    labels = _bucket_array(h)
    out: dict[str, float] = {}
    for a, b in HORIZON_BUCKETS:
        lab = f"{a}-{b}"
        s = scores[labels == lab]
        s = s[np.isfinite(s)]
        if len(s) < 30:
            out[lab] = 0.0
            continue
        k = min(math.ceil((len(s) + 1) * (1 - alpha)), len(s))
        out[lab] = float(np.sort(s)[k - 1])
    return out


def apply_cqr(lo: np.ndarray, hi: np.ndarray, h: np.ndarray, offsets: dict) -> tuple[np.ndarray, np.ndarray]:
    q = np.array([offsets.get(l, 0.0) for l in _bucket_array(h)])
    return lo - q, hi + q


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) if m.any() else float("nan")


def fit_calibration(y: np.ndarray, q: np.ndarray, h: np.ndarray) -> dict:
    """y: (n,) normalised truth; q: (n, 5) normalised quantile predictions."""
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    y, q, h = y[ok], q[ok], h[ok]
    cal = {"n": int(len(y)), "offsets": {}, "raw_val_coverage": {}, "cal_val_coverage": {}}
    for level, (i, j) in LEVELS.items():
        off = fit_cqr(y, q[:, i], q[:, j], h, level)
        lo, hi = apply_cqr(q[:, i], q[:, j], h, off)
        cal["offsets"][str(level)] = off
        cal["raw_val_coverage"][str(level)] = coverage(y, q[:, i], q[:, j])
        cal["cal_val_coverage"][str(level)] = coverage(y, lo, hi)
    return cal


def calibrated_interval(q: np.ndarray, h: np.ndarray, cal: dict, level: int) -> tuple[np.ndarray, np.ndarray]:
    i, j = LEVELS[level]
    lo, hi = apply_cqr(q[:, i], q[:, j], h, cal["offsets"][str(level)])
    return lo, hi


def save_calibration(cal: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cal, indent=2, sort_keys=True))


def load_calibration(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
