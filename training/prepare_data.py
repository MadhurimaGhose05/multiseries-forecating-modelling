"""Prepare training data from the shared folder.

    python -m training.prepare_data --src /path/to/onedrive/folder

* reads every *.csv, checks the schema, sorts, de-duplicates
* drops byte-identical copies (e.g. series_A_1.csv == series_A.csv)
* refuses two *different* files claiming the same series_id
* writes data/processed/all_series.csv.gz and artifacts/data_manifest.json

The raw data is never copied into git (see data/README.md and .gitignore).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models.preprocessing import NUMERIC, SCHEMA, to_hourly

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "all_series.csv.gz"
MANIFEST = ROOT / "artifacts" / "data_manifest.json"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df = df[SCHEMA].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["series_id", "timestamp"]).drop_duplicates(["series_id", "timestamp"])


def prepare(src: Path) -> pd.DataFrame:
    files = sorted(src.glob("*.csv"))
    if not files:
        sys.exit(f"No CSV files found in {src}")
    seen_hash: dict[str, str] = {}
    by_series: dict[str, tuple[str, pd.DataFrame]] = {}
    manifest = {"files": [], "dropped_duplicates": [], "series": []}
    for f in files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h in seen_hash:
            manifest["dropped_duplicates"].append({"file": f.name, "identical_to": seen_hash[h]})
            print(f"  skip {f.name}: byte-identical to {seen_hash[h]}")
            continue
        seen_hash[h] = f.name
        df = load_csv(f)
        for sid, g in df.groupby("series_id"):
            if sid in by_series:
                raise ValueError(f"series_id {sid!r} appears in two different files: "
                                 f"{by_series[sid][0]} and {f.name}")
            by_series[sid] = (f.name, g)
        manifest["files"].append({"file": f.name, "sha256": h, "rows": len(df)})
    frames = []
    for sid, (fname, g) in sorted(by_series.items()):
        hourly = to_hourly(g)
        n_gap = len(hourly) - len(g)
        manifest["series"].append({"series_id": sid, "source": fname, "rows": len(hourly),
                                   "inserted_gap_rows": n_gap,
                                   "start": str(hourly.index.min()), "end": str(hourly.index.max()),
                                   "target_missing_pct": round(100 * hourly["target"].isna().mean(), 2)})
        hourly = hourly.assign(series_id=sid).reset_index()
        frames.append(hourly[SCHEMA])
    out = pd.concat(frames, ignore_index=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED, index=False, compression="gzip")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {PROCESSED} ({len(out):,} rows, {len(by_series)} series: {sorted(by_series)})")
    return out


def load_processed() -> dict[str, pd.DataFrame]:
    """-> {series_id: regular hourly frame indexed by timestamp}"""
    if not PROCESSED.exists():
        raise FileNotFoundError(f"{PROCESSED} not found. Run `python -m training.prepare_data --src <folder>`.")
    df = pd.read_csv(PROCESSED, parse_dates=["timestamp"])
    return {sid: g.set_index("timestamp").drop(columns="series_id") for sid, g in df.groupby("series_id")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "raw"), help="folder with the series CSVs")
    prepare(Path(ap.parse_args().src))
