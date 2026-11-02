"""One-command pipeline:  python train.py --src /path/to/series_csvs [--chronos] [--loso]

  1. prepare data      (dedupe, schema check, manifest)
  2. train global      (exog + no-exog variants, conformal calibration on val)
  3. calibrate chronos (zero-shot; only with --chronos and weights available)
  4. evaluate          (identical test origins for every model) -> artifacts/results.md
"""
import argparse
import subprocess
import sys


def run(*args: str) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--chronos", action="store_true", help="also calibrate + evaluate Chronos-2")
    ap.add_argument("--loso", action="store_true", help="leave-one-series-out evaluation (slow)")
    a = ap.parse_args()
    run("-m", "training.prepare_data", "--src", a.src)
    run("-m", "training.train_global")
    if a.chronos:
        run("-m", "training.train_chronos")
    ev = ["-m", "training.evaluate"]
    if a.chronos:
        ev.append("--chronos")
    if a.loso:
        ev.append("--loso")
    run(*ev)
