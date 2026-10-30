"""Chooses the activation calibration method for int8 quantization.

Quantizes every trained fp32 model with each candidate method and measures the
accuracy lost against the fp32 model on a validation split: 5,000 training
images that are not among the calibration images. The test set is not used,
so the choice does not leak into the reported results.

Decision rule (fixed before running): the method with the smallest mean
accuracy drop over all models wins; on a tie within 0.1 pp the simpler method
(earlier in CANDIDATES) wins. The choice is written to calibration_choice.json,
which train.py reads when config.yaml has calibration.method: auto.

Usage: python calibrate_check.py --out /content/drive/MyDrive/cnn-study
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

import train as T

CANDIDATES = [{"method": "minmax"},
              {"method": "percentile", "percentile": 99.999},
              {"method": "percentile", "percentile": 99.99},
              {"method": "percentile", "percentile": 99.9}]
FIELDS = ["model", "pruning", "seed", "calibration", "fp32_val_acc", "int8_val_acc", "drop_pp"]


def label(c):
    return c["method"] if c["method"] == "minmax" else f"percentile {c['percentile']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(T.HERE / "config.yaml"))
    out = Path("smoke_out" if args.smoke else args.out)
    n_calib, n_val = (32, 64) if args.smoke else (cfg["calibration_images"], 5000)

    _, calib_ds, _ = T.get_data(args.smoke, T.HERE / "data", out)
    calib_idx = set(T.calibration_indices(len(calib_ds), n_calib).tolist())
    rest = np.array([i for i in range(len(calib_ds)) if i not in calib_idx])
    val_idx = np.random.default_rng(1).choice(rest, min(n_val, len(rest)), replace=False)
    val = [calib_ds[int(i)] for i in val_idx]

    csv_path, tmp = out / "calibration_check.csv", out / "models" / "calib_tmp.onnx"
    done = set()
    if csv_path.exists():
        done = {(r["model"], r["calibration"]) for r in csv.DictReader(open(csv_path))}
    else:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    for seed in cfg["seeds"]:
        for p in cfg["pruning_ratios"]:
            name = f"{T.tag(p)}_fp32_s{seed}"
            fp32 = out / "models" / f"{name}.onnx"
            if not fp32.exists():
                print(f"[missing] {fp32.name}")
                continue
            todo = [c for c in CANDIDATES if (name, label(c)) not in done]
            if not todo:
                continue
            fp32_acc = T.eval_onnx(fp32, val)
            for c in todo:
                T.quantize_int8(fp32, tmp, calib_ds, n_calib, c)
                acc = T.eval_onnx(tmp, val)
                row = {"model": name, "pruning": p, "seed": seed, "calibration": label(c),
                       "fp32_val_acc": f"{fp32_acc:.2f}", "int8_val_acc": f"{acc:.2f}",
                       "drop_pp": f"{fp32_acc - acc:.2f}"}
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                print(f"{name:18s} {label(c):18s} fp32 {fp32_acc:6.2f}  int8 {acc:6.2f}  drop {fp32_acc - acc:6.2f}",
                      flush=True)
    tmp.unlink(missing_ok=True)

    rows = list(csv.DictReader(open(csv_path)))
    print("\nMean accuracy drop (pp) on the validation split, by pruning ratio:")
    table = {}
    for c in CANDIDATES:
        drops = {}
        for r in rows:
            if r["calibration"] == label(c):
                drops.setdefault(float(r["pruning"]), []).append(float(r["drop_pp"]))
        if drops:
            table[label(c)] = (np.mean([d for v in drops.values() for d in v]), drops)
            per_ratio = "  ".join(f"p{round(k * 100):02d} {np.mean(v):6.2f}" for k, v in sorted(drops.items()))
            print(f"  {label(c):18s} overall {table[label(c)][0]:6.2f}   {per_ratio}")

    best = None
    for c in CANDIDATES:  # earlier = simpler; a later method must beat it by more than 0.1 pp
        if label(c) in table and (best is None or table[label(c)][0] < table[label(best)][0] - 0.1):
            best = c
    (out / "calibration_choice.json").write_text(json.dumps(best))
    print(f"\nChosen: {label(best)}  -> {out / 'calibration_choice.json'}")


if __name__ == "__main__":
    main()
