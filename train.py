"""One cell of the experiment matrix.

Trains (or loads) the fp32 baseline for a seed, prunes and fine-tunes it,
exports it to ONNX, quantizes it to int8 (PTQ), measures accuracy on the
exported files and appends one row per model to results.csv.

Usage:
  python train.py --pruning 0.5 --seed 1 --out /content/drive/MyDrive/cnn-study
  python train.py --smoke        # tiny fake data: checks the pipeline only
"""
import argparse
import csv
import datetime
import os
import random
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision
import torchvision.transforms as T
import yaml
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

HERE = Path(__file__).resolve().parent
warnings.filterwarnings("ignore", category=DeprecationWarning)  # legacy ONNX exporter notice
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
FIELDS = ["run_id", "date", "commit", "pruning", "quant", "seed", "accuracy", "params", "macs",
          "latency_ms", "flash_kb", "ram_kb", "energy_mj", "notes"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
        return run("rev-parse", "--short", "HEAD") + ("-dirty" if run("status", "--porcelain", "--untracked-files=no") else "")
    except OSError:
        return "unknown"


def build_model(width):
    model = torchvision.models.mobilenet_v2(width_mult=width, num_classes=10)
    # CIFAR-10 images are 32x32, not 224x224: remove two early downsamplings
    # so the last feature map is 4x4 instead of 1x1.
    model.features[0][0].stride = (1, 1)
    model.features[2].conv[1][0].stride = (1, 1)
    return model


def get_data(smoke, root):
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    if smoke:
        fake = lambda n, tf, off: torchvision.datasets.FakeData(n, (3, 32, 32), 10, tf, random_offset=off)
        return fake(256, train_tf, 0), fake(256, test_tf, 0), fake(64, test_tf, 10_000)
    train = torchvision.datasets.CIFAR10(root, train=True, transform=train_tf, download=True)
    calib = torchvision.datasets.CIFAR10(root, train=True, transform=test_tf)  # no augmentation
    test = torchvision.datasets.CIFAR10(root, train=False, transform=test_tf, download=True)
    return train, calib, test


def train(model, loader, epochs, lr, wd, device, ckpt=None):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    start = 0
    if ckpt and ckpt.exists():  # resume after a Colab disconnect
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"]); start = state["epoch"]
        print(f"Resumed from epoch {start}")
    for ep in range(start, epochs):
        model.train()
        t, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
        print(f"  epoch {ep + 1}/{epochs}  loss {total / len(loader):.3f}  ({time.time() - t:.0f}s)", flush=True)
        if ckpt and ((ep + 1) % 5 == 0 or ep + 1 == epochs):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep + 1}, ckpt)


@torch.no_grad()
def eval_torch(model, ds, device):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=256)
    correct = sum((model(x.to(device)).argmax(1).cpu() == y).sum().item() for x, y in loader)
    return 100 * correct / len(ds)


def eval_onnx(path, ds):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    correct = sum(int(sess.run(None, {"input": x.unsqueeze(0).numpy()})[0].argmax() == y) for x, y in ds)
    return 100 * correct / len(ds)


def prune(model, ratio, device):
    # Remove the same share of channels from every layer, smallest L1 norm first.
    example = torch.randn(1, 3, 32, 32, device=device)
    pruner = tp.pruner.MagnitudePruner(
        model, example, importance=tp.importance.MagnitudeImportance(p=1),
        pruning_ratio=ratio, ignored_layers=[model.classifier[1]])
    pruner.step()


def export_onnx(model, path):
    model = model.eval().cpu()
    torch.onnx.export(model, (torch.randn(1, 3, 32, 32),), str(path), input_names=["input"],
                      output_names=["logits"], opset_version=13, dynamo=False)


class Calibration(CalibrationDataReader):
    def __init__(self, ds, idx):
        self.items = iter([{"input": ds[int(i)][0].unsqueeze(0).numpy()} for i in idx])

    def get_next(self):
        return next(self.items, None)


def quantize_ptq(fp32_path, int8_path, calib_ds, n):
    idx = np.random.default_rng(0).choice(len(calib_ds), n, replace=False)  # same images for every run
    pre = fp32_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    quantize_static(str(pre), str(int8_path), Calibration(calib_ds, idx), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    pre.unlink()


def append_rows(csv_path, rows):
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruning", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="out")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    epochs, ft_epochs, n_calib = cfg["train"]["epochs"], cfg["finetune_epochs"], cfg["calibration_images"]
    if args.smoke:
        epochs, ft_epochs, n_calib, args.out = 1, 1, 32, "smoke_out"

    out = Path(args.out); models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p, seed = args.pruning, args.seed
    tag = f"mbv2_p{round(p * 100):02d}"
    print(f"Run {tag} seed {seed} on {device}, commit {git_commit()}")

    set_seed(seed)
    train_ds, calib_ds, test_ds = get_data(args.smoke, HERE / "data")  # local disk: much faster than Drive
    loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                                         num_workers=2, pin_memory=device == "cuda", drop_last=True)

    # 1. fp32 baseline, trained once per seed and reused by every pruning ratio
    model = build_model(cfg["width_mult"]).to(device)
    base = out / f"baseline_s{seed}.pt"
    if base.exists():
        model.load_state_dict(torch.load(base, map_location=device))
        print("Loaded baseline", base)
    else:
        print("Training baseline")
        train(model, loader, epochs, cfg["train"]["lr"], cfg["train"]["weight_decay"], device,
              ckpt=out / f"baseline_s{seed}_partial.pt")
        torch.save(model.state_dict(), base)

    # 2. structured pruning + fine-tuning
    if p > 0:
        set_seed(seed)
        prune(model, p, device)
        print(f"Fine-tuning pruned model ({p:.0%} of channels removed)")
        train(model, loader, ft_epochs, cfg["finetune_lr"], cfg["train"]["weight_decay"], device)
    torch_acc = eval_torch(model, test_ds, device)
    macs, params = tp.utils.count_ops_and_params(model, torch.randn(1, 3, 32, 32, device=device))
    torch.save(model, models / f"{tag}_fp32_s{seed}.pt")  # full model: needed later for QAT

    # 3. export fp32 and int8 (PTQ), measure accuracy on the exported files
    fp32_path = models / f"{tag}_fp32_s{seed}.onnx"
    ptq_path = models / f"{tag}_ptq_s{seed}.onnx"
    export_onnx(model, fp32_path)
    quantize_ptq(fp32_path, ptq_path, calib_ds, n_calib)

    common = {"date": datetime.date.today().isoformat(), "commit": git_commit(), "pruning": p,
              "seed": seed, "params": int(params), "macs": int(macs)}
    rows = []
    for quant, path in [("fp32", fp32_path), ("ptq", ptq_path)]:
        acc = eval_onnx(path, test_ds)
        note = f"torch_acc={torch_acc:.2f}" if quant == "fp32" else ""
        rows.append({**common, "run_id": f"{tag}_{quant}_s{seed}", "quant": quant,
                     "accuracy": f"{acc:.2f}", "notes": ("smoke " if args.smoke else "") + note})
        print(f"{quant}: accuracy {acc:.2f}%  file {path.stat().st_size / 1024:.0f} KB  -> {path}")
    append_rows(out / "results.csv", rows)
    print(f"params {params:,}  MACs {macs:,}  rows added to {out / 'results.csv'}")


if __name__ == "__main__":
    main()
