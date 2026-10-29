"""Runs the experiment matrix: seeds x pruning ratios x {fp32, PTQ, QAT}.

For each seed the fp32 baseline is trained once. For each pruning ratio it is
pruned and fine-tuned, exported to ONNX, quantized to int8 with PTQ and with
QAT, and every exported file is evaluated. One row per model goes to
results.csv. Work that is already in results.csv is skipped, so the script
can be re-run after a disconnect and continues where it stopped.

Usage:
  python train.py --out /content/drive/MyDrive/cnn-study          # whole matrix
  python train.py --pruning 0.5 --seed 1 --out ...                  # one cell
  python train.py --smoke                                           # pipeline check on fake data
"""
import argparse
import copy
import csv
import datetime
import random
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch_pruning as tp
import torchvision
import torchvision.transforms as T
import yaml
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models.mobilenetv2 import InvertedResidual

HERE = Path(__file__).resolve().parent
warnings.filterwarnings("ignore", category=DeprecationWarning)  # legacy ONNX exporter notice
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")  # harmless with AMP
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
FIELDS = ["run_id", "date", "commit", "pruning", "quant", "seed", "accuracy", "params", "macs",
          "latency_ms", "flash_kb", "ram_kb", "energy_mj", "notes"]
QUANTS = ["fp32", "ptq", "qat"]


# ---------------------------------------------------------------- utilities

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


def tag(p):
    return f"mbv2_p{round(p * 100):02d}"


def run_id(p, quant, seed):
    return f"{tag(p)}_{quant}_s{seed}"


def done_runs(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def append_rows(csv_path, rows):
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        if new:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- model and data

def build_model(width):
    model = torchvision.models.mobilenet_v2(width_mult=width, num_classes=10)
    # CIFAR-10 images are 32x32, not 224x224: remove two early downsamplings
    # so the last feature map is 4x4 instead of 1x1.
    model.features[0][0].stride = (1, 1)
    model.features[2].conv[1][0].stride = (1, 1)
    return model


def get_data(smoke, root, cache_dir):
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    if smoke:
        fake = lambda n, tf, off: torchvision.datasets.FakeData(n, (3, 32, 32), 10, tf, random_offset=off)
        return fake(256, train_tf, 0), fake(256, test_tf, 0), fake(64, test_tf, 10_000)
    # The download is slow in Colab, so the archive is kept on Drive and copied back in later sessions.
    archive, cached = root / "cifar-10-python.tar.gz", cache_dir / "cifar-10-python.tar.gz"
    root.mkdir(parents=True, exist_ok=True)
    if cached.exists() and not archive.exists():
        shutil.copy(cached, archive)
    train = torchvision.datasets.CIFAR10(root, train=True, transform=train_tf, download=True)
    calib = torchvision.datasets.CIFAR10(root, train=True, transform=test_tf)  # no augmentation
    test = torchvision.datasets.CIFAR10(root, train=False, transform=test_tf, download=True)
    if archive.exists() and not cached.exists():
        shutil.copy(archive, cached)
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
        print(f"  resumed from epoch {start}")
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


# ---------------------------------------------------------------- compression

def prune(model, ratio, device):
    # Remove the same share of channels from every layer, smallest L1 norm first.
    example = torch.randn(1, 3, 32, 32, device=device)
    pruner = tp.pruner.MagnitudePruner(
        model, example, importance=tp.importance.MagnitudeImportance(p=1),
        pruning_ratio=ratio, ignored_layers=[model.classifier[1]])
    pruner.step()


def fold_bn(model):
    """Fold every BatchNorm into the preceding conv, as the ONNX export does."""
    model.eval()
    for seq in [m for m in model.modules() if isinstance(m, nn.Sequential)]:
        kids = list(seq.children())
        for i in range(len(kids) - 1):
            if isinstance(kids[i], nn.Conv2d) and isinstance(kids[i + 1], nn.BatchNorm2d):
                seq[i] = fuse_conv_bn_eval(kids[i], kids[i + 1])
                seq[i + 1] = nn.Identity()
    return model


class WeightFakeQuant(nn.Module):
    """int8 per-channel symmetric rounding of weights, gradient passes straight through."""
    def forward(self, w):
        s = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp(min=1e-8) / 127
        return w + (torch.clamp(torch.round(w / s), -127, 127) * s - w).detach()


def act_fake_quant_hook(state):
    """int8 per-tensor asymmetric rounding of activations, range tracked as a moving average."""
    def hook(module, inputs, out):
        if module.training:
            lo, hi = out.detach().min().clamp(max=0), out.detach().max().clamp(min=0)
            if state["lo"] is None:
                state["lo"], state["hi"] = lo, hi
            else:
                state["lo"], state["hi"] = 0.99 * state["lo"] + 0.01 * lo, 0.99 * state["hi"] + 0.01 * hi
        if state["lo"] is None:
            return out
        s = ((state["hi"] - state["lo"]) / 255).clamp(min=1e-8)
        zp = torch.round(-128 - state["lo"] / s)
        q = (torch.clamp(torch.round(out / s) + zp, -128, 127) - zp) * s
        return out + (q - out).detach()
    return hook


def qat(model, loader, epochs, lr, wd, device):
    """Quantization-aware fine-tuning on the BN-folded model.

    Weights and the activations that the int8 ONNX model quantizes (after ReLU6,
    after each block's projection conv and after each residual add) are rounded
    to int8 in the forward pass. Afterwards the hooks are removed and the weights
    are left on the int8 grid, so the model goes through the same ONNX quantizer
    as PTQ and gets an identical graph.
    """
    fold_bn(model)
    layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for m in layers:
        parametrize.register_parametrization(m, "weight", WeightFakeQuant())
    act_points = [m for m in model.modules() if isinstance(m, (nn.ReLU6, InvertedResidual))]
    act_points += [blk.conv[-2] for blk in model.modules() if isinstance(blk, InvertedResidual)]
    handles = [m.register_forward_hook(act_fake_quant_hook({"lo": None, "hi": None})) for m in act_points]
    train(model, loader, epochs, lr, wd, device)
    for h in handles:
        h.remove()
    for m in layers:
        parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    return model.eval()


def export_onnx(model, path):
    model = copy.deepcopy(model).eval().cpu()
    torch.onnx.export(model, (torch.randn(1, 3, 32, 32),), str(path), input_names=["input"],
                      output_names=["logits"], opset_version=13, dynamo=False)


class Calibration(CalibrationDataReader):
    def __init__(self, ds, idx):
        self.items = iter([{"input": ds[int(i)][0].unsqueeze(0).numpy()} for i in idx])

    def get_next(self):
        return next(self.items, None)


def quantize_int8(fp32_path, int8_path, calib_ds, n):
    idx = np.random.default_rng(0).choice(len(calib_ds), n, replace=False)  # same images for every run
    pre = int8_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    quantize_static(str(pre), str(int8_path), Calibration(calib_ds, idx), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    pre.unlink()


# ---------------------------------------------------------------- one cell of the matrix

def run_cell(p, seed, cfg, data, out, device, smoke):
    csv_path, models = out / "results.csv", out / "models"
    todo = [q for q in QUANTS if run_id(p, q, seed) not in done_runs(csv_path)]
    if not todo:
        print(f"[skip] {tag(p)} seed {seed}: already in results.csv")
        return
    print(f"\n=== {tag(p)} seed {seed}: {', '.join(todo)} ===", flush=True)
    t0 = time.time()
    train_ds, calib_ds, test_ds = data
    loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                                         num_workers=2, pin_memory=device == "cuda", drop_last=True)
    wd = cfg["train"]["weight_decay"]

    # fp32 model: baseline trained once per seed, then pruned and fine-tuned
    fp32_pt = models / f"{tag(p)}_fp32_s{seed}.pt"
    if fp32_pt.exists():
        model = torch.load(fp32_pt, map_location=device, weights_only=False)
        print("  loaded", fp32_pt.name)
    else:
        model = build_model(cfg["width_mult"]).to(device)
        base = out / f"baseline_s{seed}.pt"
        if base.exists():
            model.load_state_dict(torch.load(base, map_location=device))
            print("  loaded", base.name)
        else:
            print("  training baseline")
            set_seed(seed)
            partial = out / f"baseline_s{seed}_partial.pt"
            train(model, loader, cfg["train"]["epochs"], cfg["train"]["lr"], wd, device, ckpt=partial)
            torch.save(model.state_dict(), base)
            partial.unlink(missing_ok=True)
        if p > 0:
            set_seed(seed)
            prune(model, p, device)
            print(f"  fine-tuning after pruning {p:.0%} of channels")
            train(model, loader, cfg["finetune_epochs"], cfg["finetune_lr"], wd, device)
        torch.save(model, fp32_pt)

    torch_acc = eval_torch(model, test_ds, device)
    macs, params = tp.utils.count_ops_and_params(model, torch.randn(1, 3, 32, 32, device=device))
    paths = {q: models / f"{tag(p)}_{q}_s{seed}.onnx" for q in QUANTS}

    if not paths["fp32"].exists():
        export_onnx(model, paths["fp32"])
    if "ptq" in todo:
        quantize_int8(paths["fp32"], paths["ptq"], calib_ds, cfg["calibration_images"])
    if "qat" in todo:
        print("  quantization-aware fine-tuning")
        set_seed(seed)
        qat_model = qat(copy.deepcopy(model), loader, cfg["qat_epochs"], cfg["qat_lr"], wd, device)
        qat_fp32 = models / f"{tag(p)}_qatfp32_s{seed}.onnx"
        export_onnx(qat_model, qat_fp32)
        quantize_int8(qat_fp32, paths["qat"], calib_ds, cfg["calibration_images"])
        qat_fp32.unlink()

    common = {"date": datetime.date.today().isoformat(), "commit": git_commit(), "pruning": p,
              "seed": seed, "params": int(params), "macs": int(macs)}
    rows = []
    for q in todo:
        acc = eval_onnx(paths[q], test_ds)
        note = f"torch_acc={torch_acc:.2f}" if q == "fp32" else ""
        rows.append({**common, "run_id": run_id(p, q, seed), "quant": q, "accuracy": f"{acc:.2f}",
                     "notes": ("smoke " if smoke else "") + note})
        print(f"  {q:4s} accuracy {acc:6.2f}%   {paths[q].stat().st_size / 1024:5.0f} KB")
    append_rows(csv_path, rows)
    print(f"  params {params:,}  MACs {int(macs):,}  ({(time.time() - t0) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruning", type=float, help="run one pruning ratio only")
    ap.add_argument("--seed", type=int, help="run one seed only")
    ap.add_argument("--out", default="out")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    if args.smoke:
        cfg["train"]["epochs"] = cfg["finetune_epochs"] = cfg["qat_epochs"] = 1
        cfg["calibration_images"], args.out = 32, "smoke_out"
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    ratios = [args.pruning] if args.pruning is not None else cfg["pruning_ratios"]

    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device}, commit {git_commit()}, {len(seeds)} seed(s) x {len(ratios)} pruning ratio(s)")
    data = get_data(args.smoke, HERE / "data", out)  # data on local disk: much faster than Drive

    for seed in seeds:  # one seed at a time: a full picture appears after the first seed
        for p in ratios:
            run_cell(p, seed, cfg, data, out, device, args.smoke)

    done = done_runs(out / "results.csv")
    expected = {run_id(p, q, s) for s in cfg["seeds"] for p in cfg["pruning_ratios"] for q in QUANTS}
    onnx = len(list((out / "models").glob("*_s[0-9].onnx")))
    print(f"\nProgress: {len(done & expected)}/{len(expected)} rows in results.csv, {onnx} ONNX models")
    missing = sorted(expected - done)
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
