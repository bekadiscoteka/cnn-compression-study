"""Collects ST Edge AI benchmark logs into results/hardware.csv.

Expects one folder per model in results/benchmarks/, named like mbv2_p25_ptq_s1, holding
the logs downloaded from ST Edge AI Developer Cloud (file names may vary; the
script looks for the generate, validation and flashing logs by keyword).
Checks that all runs share board, clock, tool version and settings, and that
nothing was placed in external memory.

Usage: python parse_benchmarks.py
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ["model", "pruning", "quant", "board", "clock_mhz", "latency_ms", "cycles", "flash_kb", "ram_kb",
          "weights_kb", "activations_kb", "optimization", "external_memory", "st_edge_ai", "board_vs_onnx_cos"]


def find(folder, *keys):
    for f in sorted(folder.iterdir()):
        if any(k in f.name.lower() for k in keys):
            return f.read_text(errors="replace")
    raise FileNotFoundError(f"{folder.name}: no log matching {keys}")


def grab(pattern, text, cast=str):
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"pattern not found: {pattern}")
    return cast(m.group(1).replace(",", ""))


def parse(folder):
    gen, val, flash = find(folder, "generat"), find(folder, "valid"), find(folder, "flash")
    footprint = json.loads(re.search(r"Memory footprint: (\{.*?\n\})", gen, re.S).group(1))
    name = folder.name
    m = re.match(r"mbv2_p(\d+)_(\w+)_s\d+", name)
    pools = re.findall(r"^\s*(EXTERNAL\S*)\s+(\S+)", gen, re.M)
    external = any(state != "unused" for _, state in pools) or "external_flash" in gen.lower() and \
        re.search(r'"external_flash":\s*[1-9]', gen) is not None
    cycles = grab(r"CPU cycles\s*:\s*\[\s*([\d,]+)", val, int)
    latency = grab(r"duration\s*:\s*([\d.]+) ms", val, float)
    return {
        "model": name, "pruning": int(m.group(1)) / 100, "quant": m.group(2),
        "board": grab(r"Board\s*:\s*(\S+)", flash),
        "clock_mhz": round(cycles / latency / 1000),
        "latency_ms": latency, "cycles": cycles,
        "flash_kb": round((footprint["weights"] + footprint["kernel_flash"]) / 1024, 1),
        "ram_kb": round((footprint["activations"] + footprint["kernel_ram"]) / 1024, 1),
        "weights_kb": round(footprint["weights"] / 1024, 1),
        "activations_kb": round(footprint["activations"] / 1024, 1),
        "optimization": grab(r"optimization\s*:\s*(\w+)", gen),
        "external_memory": "yes" if external else "no",
        "st_edge_ai": grab(r"ST Edge AI Core v([\d.]+)", gen),
        "board_vs_onnx_cos": grab(r"cos=([\d.]+)", val),
    }


def main():
    rows = [parse(d) for d in sorted((HERE / "results" / "benchmarks").iterdir()) if d.is_dir()]
    with open(HERE / "results" / "hardware.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']:18s} {r['latency_ms']:8.2f} ms  flash {r['flash_kb']:7.1f} KB  ram {r['ram_kb']:6.1f} KB"
              f"  {r['optimization']:8s} ext={r['external_memory']}  cos={r['board_vs_onnx_cos']}")
    for key in ["board", "clock_mhz", "st_edge_ai", "optimization", "external_memory"]:
        values = {r[key] for r in rows}
        if len(values) > 1:
            print(f"WARNING: runs differ in {key}: " + ", ".join(f"{r['model']}={r[key]}" for r in rows))
    print(f"{len(rows)} models written to results/hardware.csv")


if __name__ == "__main__":
    main()
