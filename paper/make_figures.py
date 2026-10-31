"""Column-width vector figures for the paper (IEEE two-column, 3.5 in wide).

Uses the same data and colors as analysis.py. Run from the repository root:
python paper/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
W = 3.5  # IEEE column width, inches
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
SHORT = {"fp32": "fp32", "ptq": "int8 PTQ", "qat": "int8 QAT-5", "qat15": "int8 QAT-15"}


def axes_style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=A.INK)
    ax.set_ylabel(ylabel, color=A.INK)
    ax.grid(True, color=A.GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#9a9994")
    ax.tick_params(colors=A.INK, width=0.6, length=2.5)


def pareto(s):
    fig, ax = plt.subplots(figsize=(W, 2.45))
    front = s[s.pareto_main].sort_values("energy_mj")
    ax.plot(front.energy_mj, front.acc_mean, color=A.INK2, linewidth=0.9, linestyle="--", zorder=1,
            label="Pareto front")
    nudge = {"fp32": 1.0, "ptq": 0.95, "qat": 1.0, "qat15": 1.05}
    for q in A.MAIN + A.SENS:
        sub = s[s.quant == q]
        ax.errorbar(sub.energy_mj * nudge[q], sub.acc_mean, yerr=sub.acc_std, fmt=A.MARKER[q], color=A.COLOR[q],
                    ms=4.2, mfc=A.COLOR[q] if q in A.MAIN else "white", mew=1.1, elinewidth=0.8, capsize=1.8,
                    label=SHORT[q], zorder=3)
    for _, r in s[s.quant == "fp32"].iterrows():
        ax.annotate(f"{r.pruning:.0%}", (r.energy_mj, r.acc_mean), textcoords="offset points", xytext=(-6, -10),
                    fontsize=6.5, color=A.INK2)
    for _, r in s[s.quant == "qat"].iterrows():
        ax.annotate(f"{r.pruning:.0%}", (r.energy_mj, r.acc_mean), textcoords="offset points", xytext=(7, -9),
                    fontsize=6.5, color=A.INK2)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1.8, 65)
    axes_style(ax, "Energy per inference (mJ, log scale)", "Top-1 accuracy (%)")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4, borderaxespad=0.2)
    fig.savefig(OUT / "pareto.pdf")
    plt.close(fig)


def seeds(df):
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ratios = sorted(df.pruning.unique())
    for k, q in enumerate(A.MAIN + A.SENS):
        sub = df[df.quant == q]
        x = np.array([ratios.index(p) for p in sub.pruning]) + (k - 1.5) * 0.16
        ax.scatter(x, sub.accuracy, s=13, marker=A.MARKER[q], color=A.COLOR[q],
                   facecolor=A.COLOR[q] if q in A.MAIN else "white", linewidth=0.9, label=SHORT[q], zorder=3)
    ax.set_xticks(range(len(ratios)), [f"{p:.0%}" for p in ratios])
    axes_style(ax, "Channels pruned", "Top-1 accuracy (%)")
    ax.legend(frameon=False, loc="lower left", ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.savefig(OUT / "seeds.pdf")
    plt.close(fig)


def qat_gain(df):
    fig, ax = plt.subplots(figsize=(W, 1.95))
    ratios = sorted(df.pruning.unique())
    ax.axhline(0, color=A.INK2, linewidth=0.7)
    for k, q in enumerate(["qat", "qat15"]):
        d = A.paired(df, q, "ptq")
        off = (k - 0.5) * 0.26
        x = np.array([ratios.index(p) for p in d.pruning]) + off
        ax.scatter(x, d["diff"], s=11, marker=A.MARKER[q], color=A.COLOR[q], alpha=0.6, linewidth=0, zorder=2)
        m = d.groupby("pruning")["diff"].mean()
        ax.scatter(np.arange(len(ratios)) + off, m.values, s=70, marker="_", color=A.COLOR[q], linewidth=1.8,
                   zorder=3, label=f"{SHORT[q]} $-$ PTQ")
    ax.set_xticks(range(len(ratios)), [f"{p:.0%}" for p in ratios])
    axes_style(ax, "Channels pruned", "Accuracy gain (pp)")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2)
    fig.savefig(OUT / "qat_gain.pdf")
    plt.close(fig)


def macs(s):
    fig, ax = plt.subplots(figsize=(W, 1.95))
    for q in ["fp32", "ptq"]:
        sub = s[s.quant == q].sort_values("macs")
        ax.plot(sub.macs / 1e6, sub.latency_ms, marker=A.MARKER[q], color=A.COLOR[q], ms=4, linewidth=1.1,
                label="fp32" if q == "fp32" else "int8 (PTQ, QAT)")
        for _, r in sub.iterrows():
            ax.annotate(f"{r.pruning:.0%}", (r.macs / 1e6, r.latency_ms), textcoords="offset points",
                        xytext=(-16, 3) if q == "fp32" else (2, -10), fontsize=6.5, color=A.INK2)
    axes_style(ax, "MACs per inference (millions)", "Latency (ms)")
    ax.set_xlim(0, 19.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "macs_latency.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df, _ = A.load()
    s, _ = A.summarize(df)
    pareto(s)
    seeds(df)
    qat_gain(df)
    macs(s)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
