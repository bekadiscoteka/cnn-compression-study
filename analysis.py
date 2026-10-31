"""Analysis: energy, Pareto fronts, selection rule and hypothesis tests.

Joins results/results_v2.csv (accuracy per model) with results/hardware.csv
(latency and memory per architecture, measured on seed 1) and writes
results/analysis/: summary_by_config.csv, selection.csv, hypotheses.md and
four figures.

Energy is an estimate: supply power from the STM32H7B3 datasheet figure
(120 uA/MHz typical, Run mode, peripherals off, SMPS, 3.3 V) at the 280 MHz
benchmark clock, times measured latency. It is proportional to latency, so it
ranks configurations exactly as latency does.

Usage: python analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "analysis"
CFG = yaml.safe_load(open(HERE / "config.yaml"))
VDD, UA_PER_MHZ = 3.3, 120.0
POWER_W = VDD * UA_PER_MHZ * 1e-6 * CFG["hardware"]["clock_mhz"]  # 0.111 W at 280 MHz
MAIN = ["fp32", "ptq", "qat"]            # pre-registered variants
SENS = ["qat15"]                         # sensitivity check
HW_SOURCE = {"fp32": "fp32", "ptq": "int8", "qat": "int8", "qat15": "int8"}  # all int8 variants share one graph

# Light theme from the dataviz reference palette (validated: scripts/validate_palette.js)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"fp32": "#2a78d6", "ptq": "#eb6834", "qat": "#1baf7a", "qat15": "#eda100"}
MARKER = {"fp32": "o", "ptq": "s", "qat": "^", "qat15": "D"}
LABEL = {"fp32": "fp32", "ptq": "int8 PTQ", "qat": "int8 QAT (5 ep.)", "qat15": "int8 QAT (15 ep.)"}


def load():
    acc = pd.read_csv(HERE / "results" / "results_v2.csv")
    hw = pd.read_csv(HERE / "results" / "hardware.csv")
    # PTQ and QAT models share one graph; their measured latencies differ only by noise (< 0.1 %),
    # so int8 uses the mean of the two measurements per pruning ratio.
    hw["hw_quant"] = np.where(hw["quant"] == "fp32", "fp32", "int8")
    per_arch = hw.groupby(["pruning", "hw_quant"]).agg(
        latency_ms=("latency_ms", "mean"), flash_kb=("flash_kb", "mean"), ram_kb=("ram_kb", "mean"),
        optimization=("optimization", lambda x: "/".join(sorted(set(x))))).reset_index()
    acc["hw_quant"] = acc["quant"].map(HW_SOURCE)
    df = acc.merge(per_arch, on=["pruning", "hw_quant"], how="left", validate="many_to_one")
    assert df["latency_ms"].notna().all(), "missing hardware row"
    df["energy_mj"] = POWER_W * df["latency_ms"]          # W x ms = mJ
    df["edp"] = df["energy_mj"] * df["latency_ms"]          # mJ x ms
    return df, hw


def summarize(df):
    base = df[(df.pruning == 0) & (df.quant == "fp32")]["accuracy"].mean()
    g = df.groupby(["pruning", "quant"], sort=False)
    s = g.agg(n=("accuracy", "size"), acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
              acc_min=("accuracy", "min"), acc_max=("accuracy", "max"), params=("params", "first"),
              macs=("macs", "first"), latency_ms=("latency_ms", "first"), energy_mj=("energy_mj", "first"),
              edp=("edp", "first"), flash_kb=("flash_kb", "first"), ram_kb=("ram_kb", "first"),
              optimization=("optimization", "first")).reset_index()
    s["drop_vs_baseline_pp"] = base - s["acc_mean"]
    s["technique"] = np.select([(s.pruning == 0) & (s.quant == "fp32"), s.pruning == 0, s.quant == "fp32"],
                               ["none", "quantization only", "pruning only"], "pruning + quantization")
    s["pareto_main"] = pareto(s, s.quant.isin(MAIN))
    s["pareto_with_qat15"] = pareto(s, s.quant.isin(MAIN + SENS))
    return s.round(3), base


def pareto(s, mask):
    """On the front: no other configuration has lower-or-equal energy and higher-or-equal
    mean accuracy, with at least one strictly better."""
    flags = []
    for i, r in s.iterrows():
        if not mask[i]:
            flags.append(False)
            continue
        others = s[mask & (s.index != i)]
        dominated = ((others.energy_mj <= r.energy_mj) & (others.acc_mean >= r.acc_mean) &
                     ((others.energy_mj < r.energy_mj) | (others.acc_mean > r.acc_mean))).any()
        flags.append(not dominated)
    return flags


def selection(s, base, variants):
    rows = []
    pool = s[s.quant.isin(variants)]
    for tol in [1, 2, 3, 5, 10, 15]:
        ok = pool[pool.acc_mean >= base - tol]
        best = ok.sort_values(["energy_mj", "acc_mean"], ascending=[True, False]).iloc[0]  # tie: more accurate
        rows.append({"max_drop_pp": tol, "pruning": best.pruning, "quant": best.quant,
                     "acc_mean": best.acc_mean, "drop_pp": round(base - best.acc_mean, 2),
                     "energy_mj": best.energy_mj, "latency_ms": best.latency_ms,
                     "energy_saving_x": round(s[(s.pruning == 0) & (s.quant == "fp32")].energy_mj.iloc[0]
                                              / best.energy_mj, 2)})
    return pd.DataFrame(rows)


def paired(df, a, b):
    """Per-model difference a - b (same pruning ratio and seed)."""
    w = df.pivot_table(index=["pruning", "seed"], columns="quant", values="accuracy")
    return (w[a] - w[b]).rename("diff").reset_index()


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def h2_block(df, a, b):
    d = paired(df, a, b)
    lines = [f"**{LABEL[a]} minus {LABEL[b]}** (pp, paired by pruning ratio and seed)", "",
             "| Pruning | seed 1 | seed 2 | seed 3 | mean | 95% CI |", "|---|---|---|---|---|---|"]
    for p, grp in d.groupby("pruning"):
        lo, hi = ci95(grp["diff"])
        vals = " | ".join(f"{v:+.2f}" for v in grp.sort_values("seed")["diff"])
        lines.append(f"| {p:.0%} | {vals} | {grp['diff'].mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lo, hi = ci95(d["diff"])
    w = stats.wilcoxon(d["diff"], alternative="greater")
    rho = stats.spearmanr(d["pruning"], d["diff"])
    lines += ["", f"All 12 pairs: mean {d['diff'].mean():+.2f} pp, 95% CI [{lo:+.2f}, {hi:+.2f}], "
              f"{(d['diff'] > 0).sum()} of 12 positive, Wilcoxon signed-rank (one-sided, greater) p = {w.pvalue:.4f}. "
              f"Trend with pruning: Spearman rho = {rho.statistic:+.2f}, p = {rho.pvalue:.3f}.", ""]
    return lines, d


def hypotheses(df, s, hw, base, sel):
    L = ["# Analysis results", "",
         f"Baseline (fp32, 0 % pruning): {base:.2f} % mean top-1 over 3 seeds. Energy = {POWER_W * 1000:.1f} mW "
         f"(datasheet estimate: {UA_PER_MHZ:.0f} uA/MHz x {CFG['hardware']['clock_mhz']} MHz x {VDD} V) x measured latency; "
         "it ranks configurations exactly as latency does.", ""]

    L += ["## Selection rule", "",
          "Pre-registered: lowest energy whose mean accuracy is within 1 pp of the baseline.", ""]
    first = sel.iloc[0]
    if first.pruning == 0 and first.quant == "fp32":
        L += ["**No compressed configuration meets the 1 pp rule**: int8 at 0 % pruning loses "
              f"{s[(s.pruning == 0) & (s.quant == 'ptq')].drop_vs_baseline_pp.iloc[0]:.2f} pp (PTQ). "
              "By the rule the uncompressed baseline is selected. The table shows the best configuration at looser tolerances.", ""]
    L += ["| max drop (pp) | pruning | variant | accuracy | drop | energy (mJ) | latency (ms) | energy saving |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in sel.iterrows():
        L.append(f"| {r.max_drop_pp} | {r.pruning:.0%} | {LABEL[r.quant]} | {r.acc_mean:.2f} | {r.drop_pp:.2f} | "
                 f"{r.energy_mj:.2f} | {r.latency_ms:.1f} | {r.energy_saving_x:.1f}x |")

    L += ["", "## Pareto front (pre-registered variants: fp32, PTQ, QAT-5)", ""]
    front = s[s.pareto_main].sort_values("energy_mj")
    for _, r in front.iterrows():
        L.append(f"- {r.pruning:.0%} {LABEL[r.quant]}: {r.acc_mean:.2f} ± {r.acc_std:.2f} %, "
                 f"{r.energy_mj:.2f} mJ, {r.latency_ms:.1f} ms ({r.technique})")

    main = s[s.quant.isin(MAIN)]
    prune_only = main[main.technique == "pruning only"]
    combined = main[main.technique == "pruning + quantization"]
    quant_only = main[main.technique == "quantization only"]

    def dominated_by(pts, by):
        return [any((b.energy_mj <= r.energy_mj) and (b.acc_mean >= r.acc_mean) for _, b in by.iterrows())
                for _, r in pts.iterrows()]
    L += ["", "## H1: pruning + int8 dominates either technique alone", "",
          f"- Pruning-only (fp32) points on the front: {int(prune_only.pareto_main.sum())} of {len(prune_only)}; "
          f"dominated by a combined point: {sum(dominated_by(prune_only, combined))} of {len(prune_only)}.",
          f"- Quantization-only (int8 at 0 %) points on the front: {int(quant_only.pareto_main.sum())} of {len(quant_only)}; "
          f"dominated by a combined point: {sum(dominated_by(quant_only, combined))} of {len(quant_only)}.",
          f"- Combined points on the front: {int(combined.pareto_main.sum())} of {len(combined)}.",
          "- Reading: combined configurations dominate pruning alone and extend the front to lower energy, "
          "but do not dominate quantization alone, which keeps the most accurate low-energy point. "
          "H1 holds against pruning alone, not against quantization alone.", ""]

    L += ["## H2: QAT recovers accuracy that PTQ loses, more so at high pruning", ""]
    for a, b in [("qat", "ptq"), ("qat15", "ptq"), ("qat15", "qat")]:
        lines, _ = h2_block(df, a, b)
        L += lines
    L += ["With 3 seeds per ratio the per-ratio intervals are wide; the 12-pair test is the main evidence.", ""]

    L += ["## H3: MAC count predicts measured latency", ""]
    distinct = hw[hw.quant.isin(["fp32", "ptq"])].merge(
        s[["pruning", "quant", "macs", "params"]], on=["pruning", "quant"])
    rho_all = stats.spearmanr(distinct.macs, distinct.latency_ms)
    L.append(f"- All 8 distinct architectures (fp32 and int8): Spearman rho = {rho_all.statistic:.2f}, "
             f"p = {rho_all.pvalue:.3f} (pre-registered threshold rho >= 0.8: "
             f"{'met' if rho_all.statistic >= 0.8 else 'not met'}).")
    for q in ["fp32", "ptq"]:
        sub = distinct[distinct.quant == q]
        r = stats.spearmanr(sub.macs, sub.latency_ms).statistic
        ns = (sub.latency_ms * 1e6 / sub.macs)
        L.append(f"- Within {LABEL[q]} only: rho = {r:.2f}; {ns.min():.1f}-{ns.max():.1f} ns per MAC.")
    swap = distinct[(distinct.quant == "ptq") & (distinct.pruning == 0)].iloc[0]
    worse = distinct[(distinct.quant == "fp32") & (distinct.macs < swap.macs) & (distinct.latency_ms > swap.latency_ms)]
    L += [f"- Counter-example: int8 at 0 % has {swap.macs / 1e6:.1f} M MACs and runs in {swap.latency_ms:.1f} ms, while "
          + ", ".join(f"fp32 at {r.pruning:.0%} has only {r.macs / 1e6:.1f} M MACs but takes {r.latency_ms:.1f} ms"
                      for _, r in worse.iterrows()) + ".",
          "- Reading: MACs rank models correctly within one numeric precision but not across precisions, "
          "because int8 kernels do far more MACs per cycle than fp32 ones.", ""]

    L += ["## Seed sensitivity", ""]
    spread = df[df.quant != "fp32"].groupby(["pruning", "quant"])["accuracy"].agg(["min", "max"])
    spread["range"] = spread["max"] - spread["min"]
    for (p, q), r in spread[spread["range"] > 5].iterrows():
        L.append(f"- {p:.0%} {LABEL[q]}: {r['min']:.1f}-{r['max']:.1f} % across seeds (range {r['range']:.1f} pp)")
    L += ["", "fp32 models vary by at most "
          f"{df[df.quant == 'fp32'].groupby('pruning').accuracy.agg(lambda x: x.max() - x.min()).max():.1f} pp across seeds."]
    return L


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


def fig_pareto(s):
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
    front = s[s.pareto_main].sort_values("energy_mj")
    ax.plot(front.energy_mj, front.acc_mean, color=INK2, linewidth=1.2, linestyle="--", zorder=1,
            label="Pareto front (fp32, PTQ, QAT-5)")
    nudge = {"fp32": 1.0, "ptq": 0.955, "qat": 1.0, "qat15": 1.045}  # int8 variants share an energy; spread for legibility
    for q in MAIN + SENS:
        sub = s[s.quant == q]
        ax.errorbar(sub.energy_mj * nudge[q], sub.acc_mean, yerr=sub.acc_std, fmt=MARKER[q], color=COLOR[q], ms=8,
                    mfc=COLOR[q] if q in MAIN else SURFACE, mew=2, elinewidth=1.2, capsize=3, label=LABEL[q],
                    zorder=3)
    for _, r in s[s.quant.isin(["fp32", "qat"])].iterrows():
        ax.annotate(f"{r.pruning:.0%} pruned", (r.energy_mj, r.acc_mean), textcoords="offset points",
                    xytext=(-28, -18) if r.quant == "fp32" else (12, -16), color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xlim(1.8, 70)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax, "Accuracy vs energy per inference (STM32H7B3, 280 MHz)",
          "Energy per inference, mJ (log scale; estimate = 111 mW x measured latency)",
          "Top-1 accuracy, % (mean ± std, 3 seeds)")
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_pareto_accuracy_energy.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_qat_gain(df):
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ratios = sorted(df.pruning.unique())
    ax.axhline(0, color=INK2, linewidth=1)
    for k, q in enumerate(["qat", "qat15"]):
        d = paired(df, q, "ptq")
        x = np.array([ratios.index(p) for p in d.pruning]) + (k - 0.5) * 0.28
        ax.scatter(x, d["diff"], s=40, marker=MARKER[q], color=COLOR[q], alpha=0.55, zorder=2,
                   edgecolor=SURFACE, linewidth=1)
        m = d.groupby("pruning")["diff"].mean()
        ax.scatter(np.arange(len(ratios)) + (k - 0.5) * 0.28, m.values, s=150, marker="_", color=COLOR[q],
                   linewidth=3, zorder=3, label=f"{LABEL[q]} − PTQ (dots: seeds, bar: mean)")
    ax.set_xticks(range(len(ratios)), [f"{p:.0%}" for p in ratios])
    style(ax, "Accuracy QAT recovers over PTQ, per model", "Channels pruned", "QAT − PTQ accuracy, pp")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_qat_gain_over_ptq.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_macs(s):
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    for q in ["fp32", "ptq"]:
        sub = s[s.quant == q].sort_values("macs")
        ax.plot(sub.macs / 1e6, sub.latency_ms, marker=MARKER[q], color=COLOR[q], ms=8, linewidth=2,
                label="fp32" if q == "fp32" else "int8 (PTQ and QAT, same graph)")
        for _, r in sub.iterrows():
            ax.annotate(f"{r.pruning:.0%}", (r.macs / 1e6, r.latency_ms), textcoords="offset points",
                        xytext=(6, 6), color=INK2, fontsize=9)
    style(ax, "MAC count vs measured latency", "MACs per inference, millions", "Latency, ms")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_macs_vs_latency.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_seeds(df):
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    ratios = sorted(df.pruning.unique())
    order = MAIN + SENS
    for k, q in enumerate(order):
        sub = df[df.quant == q]
        off = (k - 1.5) * 0.15
        x = np.array([ratios.index(p) for p in sub.pruning]) + off
        ax.scatter(x, sub.accuracy, s=42, marker=MARKER[q], color=COLOR[q],
                   facecolor=COLOR[q] if q in MAIN else SURFACE, linewidth=1.5, label=LABEL[q], zorder=3)
    ax.set_xticks(range(len(ratios)), [f"{p:.0%}" for p in ratios])
    style(ax, "Accuracy of every model: int8 spread grows with pruning", "Channels pruned",
          "Top-1 accuracy, % (one marker per seed)")
    ax.legend(frameon=False, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_accuracy_per_seed.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, hw = load()
    s, base = summarize(df)
    sel = selection(s, base, MAIN)
    s.to_csv(OUT / "summary_by_config.csv", index=False)
    sel.to_csv(OUT / "selection.csv", index=False)
    text = hypotheses(df, s, hw, base, sel)
    sel15 = selection(s, base, MAIN + SENS)
    text += ["", "## Selection including QAT-15 (sensitivity)", "",
             "| max drop (pp) | pruning | variant | accuracy | energy (mJ) |", "|---|---|---|---|---|"]
    text += [f"| {r.max_drop_pp} | {r.pruning:.0%} | {LABEL[r.quant]} | {r.acc_mean:.2f} | {r.energy_mj:.2f} |"
             for _, r in sel15.iterrows()]
    (OUT / "hypotheses.md").write_text("\n".join(text) + "\n")
    fig_pareto(s)
    fig_qat_gain(df)
    fig_macs(s)
    fig_seeds(df)
    print("\n".join(text))


if __name__ == "__main__":
    main()
