# Analysis results

Baseline (fp32, 0 % pruning): 89.49 % mean top-1 over 3 seeds. Energy = 110.9 mW (datasheet estimate: 120 uA/MHz x 280 MHz x 3.3 V) x measured latency; it ranks configurations exactly as latency does.

## Selection rule

Pre-registered: lowest energy whose mean accuracy is within 1 pp of the baseline.

**No compressed configuration meets the 1 pp rule**: int8 at 0 % pruning loses 1.61 pp (PTQ). By the rule the uncompressed baseline is selected. The table shows the best configuration at looser tolerances.

| max drop (pp) | pruning | variant | accuracy | drop | energy (mJ) | latency (ms) | energy saving |
|---|---|---|---|---|---|---|---|
| 1 | 0% | fp32 | 89.49 | 0.00 | 47.36 | 427.2 | 1.0x |
| 2 | 0% | int8 PTQ | 87.88 | 1.61 | 11.14 | 100.5 | 4.2x |
| 3 | 0% | int8 PTQ | 87.88 | 1.61 | 11.14 | 100.5 | 4.2x |
| 5 | 25% | int8 QAT (5 ep.) | 85.88 | 3.61 | 9.33 | 84.2 | 5.1x |
| 10 | 25% | int8 QAT (5 ep.) | 85.88 | 3.61 | 9.33 | 84.2 | 5.1x |
| 15 | 50% | int8 QAT (5 ep.) | 75.66 | 13.83 | 4.69 | 42.3 | 10.1x |

## Pareto front (pre-registered variants: fp32, PTQ, QAT-5)

- 75% int8 QAT (5 ep.): 53.19 ± 16.88 %, 2.24 mJ, 20.2 ms (pruning + quantization)
- 50% int8 QAT (5 ep.): 75.66 ± 6.00 %, 4.69 mJ, 42.3 ms (pruning + quantization)
- 25% int8 QAT (5 ep.): 85.88 ± 0.74 %, 9.33 mJ, 84.2 ms (pruning + quantization)
- 0% int8 PTQ: 87.88 ± 0.68 %, 11.14 mJ, 100.5 ms (quantization only)
- 0% fp32: 89.49 ± 0.13 %, 47.36 mJ, 427.2 ms (none)

## H1: pruning + int8 dominates either technique alone

- Pruning-only (fp32) points on the front: 0 of 3; dominated by a combined point: 2 of 3.
- Quantization-only (int8 at 0 %) points on the front: 1 of 2; dominated by a combined point: 0 of 2.
- Combined points on the front: 3 of 6.
- Reading: combined configurations dominate pruning alone and extend the front to lower energy, but do not dominate quantization alone, which keeps the most accurate low-energy point. H1 holds against pruning alone, not against quantization alone.

## H2: QAT recovers accuracy that PTQ loses, more so at high pruning

**int8 QAT (5 ep.) minus int8 PTQ** (pp, paired by pruning ratio and seed)

| Pruning | seed 1 | seed 2 | seed 3 | mean | 95% CI |
|---|---|---|---|---|---|
| 0% | -0.49 | +0.02 | -0.04 | -0.17 | [-0.86, +0.52] |
| 25% | +0.26 | +0.14 | -0.19 | +0.07 | [-0.51, +0.65] |
| 50% | +1.26 | +0.51 | +0.22 | +0.66 | [-0.67, +2.00] |
| 75% | +2.13 | +1.09 | +0.30 | +1.17 | [-1.11, +3.45] |

All 12 pairs: mean +0.43 pp, 95% CI [-0.03, +0.90], 9 of 12 positive, Wilcoxon signed-rank (one-sided, greater) p = 0.0261. Trend with pruning: Spearman rho = +0.84, p = 0.001.

**int8 QAT (15 ep.) minus int8 PTQ** (pp, paired by pruning ratio and seed)

| Pruning | seed 1 | seed 2 | seed 3 | mean | 95% CI |
|---|---|---|---|---|---|
| 0% | -0.02 | +0.01 | +0.19 | +0.06 | [-0.22, +0.34] |
| 25% | +0.40 | +0.26 | +0.15 | +0.27 | [-0.04, +0.58] |
| 50% | +2.24 | +3.52 | +0.67 | +2.14 | [-1.40, +5.69] |
| 75% | +4.71 | +1.48 | +0.18 | +2.12 | [-3.67, +7.92] |

All 12 pairs: mean +1.15 pp, 95% CI [+0.16, +2.14], 11 of 12 positive, Wilcoxon signed-rank (one-sided, greater) p = 0.0007. Trend with pruning: Spearman rho = +0.69, p = 0.013.

**int8 QAT (15 ep.) minus int8 QAT (5 ep.)** (pp, paired by pruning ratio and seed)

| Pruning | seed 1 | seed 2 | seed 3 | mean | 95% CI |
|---|---|---|---|---|---|
| 0% | +0.47 | -0.01 | +0.23 | +0.23 | [-0.37, +0.83] |
| 25% | +0.14 | +0.12 | +0.34 | +0.20 | [-0.10, +0.50] |
| 50% | +0.98 | +3.01 | +0.45 | +1.48 | [-1.88, +4.84] |
| 75% | +2.58 | +0.39 | -0.12 | +0.95 | [-2.61, +4.51] |

All 12 pairs: mean +0.72 pp, 95% CI [+0.07, +1.36], 10 of 12 positive, Wilcoxon signed-rank (one-sided, greater) p = 0.0012. Trend with pruning: Spearman rho = +0.28, p = 0.377.

With 3 seeds per ratio the per-ratio intervals are wide; the 12-pair test is the main evidence.

## H3: MAC count predicts measured latency

- All 8 distinct architectures (fp32 and int8): Spearman rho = 0.73, p = 0.039 (pre-registered threshold rho >= 0.8: not met).
- Within fp32 only: rho = 1.00; 24.3-35.2 ns per MAC.
- Within int8 PTQ only: rho = 1.00; 5.7-10.2 ns per MAC.
- Counter-example: int8 at 0 % has 17.6 M MACs and runs in 100.5 ms, while fp32 at 25% has only 10.8 M MACs but takes 289.9 ms, fp32 at 50% has only 5.6 M MACs but takes 165.4 ms.
- Reading: MACs rank models correctly within one numeric precision but not across precisions, because int8 kernels do far more MACs per cycle than fp32 ones.

## Seed sensitivity

- 50% int8 PTQ: 68.2-79.2 % across seeds (range 11.0 pp)
- 50% int8 QAT (5 ep.): 68.7-79.5 % across seeds (range 10.7 pp)
- 50% int8 QAT (15 ep.): 71.8-79.9 % across seeds (range 8.2 pp)
- 75% int8 PTQ: 31.7-64.4 % across seeds (range 32.8 pp)
- 75% int8 QAT (5 ep.): 33.8-64.7 % across seeds (range 30.9 pp)
- 75% int8 QAT (15 ep.): 36.4-64.6 % across seeds (range 28.2 pp)

fp32 models vary by at most 2.3 pp across seeds.

## Selection including QAT-15 (sensitivity)

| max drop (pp) | pruning | variant | accuracy | energy (mJ) |
|---|---|---|---|---|
| 1 | 0% | fp32 | 89.49 | 47.36 |
| 2 | 0% | int8 QAT (15 ep.) | 87.94 | 11.14 |
| 3 | 0% | int8 QAT (15 ep.) | 87.94 | 11.14 |
| 5 | 25% | int8 QAT (15 ep.) | 86.08 | 9.33 |
| 10 | 25% | int8 QAT (15 ep.) | 86.08 | 9.33 |
| 15 | 50% | int8 QAT (15 ep.) | 77.14 | 4.69 |
