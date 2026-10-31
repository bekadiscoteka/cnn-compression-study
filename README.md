# Quantization and Pruning Trade-offs for Embedded CNN Inference

Controlled comparison of structured channel pruning (0, 25, 50, 75 %) and int8 quantization (PTQ, QAT) for MobileNetV2 (width 0.35) on CIFAR-10. Accuracy is measured on the exported ONNX models; latency and memory on a real STM32H7B3I-DK board through ST Edge AI Developer Cloud. 3 seeds × 4 pruning ratios × 3 variants (fp32, PTQ, QAT) = 36 models, plus a QAT-15-epoch sensitivity check (12 more).

The research plan is fixed in `docs/methodology.docx`, committed before any experiment. Every deviation from it is dated and explained in `docs/notes.md`.

## Status

| Step | State |
|---|---|
| Plan fixed | done |
| Training pipeline (baseline, pruning, ONNX export, PTQ, QAT) | done |
| 36 models + 12 `qat15` trained and evaluated | done (`results/results_v2.csv`, 48 rows) |
| Calibration method chosen on a validation split | done: percentile 99.99 (`results/calibration_check.csv`) |
| Board fixed | done: STM32H7B3I-DK, 280 MHz, ST Edge AI 4.1.0 |
| Hardware benchmarks, seed 1 | 12 of 12 done; one Balanced vs RAM check on a single model pending |
| Energy, Pareto analysis, hypothesis tests | done: `analysis.py` -> `results/analysis/` (see `hypotheses.md`) |
| QAT budget check (`qat15`, 15 epochs) | done |

## Structure

```
├── README.md
├── config.yaml            all experiment settings
├── requirements.txt
├── colab.ipynb            runs the pipeline on a Colab GPU (Run all)
├── train.py               whole matrix: train, prune, export, quantize, evaluate
├── calibrate_check.py     picks the int8 calibration method on a validation split
├── parse_benchmarks.py    results/benchmarks/ logs -> results/hardware.csv
├── analysis.py            energy, Pareto fronts, selection rule, H1-H3 tests, figures
├── docs/
│   ├── methodology.docx   research plan, hypotheses, selection rule
│   └── notes.md           dated log of problems, deviations and decisions
└── results/
    ├── results_v2.csv         36 models: accuracy, params, MACs  <- source of accuracy figures
    ├── results_v1_minmax.csv  first run with MinMax calibration, kept for the record
    ├── calibration_check.csv  int8 accuracy drop per calibration method (validation split)
    ├── hardware.csv           one row per benchmarked model: latency, Flash, RAM, settings
    ├── benchmarks/            raw ST Edge AI logs, one folder per model
    └── analysis/              summary_by_config.csv, selection.csv, hypotheses.md, fig1-fig4
```

Trained models (`.pt`, `.onnx`) and the dataset are not in git; they live on Google Drive in `cnn-study/`.

## How data flows

```
colab.ipynb ─► calibrate_check.py ─► calibration_choice.json ─┐
            └► train.py ◄──────────────────────────────────────┘
                 │  models/*.onnx (Drive)          results_v2.csv
                 ▼
ST Edge AI Developer Cloud (STM32H7B3I-DK) ─► logs ─► results/benchmarks/<model>/
                                                         │
                                         parse_benchmarks.py ─► results/hardware.csv
                                                         │
                      analysis.py: results_v2.csv + hardware.csv ─► results/analysis/
```

Hardware figures are measured once per architecture (seed 1): seeds change weight values, not layer sizes, and PTQ and QAT models share one graph. The analysis joins them to every row of the same pruning ratio and int8/fp32 type.

## Naming

`mbv2_p50_qat_s2` = MobileNetV2, 50 % of channels pruned, int8 via QAT, seed 2. Quantization: `fp32`, `ptq`, `qat` (5 epochs, pre-registered), `qat15` (15 epochs, sensitivity check). The same name is used for files on Drive, rows in `results_v2.csv` and folders in `results/benchmarks/`.

## Running

1. Open `colab.ipynb` in Google Colab (File → Open notebook → GitHub), choose a T4 GPU, Run all. Finished work is skipped, so after a disconnect just Run all again.
2. Benchmark the `.onnx` files from Drive in ST Edge AI Developer Cloud on STM32H7B3I-DK (Quantize step skipped, models are already int8). Save the logs to `results/benchmarks/<model>/` and run `python parse_benchmarks.py`; it warns if settings differ between runs.

Local pipeline check without a GPU or dataset: `python train.py --smoke`.

## Rules

1. Change settings in `config.yaml`, not in code.
2. Fix the seed; report mean ± std over 3 seeds.
3. Never delete result rows or files; start a new version and say why in `docs/notes.md`.
4. Measure accuracy on the exported ONNX model, never report latency or energy without accuracy.
5. Choose methods on validation data, never on the test set.
