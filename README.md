# Quantization and Pruning Trade-offs for Embedded CNN Inference

Controlled comparison of structured pruning and int8 quantization (PTQ, QAT) for MobileNetV2 on CIFAR-10, benchmarked on a real STM32 board via ST Edge AI Developer Cloud.

The plan is fixed in `methodology.docx` (committed before any experiments).

## Files

| File | Purpose |
|---|---|
| `methodology.docx` | Research plan, hypotheses and selection rule |
| `config.yaml` | All experiment settings |
| `results_v2.csv` | One row per model (accuracy, params, MACs); the source of accuracy figures |
| `results_v1_minmax.csv` | First run with MinMax calibration, kept for the record (see `notes.md`) |
| `hardware.csv` | One row per benchmarked model: latency, Flash, RAM on the board |
| `benchmarks/` | Raw ST Edge AI logs, one folder per model |
| `notes.md` | Problems, deviations from the plan, decisions |
| `train.py` | Whole matrix: baseline, pruning, ONNX export, int8 PTQ and QAT, accuracy, rows in `results.csv` |
| `calibrate_check.py` | Chooses the int8 calibration method on a validation split |
| `parse_benchmarks.py` | Turns `benchmarks/` logs into `hardware.csv` and checks settings match |
| `colab.ipynb` | Runs `calibrate_check.py` and `train.py` on a Colab GPU |

## Running

Open `colab.ipynb` in Google Colab (File → Open notebook → GitHub), choose a T4 GPU and Run all. `train.py` runs the whole matrix (36 models), skips anything already in `results.csv`, and saves models and results to Google Drive in `cnn-study/`. One cell only: `python train.py --pruning 0.5 --seed 1 --out ...`.

Local pipeline check without a GPU: `python train.py --smoke`.

## Rules

1. Change settings in `config.yaml`, not in code.
2. Always fix the seed.
3. Never delete rows from `results.csv`; mark failed runs in `notes.md`.
4. Measure accuracy on the exported ONNX model.
5. Save checkpoints and models to Google Drive.
