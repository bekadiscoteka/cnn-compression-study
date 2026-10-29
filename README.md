# Quantization and Pruning Trade-offs for Embedded CNN Inference

Controlled comparison of structured pruning and int8 quantization (PTQ, QAT) for MobileNetV2 on CIFAR-10, benchmarked on a real STM32 board via ST Edge AI Developer Cloud.

The plan is fixed in `methodology.docx` (committed before any experiments).

## Files

| File | Purpose |
|---|---|
| `methodology.docx` | Research plan, hypotheses and selection rule |
| `config.yaml` | All experiment settings |
| `results.csv` | One row per run; the only source of numbers for the thesis |
| `notes.md` | Problems, deviations from the plan, decisions |
| `train.py` | One run: baseline, pruning, ONNX export, int8 PTQ, accuracy, row in `results.csv` |
| `colab.ipynb` | Runs `train.py` on a Colab GPU |

## Running

Open `colab.ipynb` in Google Colab (File → Open notebook → GitHub), set `PRUNING` and `SEED`, run all cells. Models and `results.csv` are saved to Google Drive in `cnn-study/`.

Local pipeline check without a GPU: `python train.py --smoke`.

## Rules

1. Change settings in `config.yaml`, not in code.
2. Always fix the seed.
3. Never delete rows from `results.csv`; mark failed runs in `notes.md`.
4. Measure accuracy on the exported ONNX model.
5. Save checkpoints and models to Google Drive.
