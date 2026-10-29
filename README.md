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

## Rules

1. Change settings in `config.yaml`, not in code.
2. Always fix the seed.
3. Never delete rows from `results.csv`; mark failed runs in `notes.md`.
4. Measure accuracy on the exported ONNX model.
5. Save checkpoints and models to Google Drive.
