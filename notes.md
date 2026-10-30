# Notes

Write a dated line only when something happens: a problem, a deviation from the plan, or a decision.

2026-10-29 — Plan fixed in methodology.docx before any experiments.
2026-10-29 — Pipeline step 1 added: baseline, structured pruning, ONNX export, int8 PTQ. QAT comes next.
2026-10-30 — train.py now runs the whole matrix in one call and skips finished runs. QAT added: BN folded into convs, 5 epochs with int8 rounding of weights and activations simulated, then exported through the same ONNX Runtime quantizer and calibration set as PTQ, so PTQ and QAT models have identical graphs and differ only in weights.
2026-10-30 — Board fixed: NUCLEO-H743ZI2 via ST Edge AI Developer Cloud, optimization Balanced, models uploaded already quantized (ST Quantize step skipped).
2026-10-30 — Board changed to STM32H7B3I-DK (280 MHz). NUCLEO-H743ZI2 failed to link: the ST benchmark firmware places the activation buffer in 128 KB DTCM, the fp32 baseline needs 270 KB. STM32N6570-DK rejected: it has a Neural-ART NPU and no internal Flash, so weights run from external memory. On the H7B3 the fp32 baseline sits fully in internal memory (weights in internal Flash, activations in DTCM + internal RAM, external SDRAM unused): 427.2 ms per inference. ST validation on the board matches the ONNX model (cosine 1.0, RMSE 2e-6).
2026-10-30 — Hardware results kept in hardware.csv (one row per architecture; seeds and PTQ/QAT share them), raw ST reports in st_reports/.
