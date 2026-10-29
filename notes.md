# Notes

Write a dated line only when something happens: a problem, a deviation from the plan, or a decision.

2026-10-29 — Plan fixed in methodology.docx before any experiments.
2026-10-29 — Pipeline step 1 added: baseline, structured pruning, ONNX export, int8 PTQ. QAT comes next.
2026-10-30 — train.py now runs the whole matrix in one call and skips finished runs. QAT added: BN folded into convs, 5 epochs with int8 rounding of weights and activations simulated, then exported through the same ONNX Runtime quantizer and calibration set as PTQ, so PTQ and QAT models have identical graphs and differ only in weights.
