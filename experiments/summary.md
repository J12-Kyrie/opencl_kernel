# Experiments Summary

| Exp | Date | Description | Kernel_ms | Total_ms | Pass | Notes |
|-----|------|-------------|-----------|----------|------|-------|
| 1 | 2026-05-26 | Initial OpenCL GPU port baseline | 0.411 | 0.879 | PASS | New best, kernel=resize_bilinear_normalized+transpose_to_patch |
| 5 | 2026-05-26 | image2d_t HW bilinear (TP) | 0.090 | 0.579 | FAIL | ROLLBACK: read_imagef broken on Adreno 663, DEAD END |
| 6 | 2026-05-26 | vload4 vectorized loads | 0.247 | 0.724 | PASS | Neutral: compiler already vectorizes, no measurable gain |
| 7 | 2026-05-26 | WG size sweep {32,64,128,256} | 0.244 | 0.719 | PASS | Neutral: driver auto-tune already optimal, all WG within 1% |
| 8 | 2026-05-26 | Transpose loop unroll + precomputed offsets | 0.311 | 0.603 | PASS | NEW BEST: -16% vs baseline |
| 9 | 2026-05-26 | vload3 in transpose (replaces 12 scalar reads w/ 4 vector reads) | 0.310 | 0.584 | PASS | NEW BEST: -3.1% vs exp_8, cumulative -18.7% vs baseline |
