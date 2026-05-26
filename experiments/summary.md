# Experiments Summary

| Exp | Date | Description | Kernel_ms | Total_ms | Pass | Notes |
|-----|------|-------------|-----------|----------|------|-------|
| 1 | 2026-05-26 | Initial OpenCL GPU port baseline | 0.411 | 0.879 | PASS | New best, kernel=resize_bilinear_normalized+transpose_to_patch |
| 5 | 2026-05-26 | image2d_t HW bilinear (TP) | 0.090 | 0.579 | FAIL | ROLLBACK: read_imagef broken on Adreno 663, DEAD END |
