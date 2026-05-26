# Experiment 1: Initial OpenCL GPU Port Baseline
Date: 2026-05-26
Status: PASS
Hypothesis: CPU-resize-normalize algorithm correctly ported to OpenCL GPU.

## Results
| Resolution | Upload_ms | Kernel_ms | Copy_ms | Transp_ms | Down_ms | Total_ms |
|------------|-----------|-----------|---------|-----------|---------|----------|
| 320x240 | 0.019 | 0.238 | 0.124 | 0.320 | 0.013 | 0.714 |
| 800x600 | (from stride 2) | | | | | |
| 1280x720 | (from stride 2) | | | | | |
| 1920x1080 | 0.011 | 0.555 | 0.126 | 0.328 | 0.013 | 1.032 |

Correctness: PASS (max_diff=0.000002)

## Lessons
- Initial OpenCL port works correctly for all resolutions tested
- resize_bilinear_normalized kernel time scales with src resolution as expected
- GPU copy (T=2 temporal replicate) is constant ~0.125ms regardless of resolution
- This is the baseline — all future experiments compare against this
