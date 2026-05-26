# Experiment 8: Transpose Loop Unrolling + Precomputed Offsets
Date: 2026-05-26
Status: PASS — NEW BEST

## Hypothesis
Manual unrolling of transpose kernel's 3 nested loops (channels=3×merge=2×merge=2=12 iterations)
with precomputed base offsets would reduce arithmetic overhead and improve throughput.

## Results
| Resolution | Baseline Total | Exp 8 Total | Delta |
|------------|---------------|-------------|-------|
| 320x240 | 0.718ms | 0.603ms | -16.0% |
| 1920x1080 | 1.033ms | 0.694ms | -32.8% |

Transpose kernel alone: 0.320→0.278ms (-13%) @ 320x240, 0.329→0.243ms (-26%) @ 1920x1080.
Additional improvement observed in resize kernel phase (possibly due to reduced memory contention or event overlap).

## Decision: ACCEPT (>5% clear win)
Update BEST.md: best_exp=8, best_total_ms=0.603
