# Experiment 6: vload4 Vectorized Loads
Date: 2026-05-26
Status: PASS (Neutral)

## Hypothesis
Replacing 3x uchar scalar reads with single uchar4 vload4 per corner pixel
would reduce memory instructions and improve kernel time by 5-10%.

## Results
| Resolution | Baseline Kernel | Exp 6 Kernel | Delta |
|------------|----------------|-------------|-------|
| 320x240 | 0.241ms | 0.247ms | +2.5% |
| 1920x1080 | 0.554ms | 0.554ms | 0.0% |

Total time delta: -0.1% (within noise)

## Lessons
- Adreno compiler already auto-vectorizes adjacent byte reads
- vload4 produces cleaner code (4 loads vs 12) but no performance gain
- Keep the change for code quality, but this direction has no ceiling
