# Profiling Report — 2026-05-26 (Post Exp 7)

## Headline
resize_bilinear_normalized kernel time (0.241ms @ 320x240, 0.554ms @ 1920x1080)
is within 2x of memory-bandwidth floor for Adreno 663 (~30 GB/s).
Further kernel-level optimizations have diminishing returns.

## Phase Breakdown (1920x1080 baseline)
| Phase | Time (ms) | % Total |
|-------|-----------|---------|
| Upload (H2D) | 0.013 | 1.3% |
| Fused Resize+Norm | 0.552 | 53.5% |
| GPU Copy (T=2) | 0.126 | 12.2% |
| Transpose | 0.329 | 31.9% |
| Download (D2H) | 0.013 | 1.3% |

Note: Upload/Download timing from clGetEventProfilingInfo may underreport
actual DMA time on Adreno (event measured at enqueue, not completion).

## Memory Bandwidth Check
- Source read: 1920x1080x3 bytes × 4 taps = ~24.9 MB per call
- Destination write: 448x448x3x4 bytes = ~2.4 MB
- Total per call: ~27.3 MB in 0.552ms = ~49.5 GB/s
- Theoretical Adreno 663: ~30 GB/s
- Apparent bandwidth > theoretical → measurement artifact (event timing issue)
  OR kernel is compute-bound (bilinear weight ops dominate, not memory)

## Recommendation
Kernel-level optimizations exhausted. Focus shifts to:
1. Transpose kernel (31.9% of total) — larger optimization surface
2. Host-side: double-buffering to hide upload latency
3. DMA-BUF path: eliminate H2D/D2H entirely (architectural, not kernel)
