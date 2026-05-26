# Convergence Report — opencl_kernel Image Preprocessing

## Final Best Kernel (Exp 9)
- Kernel: vload4 resize_bilinear_normalized + vload3 transpose_to_patch
- Best Total (320x240): **0.581ms** (baseline: 0.718ms, **-19.1%**)
- Best Total (1920x1080): **0.736ms** (baseline: 1.033ms, **-28.8%**)
- GPU: QUALCOMM Adreno(TM) 663, Max WG: 1024
- Correctness: ALL 8 RESOLUTIONS PASS (max_diff < 1e-5)

## Optimization Trail
| Exp | Direction | Result | Cumulative Gain |
|-----|-----------|--------|----------------|
| 1 | Buffer-based baseline | 0.718ms | — |
| 5 | image2d_t TP HW bilinear | DEAD END | — (driver bug) |
| 6 | vload4 vectorized loads | Neutral | — (compiler auto) |
| 7 | WG size sweep | Flat | — (driver auto-tune) |
| 8 | Transpose loop unroll | **ACCEPT** | -16.0% |
| 9 | vload3 in transpose | **ACCEPT** | -19.1% |

## Known Constraints (Why We Stop)
1. Adreno 663 read_imagef broken in kernel → image2d_t path unavailable
2. Adreno compiler auto-vectorizes → vload4 no gain
3. Adreno driver auto-tunes WG size → explicit WG tuning no gain
4. 448x448 fixed output → no shape-dependent optimization opportunities
5. Transpose kernel now fully unrolled with vload3 → diminishing returns

## Remaining Host-Level Opportunities (Not in kernel.cl scope)
- DMA-BUF zero-copy: eliminate H2D/D2H entirely (frame already in GPU memory)
- Double-buffering: overlap upload with compute for streaming workloads
- clFinish removal: event-chain instead of blocking

## Verdict: CONVERGED (kernel-level)
Per success criteria:
- (b) Pareto optimal: 19% gain with no code complexity regression
- (c) No further improvement under known kernel-level constraints
- (Q1) 5 consecutive neutral results + Research Agent ceiling analysis = STOP
