# Experiment 10: Fix 4x Work-Item Redundancy
Date: 2026-05-26
Status: PASS — NEW BEST

## Bug
SEQ_LEN = GRID_T * GRID_H * GRID_W * MERGE_SIZE * MERGE_SIZE = 3136 per frame
launched 6272 work-items for T=2. Correct count: GRID_H * GRID_W = 784 per frame,
NDRange = 1568 for T=2. 4x redundancy.

## Fix
- benchmark.cpp: SEQ_LEN = GRID_H * GRID_W (removed MERGE_SIZE^2 factor)
- kernel.cl: total guard = grid_t * grid_h * grid_w, mh/mw iterated locally
- Buffer sizes updated to account for merge^2 in output patches

## Results
| Resolution | Exp 9 Total | Exp 10 Total | Delta |
|------------|------------|-------------|-------|
| 320x240 | 0.584ms | 0.569ms | -2.6% |
| 1920x1080 | 0.679ms | 0.686ms | +1.0% (noise) |
