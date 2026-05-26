# /optimize — 8-Step Autonomous OpenCL Kernel Optimization Loop

## Overview
This command drives the continuous optimization of `solution/kernel/kernel.cl`.
It follows a bootstrap check + 8-step cycle.

## Step 0: BOOTSTRAP — First-Run Baseline (conditional)
- Check if `experiments/summary.md` is empty (no experiments logged)
- If empty: run `/benchmark full`, log as `exp_1` (baseline)
- Update summary.md with baseline entry
- If summary.md already has entries: skip to Step 1
- This ensures every optimization campaign starts from a measured baseline

## Step 1: ASSESS — Read Current State
- Read `solution/kernel/kernel.cl` (current kernel)
- Read `solution/kernel/baseline.cl` (reference)
- Read `experiments/summary.md` (all experiment history)
- Read `experiments/LESSONS.md` (cross-experiment knowledge)
- Check if `exp_N/plan.md` exists without `result.md` -> folder is RESERVED
- Verify SSH connectivity: `sshpass -p "Hello123" ssh -o StrictHostKeyChecking=no ubuntu@192.168.100.102 "echo OK"`
- If SSH fails: report and pause

## Step 2: PLAN — Plan One Optimization
- If folder reserved (plan.md exists, result.md absent): skip planning, go to Step 3
- Scan summary.md for previously failed attempts — DO NOT repeat
- Consult `opencl_gpu_optimization_guide.md` for technique catalog
- Early iteration priority: correctness -> OpenCL port -> fused kernels -> memory -> TP hardware
- Allowed: anysearch for OpenCL/Adreno documentation
- Write `experiments/exp_(N+1)/plan.md` with:
  - Hypothesis (1 sentence)
  - Expected change description
  - Expected effect on kernel time and transfer time
  - Risk of correctness regression (low/med/high)

## Step 3: IMPLEMENT — Edit kernel.cl
- Edit ONLY `solution/kernel/kernel.cl`
- Single optimization per iteration
- Keep kernel function signatures stable
- Minor benchmark.cpp edits allowed if profiling phases change (rare)
- Write a 1-line commit message summarizing the change

## Step 4: VALIDATE — /benchmark quick
- 2 resolutions (320x240, 1920x1080)
- Verifies: compilation, correctness (max CPU diff < 1e-3), no crashes
- If build fails: fix kernel.cl, re-validate
- If correctness fails: mark as regression, go to Step 6 (LOG) with FAIL status, then Step 7 (DECIDE)

## Step 5: MEASURE — /benchmark stride 2
- 4 resolutions, ~2 minutes
- Check reference latency hasn't drifted >30% from previous best
- Report data per resolution (not just mean)
- If sub-5% difference vs previous best: flag for A/B confirmation

## Step 6: LOG — /log-experiment
- NEVER skip (including failures)
- See `.claude/commands/log-experiment.md` for detailed flow

## Step 7: DECIDE — Quantified Rollback Defense (Layer 3)

Read `experiments/BEST.md` to get current champion metrics.
Compute `delta = (current_total_ms - best_total_ms) / best_total_ms`.

```
┌─────────────────┬──────────────────────────────┬──────────────────────────────────────────────┐
│    Result       │          Threshold            │                   Action                      │
├─────────────────┼──────────────────────────────┼──────────────────────────────────────────────┤
│ Clear win       │ delta < -5% (faster by >5%)   │ ACCEPT: keep + update BEST.md + mark NEW BEST│
│ Marginal        │ |delta| <= 5%                 │ A/B paired test (same device, 3 runs mean)   │
│ Regression      │ delta > +5% (slower by >5%)   │ ROLLBACK: cp exp_{best}/kernel.cl → solution/│
│ Noise suspicion │ reference latency drifts >30% │ Device anomaly → re-run once                 │
└─────────────────┴──────────────────────────────┴──────────────────────────────────────────────┘
```

### ACCEPT (clear win) Procedure
1. Update `experiments/BEST.md`: best_exp=N, best_kernel_ms=X, best_total_ms=Y, best_date=today
2. Copy kernel snapshot as golden copy: `cp solution/kernel/kernel.cl solution/kernel/best_kernel.cl`
3. In `experiments/summary.md`: annotate as "NEW BEST" in Notes column
4. Continue to next experiment on same optimization axis

### MARGINAL (A/B paired test) Procedure
1. Run `/benchmark stride 2` with current kernel → bench_A.log
2. Restore kernel from `experiments/exp_{BEST}/kernel.cl`
3. Run `/benchmark stride 2` with best kernel → bench_B.log
4. Compare per-resolution mean total_ms
5. If A is better on ALL resolutions → treat as ACCEPT
6. If A is worse on ANY resolution → treat as ROLLBACK
7. If mixed (better on some, worse on others) → ROLLBACK (conservative)

### ROLLBACK Procedure
1. Restore: `cp experiments/exp_{BEST}/kernel.cl solution/kernel/kernel.cl`
2. Rebuild: `cd build && cmake ../solution/host && make -j$(nproc)`
3. Verify: `./benchmark --quick` (total_ms must match best within 5%)
4. In `experiments/exp_N/result.md`: add "Status: ROLLBACK — restored to exp_{BEST}"
5. In `experiments/LESSONS.md`: record this direction as dead end
6. In `experiments/summary.md`: annotate "ROLLBACK" in Notes column
7. Next PLAN must choose a DIFFERENT optimization direction

### Plateau Detection
5+ consecutive experiments within ±5% of each other → TRIGGER Research Agent.
Research Agent reads ONLY from disk (summary.md, LESSONS.md, profile.md) — clean context, no optimizer bias.

## Step 8: BUDGET
- Default: /benchmark stride 2 (~2 min)
- /benchmark full (~8 min) only for: confirming new best, or every 5 iterations
- If experiment count > 50: increase default stride to 4

## Pause Checkpoints
Before executing any of these direction changes, pause and ask user:
- Switching from buffer-based to image2d_t-based pipeline
- Introducing local memory tiling (reduces occupancy)
- Changing kernel I/O interface
- Adding a new kernel that changes the pipeline structure
