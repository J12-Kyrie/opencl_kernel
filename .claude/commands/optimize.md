# /optimize — 8-Step Autonomous OpenCL Kernel Optimization Loop

## Overview
This command drives the continuous optimization of `solution/kernel/kernel.cl`.
It follows an 8-step cycle: Assess -> Plan -> Implement -> Validate -> Measure -> Log -> Decide -> Budget.

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

## Step 7: DECIDE
- >=5% improvement in kernel time or total time -> KEEP, mark as new best
- <5% but positive -> A/B paired test to confirm
- Regression (worse than previous best) -> REVERT kernel.cl to previous best
- 5 consecutive iterations with <5% improvement -> TRIGGER Research Agent
- Correctness failure that can't be fixed in 2 attempts -> REVERT

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
