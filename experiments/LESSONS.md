# LESSONS.md — Cross-Experiment Knowledge

> Updated: 2026-05-26 | Source: 10 experiments on QUALCOMM Adreno(TM) 663

---

## 1. Adreno 663 Hardware Constraints

### read_imagef Broken in OpenCL Kernels (Exp 5)
- `read_imagef(image2d_t, sampler, coord)` returns garbage (zeros/NaN) for ALL tested formats:
  CL_FLOAT, CL_UNORM_INT8, CL_UNSIGNED_INT8, CL_BGRA.
- `clEnqueueReadImage` from HOST works correctly — image upload is fine.
- **Impact**: image2d_t + Texture Processor hardware bilinear path is NOT viable.
  Kernel time showed 63% potential (0.241→0.090ms), but correctness unachievable.
- **Future**: revisit if Qualcomm releases Adreno 663 driver update.

### Adreno Compiler Auto-Vectorizes (Exp 6)
- Manual `vload4(uchar*)` gives ZERO performance benefit vs 3 scalar `src[i]` reads.
- Adreno OpenCL compiler already combines adjacent byte reads into wider loads.
- **Rule**: Don't manually vectorize uchar loads — trust the compiler.

### Adreno Driver Auto-Tunes Work-Group Size (Exp 7)
- Explicit WG sizes {32, 64, 128, 256} all within 1% of each other.
- Passing `NULL` as local_work_size lets the driver pick the optimal WG size.
- **Rule**: Don't tune WG size unless profiling shows a clear problem.

### Scalar Architecture — vload3 on float is Counter-Productive (Exp 10 vs Baseline)
- Adreno 663 is a SCALAR architecture (not SIMD). `vload3(float*)` decomposes to
  3 separate `ld.global.f32` instructions + OpenCL wrapper overhead.
- Scalar `norm_img[idx]` indexing is often FASTER than `vload3`.
- **Rule**: Use scalar memory access on Adreno unless proven otherwise.

### Occupancy Dominates per-WI Efficiency (Exp 10)
- Baseline transpose: 6272 WI → 98 work-groups → 6-12 WG/CU → excellent latency hiding.
- Exp10 transpose: 1568 WI → 25 work-groups → 1.5-3 WG/CU → poor latency hiding.
- Result: Exp10 transpose is 40% SLOWER (0.080ms vs 0.057ms) despite being correct
  and doing less total work.
- **Rule**: On Adreno 663, prioritize occupancy (WI count) over per-WI code quality.
  Memory-bound kernels especially need high WG/CU ratio to hide global memory latency.

---

## 2. Optimization Techniques That WORK

### Fused Kernels (Baseline → Exp 10)
- Fusing resize + normalize into a single kernel eliminates intermediate D2H/H2D round-trips.
- 1920×1080 → 448×448: 0.531ms (fused) vs CPU resize (78ms) + GPU normalize (0.30ms).
- **Rule**: Fuse operations that share the same 2D NDRange iteration space.

### Loop Unrolling (Exp 8)
- Manual unrolling of transpose's triple nested loops (c × mh × mw = 12 iterations)
  eliminated loop overhead and enabled further vectorization experiments.
- Transpose time: 0.320ms → 0.278ms (-13% @ 320×240).

### Persistent GPU Buffers (Baseline)
- Allocating buffers once and reusing across frames avoids `clCreateBuffer` overhead
  (~50-200μs per call due to GPU MMU page table updates).
- Dimension-change detection skips reallocation for fixed-resolution streams.

### Fused Resize+Normalize is the Single Biggest Win
- Moving CPU resize (stbir, ~78ms) into the GPU fused kernel yields >> any kernel-level micro-optimization.
- **Rule**: Look for host↔device boundary crossings before optimizing individual kernels.

---

## 3. Optimization Techniques That DON'T Work on Adreno 663

| Technique | Experiment | Why It Failed |
|-----------|-----------|---------------|
| image2d_t hardware bilinear | Exp 5 | `read_imagef` broken in kernel (driver bug) |
| vload4 uchar vectorization | Exp 6 | Compiler already auto-vectorizes |
| Explicit WG size tuning | Exp 7 | Driver auto-tunes optimally |
| vload3 float vectorization | Exp 10 | Scalar arch — decomposes to 3 scalar loads + overhead |

---

## 4. Work-Item / Occupancy Rules

### The 4x Redundancy Bug (Exp 10 Fix)
- Original `SEQ_LEN = GRID_H * GRID_W * MERGE_SIZE * MERGE_SIZE = 3136` was WRONG.
- Correct: `SEQ_LEN = GRID_H * GRID_W = 784`. The merge dimension (2×2) should be
  iterated WITHIN each work-item, not expanded into the work-item count.
- **Root cause**: 6D coordinate decode treated `(mh, mw)` as work-item-level dimensions
  instead of inner-loop dimensions.

### Minimum Work-Group Count for Adreno 663
- Aim for ≥50 work-groups to achieve good CU occupancy (target: ≥4 WG/CU for ~10 CUs).
- Below ~25 work-groups, memory latency hiding collapses.
- **For transpose-like memory-bound kernels**: prefer more work-items (even with redundancy)
  over fewer work-items with heavier per-WI work.
- **For compute-bound kernels (resize)**: 200K work-items (448×448) naturally provide
  excellent occupancy, so WG size doesn't matter.

### Correctness Sanity Checks
- `vload4` on uchar at `src + o00` reads 4 bytes: RGB of current pixel + R of next pixel.
  The `.w` channel contains adjacent-pixel data — never use `.w` for per-pixel math.
- `vload3` on float reads exactly 3 consecutive floats — no overflow risk.

---

## 5. Experiment Design & Process

### Always Verify with CPU Reference
- Max CPU/GPU diff threshold: `1e-3` for float32 buffer comparison.
- Exp 5 image2d_t kernel: ran FASTER (63% improvement!) but failed correctness.
  Without CPU reference, we would have shipped a broken kernel.
- **Rule**: Correctness before performance. Layer 2 defense caught it.

### One Optimization Per Iteration
- Coupling changes makes root-cause impossible. Exp 8 (loop unroll) and Exp 9 (vload3)
  were separated, making it clear that unrolling was the big win and vload3 was marginal.
- **Rule**: Resist the urge to bundle "obvious" improvements.

### Failed Experiments Are the Most Valuable
- Exp 5 (image2d_t DEAD END) discovered an Adreno 663 driver limitation that no
  amount of kernel optimization could fix. Saved weeks of fruitless effort.
- Exp 6 (vload4 neutral) confirmed compiler capability — important for future kernel design.
- **Rule**: Log EVERY experiment, especially failures. They prevent repetition.

### A/B Paired Testing on Same Device
- When |delta| ≤ 5%, the measurement noise on Adreno 663 can exceed the signal.
  Same-device paired testing (run A, run B, alternate ×3) eliminates thermal drift.
- Exp 9's 3.1% improvement was confirmed with multiple runs showing consistent direction.

---

## 6. Benchmark Harness Design

### clGetEventProfilingInfo May Underreport Transfer Time
- On Adreno 663, H2D/D2H event times (0.01-0.02ms for 6.2MB) are implausibly fast.
  Likely measures API enqueue time, not actual DMA completion.
- **Workaround**: Use TOTAL wall-clock time as the primary metric; use event times
  only for relative kernel-to-kernel comparisons.

### GPU Temperature Monitoring
- IQ9 thermal zones: `/sys/class/thermal/thermal_zone*/temp` (millidegrees C).
- Observed range during benchmarking: 35-46°C, delta typically <1°C per run.
- No thermal throttling observed at these temperatures.

### Resolution Test Set
- 8 resolutions from 320×240 to 3840×2160 provide good coverage.
- 320×240 identifies small-image edge cases; 1920×1080 is the production target.
- 3840×2160 is useful for stress-testing memory bandwidth limits.

---

## 7. Key Design Decisions (QA-Gated)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Stop condition | 5 consecutive <5% + Research Agent ceiling <1.05x | Prevents infinite loops on converged kernels |
| Exploration mode | Every 10 rounds, branch from intermediate best | Avoids local optima without restarting from baseline |
| Precision budget | Cumulative <5e-4, Pareto: perf_gain%/1% ÷ prec_loss/1e-4 > 2.0 | Binary accept/reject is too coarse |
| Rollback granularity | Restore from previous BEST, not baseline | Preserves valid intermediate improvements |
| Semi-autonomous | Auto within axis, pause on direction change | Balances speed with human oversight |
