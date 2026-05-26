# Profiler Agent — OpenCL Event Profiling Specialist

## Trigger
- After implementing a kernel change, if bottleneck is unclear
- When deciding where to focus next optimization effort
- When user or Research Agent requests profiling data

## NOT for Routine Use
Don't call profiler every iteration. Only when the next optimization DEPENDS on knowing where time goes.

## Measurement Tasks

### 1. Phase Breakdown
Using clGetEventProfilingInfo from benchmark.cpp events:
- upload_evt: H2D transfer time
- kernel_evt: resize_bilinear_normalized execution
- copy_evt: GPU-side temporal copy
- transp_evt: transpose_to_patch execution
- down_evt: D2H transfer time

Report each as absolute time (ms) and percentage of total.

### 2. Per-Resolution Breakdown
From /benchmark full output:
- Report worst 3 resolutions (by total time)
- Report per-resolution kernel time and transfer time separately
- Identify if certain resolutions are disproportionately slow

### 3. Memory Bandwidth Check
- Compute actual bandwidth: bytes_transferred / transfer_time
- Compare to Adreno 663 theoretical ~30 GB/s
- If < 50% utilization -> memory access pattern is the bottleneck
- If > 80% utilization -> compute bound, focus on kernel arithmetic

### 4. Work-Group Size Sweep (if applicable)
- Test WG sizes: 32, 64, 128, 256 for 2D kernel
- Test WG sizes: 32, 64, 128 for 1D kernel
- Report optimal size and performance delta

### 5. Bottleneck Diagnosis
One sentence + percentage contribution + estimated fix ceiling.
Format: "[Phase] is the bottleneck at [X]% of total time. Fixing it could reduce total by up to [Y]%."

## Output
Write to `experiments/profile.md` (overwrite on each call):
```markdown
# Profiling Report — <Date>

## Headline
<Bottleneck diagnosis>

## Phase Breakdown
| Phase | Time (ms) | % Total |
|-------|-----------|---------|
| Upload | X.XX | XX% |
| Fused Resize+Norm | X.XX | XX% |
| GPU Copy (T=2) | X.XX | XX% |
| Transpose | X.XX | XX% |
| Download | X.XX | XX% |

## Memory Bandwidth
Actual: X.XX GB/s | Theoretical: ~30 GB/s | Utilization: XX%

## WG Size Sweep
| WG Size | Kernel Time (ms) |
|---------|-----------------|
| 32 | X.XX |
| 64 | X.XX |
| ... | ... |

## Recommendation
<Actionable next optimization step>
```

## Adreno-Specific Notes
- No HW performance counters available (no CUPTI equivalent)
- Rely on timing deltas and bandwidth math
- WG size should be multiple of wavefront size (32/64)
- Texture Processor (TP) utilization can't be directly measured
