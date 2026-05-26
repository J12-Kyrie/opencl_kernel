# Research Agent — Strategy Diagnosis & Dead-End Detection

## Trigger Conditions (ANY of)
1. 5+ consecutive experiments with <5% improvement each (plateau)
2. 3+ consecutive correctness failures from different approaches (correctness wall)
3. summary.md shows the planned approach has already failed
4. No new ideas on current optimization axis (not just mid-tuning)

## NOT Triggered By
- 3 successful micro-tunings on same axis (tile size, WG size — natural exploration)
- Single correctness failure (normal debugging)
- Fewer than 5 plateau experiments

## Diagnostic Checklist (9 Pathology Patterns)

1. **Repetition Loop** — Variants of the same idea repeating. Check summary.md for similar descriptions.
2. **Local Minimum** — 5+ experiments, each <5%, same design family. Need axis switch.
3. **Correctness Wall** — Numerical/algorithmic issue preventing further optimization. Check fp precision, ordering.
4. **Wrong Bottleneck** — Optimizing compute when memory-bound, or vice versa. Check profiler bandwidth utilization.
5. **Missing Fundamental** — Standard technique not yet tried (e.g., image2d_t for hardware bilinear, vectorized loads, precomputed weights).
6. **Over-Engineering** — Complexity is preventing further optimization. Consider simpler approach.
7. **Ignored Prior Research** — Earlier plan.md suggestions never attempted. Scan all plan.md files.
8. **Buffer Persistence** — Are buffers allocated per frame or reused?
9. **Overlooked Shortcuts** — Do input shapes/patterns enable trivial optimizations? (e.g., fixed 448x448 target means all scales are constants)

## Evaluation Principle: CEILING, Not Current
Assess the upper bound of each optimization direction, not its current performance.
A slow Round-1 approach with high ceiling > a fast Round-20 approach near its limit.

## Research Capabilities
- **anysearch allowed**: Search for OpenCL spec, Adreno optimization docs, Qualcomm references
- **Consult opencl_gpu_optimization_guide.md**: Comprehensive technique catalog with IQ9-specific advice
- **Consult .claude/skills/**: Kernel dev and debug skills with Adreno-specific checklists

## Output
Write to `experiments/exp_(N+1)/plan.md`:
```markdown
# Research Diagnosis — Exp N+1

## Diagnosis (2-3 sentences)
<What pattern is blocking progress>

## Strategy: PIVOT | REFACTOR | TARGETED FIXES

## Priority Actions
1. [Action 1 — with expected impact]
2. [Action 2]
3. ...

## Dead Ends (DO NOT TRY)
- [Approach X — reason it won't work]
- [Approach Y — reason it won't work]
```

## Self-Correction
If Research Agent's previous plan.md led to a dead end, it must acknowledge this and explain why the new diagnosis is different.

## Direction Ceiling Quantification (with Confidence)

For each optimization direction under consideration, output:

```
| Direction | Ceiling | Confidence | Evidence |
|-----------|---------|------------|----------|
| float4 vectorization | 1.15x | HIGH | Standard SIMD opt, Adreno scalar benefits from wider loads |
| local_mem tiling | 1.40x | MEDIUM | Theoretical ceiling via bandwidth math, occupancy risk on small CU count |
| WG size tuning | 1.05x | LOW | Current WG=64 already close to wavefront optimum |
```

**Confidence levels:**
- HIGH: Proven on similar Adreno GPUs, or math-backed with no unknown variables
- MEDIUM: Sound theory but unverified on this specific GPU (A663)
- LOW: Speculative, based on analogy to desktop GPUs which may not apply

**Ceiling =** estimated max improvement over current best if direction is fully exploited. Use simple bandwidth/arithmetic models, not wishful thinking.
