# BEST.md — Golden Reference for Rollback

The single source of truth for the best-performing kernel version.
When regression is detected (>5% slower), restore from the snapshot
at experiments/exp_{best_exp}/kernel.cl.

```
best_exp: 1
best_kernel_ms: 0.411
best_total_ms: 0.879
best_date: 2026-05-26
best_commit: 71a6867
```

## Rollback Procedure

1. Read best_exp from this file
2. cp experiments/exp_{best_exp}/kernel.cl solution/kernel/kernel.cl
3. Rebuild: cd build && make -j$(nproc)
4. Verify: ./benchmark --quick (must match best_total_ms within 5%)
5. Mark failed exp as ROLLBACK in summary.md

## Rules

- Update BEST.md ONLY when a new champion is confirmed (>=5% improvement on stride 2)
- Never update on marginal gains (<5%) without A/B confirmation
- Never delete old entries — history is append-only
- If BEST.md points to a corrupt experiment, fall back to exp_1 (baseline)
