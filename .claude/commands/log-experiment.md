# /log-experiment — Experiment Logging Automation

## Flow

### 1. Pick Experiment Folder
- Find highest N in experiments/exp_N/
- If exp_N/plan.md exists but result.md is absent -> USE exp_N (folder reserved)
- Otherwise -> CREATE exp_(N+1)

### 2. Write Artifacts
```
cp solution/kernel/kernel.cl experiments/exp_N/kernel.cl
cp /tmp/bench.log experiments/exp_N/bench.log   (after benchmark runs)
```

### 3. Write result.md Template
```markdown
# Experiment N: <brief description>
Date: YYYY-MM-DD HH:MM
Status: PASS | FAIL
Hypothesis: <from plan.md>

## Results
| Resolution | Kernel_ms | Total_ms |
|------------|-----------|----------|
| ... | ... | ... |

Best kernel time: X.XXX ms
Best total time: X.XXX ms
Delta vs previous best: +/-X.X%

## Lessons
- <What was learned from this experiment>
- <What to try or avoid next>
```

### 4. Update summary.md Index
Append row to `experiments/summary.md`:
```
| Exp N | Date | Description | Kernel_ms | Total_ms | Pass | Notes |
```

### 5. Update LESSONS.md
If the experiment reveals a cross-experiment insight, append:
```
- [YYYY-MM-DD Exp N] <insight>
```
If an existing LESSON is disproven by this experiment, DELETE it from LESSONS.md.

## Rules
- NEVER skip logging (including failed experiments)
- NEVER overwrite an existing result.md
- Failed experiments are marked "Status: FAIL" with the error reason
- Correctness failures go in Notes column of summary.md
