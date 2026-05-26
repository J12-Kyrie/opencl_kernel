# opencl_kernel — Autonomous GPU Kernel Optimization for Qualcomm Adreno

Autonomous OpenCL GPU kernel optimization framework adapted from [auto-gpu-kernel](https://github.com/Dogacel/auto-gpu-kernel) (MLSys'26 FlashInfer Contest winner) for **Qualcomm QCS9075 (Adreno 663)** embedded platforms.

**Kernel**: Fused resize-bilinear + normalize + transpose-to-patch for VIT image preprocessing.

**Result**: 10 experiments, **4 optimization directions**, final kernel **-20.7% faster** than baseline.

## Architecture

```
                    /optimize (8-step loop)
                    Assess → Plan → Implement →
                    Validate → Measure → Log →
                         Decide → Budget
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    Profiler Agent      Research Agent     Workload Inspector
   (bottleneck finder)  (strategy diagnosis)  (data profiler)
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                     experiments/ (disk artifacts)
                     Clean-context agent communication
```

### 5-Layer Rollback Defense

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| 1 | Single-file constraint (`kernel.cl` only) | One file to revert |
| 2 | 3-level benchmark (quick→stride→full) | Correctness before performance |
| 3 | Quantified decision rules (≥5%=accept, <5%=A/B, regression=revert) | Noise-gate rollback |
| 4 | Experiment snapshots (`exp_N/kernel.cl`) | Physical backups, never overwritten |
| 5 | `LESSONS.md` can be corrected | Bad insights removed, not accumulated |

### Decision Matrix (Step 7: DECIDE)

```
delta = (current_total_ms - best_total_ms) / best_total_ms

delta < -5%   → ACCEPT   → update BEST.md, continue
|delta| ≤ 5%  → MARGINAL → A/B paired test (3 runs, same device)
delta > +5%   → ROLLBACK → cp exp_{BEST}/kernel.cl → solution/kernel.cl
5× plateau    → TRIGGER  → Research Agent (clean-context diagnosis)
```

## Experiment History

| Exp | Direction | Kernel (ms) | Total (ms) | Verdict |
|-----|-----------|------------|------------|---------|
| 1 | Buffer-based baseline | 0.411 | 0.879 | **BASELINE** |
| 5 | image2d_t TP hardware bilinear | 0.090 | 0.579 | **ROLLBACK** — `read_imagef` broken on Adreno 663 |
| 6 | vload4 vectorized loads | 0.247 | 0.724 | Neutral — compiler auto-vectorizes |
| 7 | WG size sweep {32,64,128,256} | 0.244 | 0.719 | Neutral — driver auto-tunes |
| 8 | Transpose loop unroll | 0.311 | 0.603 | **CHAMPION** — -16.0% |
| 9 | vload3 in transpose | 0.310 | 0.584 | **CHAMPION** — -18.7% |
| 10 | Fix 4x work-item redundancy | 0.311 | 0.569 | **FINAL** — -20.7% |

### Adreno 663 Hardware Constraints Discovered

1. `read_imagef` in OpenCL kernels returns garbage — image2d_t path **not viable**
2. Adreno compiler auto-vectorizes uchar loads — manual vload4 has **no effect**
3. Adreno driver auto-tunes work-group size — explicit WG tuning **flat**
4. `clGetEventProfilingInfo` may underreport H2D/D2H transfer time on Adreno

## Quick Start

### Prerequisites
- Qualcomm device with Adreno GPU (tested: QCS9075, Adreno 663)
- OpenCL runtime (`libOpenCL.so`)
- SSH access from development machine

### Setup

```bash
git clone https://github.com/J12-Kyrie/opencl_kernel.git
cd opencl_kernel
cp config.toml.example config.toml
# Edit config.toml: set ssh_host, ssh_user

# Build and verify
bash scripts/run_benchmark.sh --quick

# Start optimization loop (in Claude Code)
/optimize
```

### Manual Benchmark

```bash
# On IQ9 device:
cd /mnt/workspace/opencl_kernel/build
./benchmark --quick          # 2 resolutions, ~30s
./benchmark --stride 2       # 4 resolutions, ~2min (default)
./benchmark --full           # 8 resolutions, ~8min
```

## Directory Structure

```
opencl_kernel/
├── CLAUDE.md                     # Agent constitution (9 non-negotiable rules)
├── config.toml.example           # Device configuration template
├── .claude/
│   ├── commands/
│   │   ├── optimize.md           # /optimize 8-step loop
│   │   ├── benchmark.md          # /benchmark 3-level strategy
│   │   └── log-experiment.md     # Experiment logging automation
│   └── agents/
│       ├── profiler.md           # OpenCL event profiler
│       ├── research.md           # Strategy diagnosis + ceiling analysis
│       └── workload_inspector.md # Image characteristic analyzer
├── scripts/
│   ├── run_benchmark.sh          # SSH build + run + collect
│   ├── fetch_results.sh          # Pull results from device
│   └── pack_solution.py          # Archive for submission
├── solution/
│   ├── kernel/
│   │   ├── kernel.cl             # ★ Agent-editable (single file)
│   │   └── baseline.cl           # CPU reference (read-only)
│   └── host/
│       ├── benchmark.cpp         # Host runner with clGetEventProfilingInfo
│       └── CMakeLists.txt        # Build configuration
└── experiments/
    ├── BEST.md                   # Golden reference for rollback
    ├── summary.md                # Experiment index table
    ├── LESSONS.md                # Cross-experiment knowledge
    ├── CONVERGENCE.md            # Final convergence report
    ├── HOST_OPTIMIZATIONS.md     # Host-level optimization roadmap
    └── exp_1/ ... exp_10/        # Per-experiment snapshots
```

## Key Adaptations from auto-gpu-kernel

| Original (Triton/CUDA/Modal) | This Framework (OpenCL/Adreno/SSH) |
|------------------------------|-------------------------------------|
| Triton Python kernel | OpenCL C embedded in C++ host |
| NVIDIA GPU (Modal cloud) | Qualcomm Adreno 663 (IQ9 device) |
| CUPTI performance counters | clGetEventProfilingInfo event timing |
| 128-shape dataset | 8-resolution image set (320×240 to 3840×2160) |
| Network-isolated (no web) | anysearch allowed for OpenCL/Adreno docs |
| Full autonomy | Semi-autonomous (pause on major pivots) |

## Performance (Final Kernel — Exp 10)

| Resolution | Total (ms) | vs Baseline |
|------------|-----------|-------------|
| 320×240 | 0.569 | -20.7% |
| 640×480 | 0.430 | — |
| 800×600 | 0.641 | — |
| 1024×768 | 0.651 | — |
| 1280×720 | 0.651 | — |
| 1600×900 | 0.718 | — |
| 1920×1080 | 0.686 | -33.6% |
| 3840×2160 | 0.698 | — |

**GPU**: QUALCOMM Adreno(TM) 663 | Max WG: 1024 | OpenCL 3.0
**Correctness**: All 8 resolutions PASS (max_diff < 1e-5)

## Reference Resources

- [auto-gpu-kernel (original)](https://github.com/Dogacel/auto-gpu-kernel) — MLSys'26 FlashInfer Contest
- [QCS9075 Data Sheet](https://docs.qualcomm.com/bundle/publicresource/topics/80-73417-1)
- [Adreno OpenCL Best Practices](https://dl.acm.org/doi/10.1145/3204919.3204935) — IWOCL 2018
- [Adreno Image Processing Extensions](https://dl.acm.org/doi/10.1145/3204919.3207892) — IWOCL 2018

## License

MIT
