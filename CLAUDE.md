# CLAUDE.md — Qualcomm OpenCL auto-gpu-kernel Workflow

## Project Identity
- **Name**: opencl_kernel — Autonomous OpenCL GPU kernel optimization for Qualcomm Adreno
- **Platform**: Qualcomm QCS9075, Adreno 663 GPU (A6x scalar architecture)
- **Language**: OpenCL C 3.0 embedded in C++17 host
- **Remote**: IQ9 Device at ubuntu@192.168.100.102 (password: Hello123)
- **Repository**: https://github.com/J12-Kyrie/opencl_kernel

## Non-Negotiable Rules

### 1. Stay in OpenCL
Don't switch to GLSL, Vulkan Compute, or CPU NEON assembly.
OpenCL compilation errors are almost always user error — fix the kernel, don't flee the language.

### 2. Absolute Latencies Only
Report microseconds (us) and milliseconds (ms) from clGetEventProfilingInfo.
Never use speedup ratios — reference implementation can vary 20-30% across runs.
A 0.00x regression is a valid measurement.

### 3. One Optimization Per Iteration
Single variable change. No multi-tactic bundles.
Sub-5% differences require A/B paired benchmark (same device, same session).

### 4. No GPU Locally — All Build/Benchmark via SSH on IQ9 Device
All timing data comes from the IQ9 device. No local emulation.

### 5. Log Every Experiment
Including failures. Failed experiments are the most valuable data — they prevent repetition.
Never skip /log-experiment. Never overwrite an existing result.md.

### 6. Never Stop the Loop
Only the user can terminate. Adaptive slowdown: intervals can increase, loop continues.

### 7. No Benchmark Gaming
No pre-warmed queues, no buffer reuse tricks, no hiding transfer time. All measurements must reflect real per-frame cost.

### 8. anysearch Allowed for Research
This workflow allows anysearch for OpenCL spec lookups, Adreno optimization references, and Qualcomm documentation.

### 9. Semi-Autonomous — Pause for Major Pivots
Auto-proceed within an optimization axis. Pause and report for major direction changes (e.g., buffer -> image2d_t, introducing local memory tiling, rewriting kernel architecture).

## Hardware Context (QCS9075 / Adreno 663)

| Property | Value |
|----------|-------|
| GPU | Adreno 663 (A6x family) |
| Architecture | Scalar (not SIMD), wavefront scheduling |
| Wavefront size | 32 (half) / 64 (full) |
| Local memory | 32 KB per CU (physical) |
| Max work-group size | 1024 |
| Memory bandwidth | ~25-35 GB/s (LPDDR5) |
| Texture Processor | Hardware bilinear, HOF/SAD/SSD extensions |
| OpenCL version | 3.0 with cl_qcom_* extensions |
| Build flags | -cl-fast-relaxed-math -cl-mad-enable |

## Kernel I/O Specification
- Input: uchar RGB24 image (src_height x src_width x 3)
- Output: float32 patches [seq_len x input_dim]
- Fixed target: 448x448, seq_len=784, input_dim=1536
- Normalization: ImageNet mean/std
- Temporal: T=2 replicate
- RoPE: Precomputed lookup tables (not in kernel scope)

## Numerical Traps
- Float32 accumulator needed for bilinear weights
- (val/255.0 - mean) * inv_std: compute division before subtraction
- Bilinear weights must sum to exactly 1.0
- Check output diff against CPU < 1e-4

## Repo Layout
Agent edits ONLY: `solution/kernel/kernel.cl`
Read-only reference: `solution/kernel/baseline.cl`

## Experiment Folder Structure
Each exp_N/: plan.md (opt) + kernel.cl (snapshot) + bench.log + result.md
Folder reservation: if plan.md exists without result.md, complete that exp first.

## Reference Resources
- opencl_gpu_optimization_guide.md (comprehensive reference)
- .claude/skills/opencl-kernel-dev/SKILL.md
- .claude/skills/opencl-kernel-debug/SKILL.md
- /mnt/workspace/develop/qwen3vit/src/qwen3vit_deploy/ (existing IQ9 kernels)
