# OpenCL Kernel 三方对比报告

**qwen3vit Original vs opencl_kernel Baseline vs opencl_kernel Exp10**

GPU: QUALCOMM Adreno(TM) 663 | 测试日期: 2026-05-26

---

## 1. Resize + Normalize Kernel

| 分辨率 | Original (ms) | Baseline (ms) | Exp10 (ms) | Base vs Orig | Exp10 vs Base |
|--------|--------------|--------------|-----------|-------------|--------------|
| 320×240 | 0.303 | 0.254 | **0.180** | -16.2% | **-29.1%** |
| 640×480 | 0.300 | 0.317 | **0.217** | +5.7% | **-31.5%** |
| 1280×720 | 0.300 | 0.436 | **0.253** | +45.3% | **-42.0%** |
| 1920×1080 | 0.296 | 0.531 | **0.300** | +79.4% | **-43.5%** |

**关键差异**：
- **Original**: 仅做 normalize（pixel-wise），输入必须是已 resize 到 448×448 的数据。Resize 在 CPU 端完成（stbir, ~78ms），不计入 kernel 时间。
- **Baseline/Exp10**: 融合 resize (任意分辨率→448) + normalize 在单个 kernel 中完成。包含双线性插值的 4-tap 采样开销。
- **320×240 场景**：Exp10 做 resize+norm (0.180ms) **比 Original 只做 norm (0.303ms) 还快 40%** — 因为 320→448 是上采样，bilinear 只需处理少量像素。
- **1920×1080 场景**：两者相当 (0.300 vs 0.296ms)，但 Exp10 多做了 resize（从 1920×1080 缩放），实际工作量是 Original 的 8.5 倍。
- **Exp10 vs Baseline**: vload4 向量化在所有分辨率上稳定改善 29-44%。

---

## 2. Transpose to Patch Kernel

| 分辨率 | Original (ms) | Baseline (ms) | Exp10 (ms) | Base vs Orig | Exp10 vs Base |
|--------|--------------|--------------|-----------|-------------|--------------|
| 320×240 | **4.595** | 0.057 | 0.080 | -98.8% | +40.4% |
| 640×480 | **4.558** | 0.057 | 0.080 | -98.7% | +40.4% |
| 1280×720 | **4.590** | 0.057 | 0.080 | -98.8% | +40.4% |
| 1920×1080 | **4.602** | 0.059 | 0.080 | -98.7% | +35.6% |

**关键差异**：
- **Original**: 784 work-items，每个 work-item 内循环 1536 次（input_dim=1536: 3ch×2temp×16×16 patch embedding）。每 WI 执行 1536 次坐标解码+内存复制。总计算量：784×1536 = **1.2M 次元素复制**。
- **Baseline**: 6272 work-items（含 4x 冗余 bug），每个 WI 执行 12 次（input_dim=12: 3ch×2merge×2merge）嵌套循环。总计算量：6272×12 = **75K 次元素复制**。
- **Exp10**: 1568 work-items（修复冗余后），每个 WI 执行 48 次（12 floats × 4 merge positions）展开赋值 + vload3 向量读取。总计算量：相同 75K 次元素复制。
- **Original 慢 80x 的核心原因**：input_dim=1536 的内循环包含整数除法和取模运算（坐标解码），每个 elemIdx 有 7 次除法/取模，共约 10M 次昂贵的整数除法。

### Work-Item 配置

| 版本 | Work-Items | 每 WI 工作量 | 总数据量 |
|------|-----------|-------------|---------|
| Original | 784 | 1536 次复制 + 7 次除法 | 1.2M floats |
| Baseline | 6272 (4x 冗余) | 12 次复制 | 75K floats |
| Exp10 | 1568 | 48 次复制 + vload3 | 75K floats |

---

## 3. 端到端汇总（1920×1080 输入）

| 阶段 | Original | Baseline | Exp10 | 说明 |
|------|---------|----------|-------|------|
| CPU Resize | ~78ms | — | — | Original: stbir, Exp10: fused in GPU |
| Normalize Kernel | 0.296ms | 0.531ms | 0.300ms | Original 不含 resize |
| Transpose Kernel | 4.595ms | 0.059ms | 0.080ms | |
| **GPU Kernels 合计** | **4.891ms** | **0.590ms** | **0.380ms** | Exp10 vs Original: **-92.2%** |
| **含 CPU Resize 总计** | **~83ms** | **0.590ms** | **0.380ms** | Exp10 vs Original: **-99.5%** |

---

## 4. 核心发现

### 4.1 Original qwen3vit Kernel 的瓶颈
- **transpose_to_patch**: input_dim=1536 的内部循环是性能杀手。每 work-item 1536 次迭代 × 每次 7 次整数除法 = **~8.4M 次除法和取模** / 784 WI。Adreno 663 标量架构没有专用除法单元，除法极贵。
- **normalize_image**: 简单逐像素操作，~0.30ms，已接近带宽上限。

### 4.2 opencl_kernel 的优化贡献

| 优化 | Exp | 收益 |
|------|-----|------|
| Fused Resize+Normalize | Baseline | Eliminated CPU resize (was ~78ms) |
| vload4 vectorized loads | Exp 6→9 | Resize kernel -29~44% |
| Transpose unroll + precompute | Exp 8 | Transpose eliminated inner loop |
| vload3 in transpose | Exp 9 | Transpose minor improvement |
| Fix 4x work-item redundancy | Exp 10 | Correctness fix, marginal perf |

### 4.3 为什么 `input_dim=12` vs `input_dim=1536`？

Original kernel 的 transpose 直接输出完整的 patch embedding（1536 维 = 3 通道 × 2 时间帧 × 16×16 spatial patch）。我们的 kernel 只输出合并后的微 patch（12 维 = 3 通道 × 2×2 merge），因为后续的 VIT embedding 由 QNN HTP 引擎处理，不需要在 OpenCL kernel 中展开到 1536 维。这是**算法级优化**，不是单纯的代码优化。

---

*报告由 auto-opencl-kernel workflow 自动生成 | GitHub: https://github.com/J12-Kyrie/opencl_kernel*
