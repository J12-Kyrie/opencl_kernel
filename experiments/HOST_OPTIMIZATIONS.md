# Host-Level Optimizations (Beyond kernel.cl)

## Double-Buffering Throughput Analysis

Double-buffering overlaps H2D upload with GPU compute across consecutive frames:

Single-buffer (sequential):
  Frame N:   [Upload 6.2MB] → [Resize+Norm] → [Copy T=2] → [Transpose] → [Download 4.8MB]
  Frame N+1:                                                 [Upload] → [Compute] → ...
  Effective: 1 frame per total_ms (no overlap)

Double-buffer (streaming):
  Frame N:   [Upload buf_A] → [Compute buf_A] → [Download buf_A]
  Frame N+1:    [Upload buf_B] → [Compute buf_B] → [Download buf_B]
                            ↖ overlapping with buf_A compute ↗
  Effective: max(upload+download, compute) per frame

For 1920x1080 with best kernel (exp_9):
- Sequential: ~0.736ms/frame = ~1359 fps
- Double-buffered theoretical: max(0.013+0.013, 0.321+0.126+0.263) = max(0.026, 0.710) = 0.710ms/frame = ~1408 fps
- Gain: ~3.6% throughput improvement (limited by GPU compute dominating transfer time)

Conclusion: Double-buffering has limited benefit for this workload because
H2D/D2H transfer time (<2% of total) is negligible vs compute time.

## DMA-BUF Zero-Copy Integration

The existing qwen3vit project already has the DMA-BUF path:
- qwen_vl_preprocessor.cpp::preprocessFromFrame()
- veg_combined_runner.cpp --frame mode

This path eliminates H2D entirely (frame already in GPU/shared memory).
Integration with optimized kernel.cl:

1. Copy exp_9/kernel.cl → qwen_vl_opencl_kernels.h (replace OPENCL_KERNEL_SOURCE)
2. Rebuild qwen3vit: cd build && make -j$(nproc)
3. Benchmark: veg-combined-runner --frame /tmp/frame.rgb --frame_width 1920 --frame_height 1080

Expected: Upload→0ms, Download→0ms (QNN consumes GPU buffer directly)
Remaining: Kernel+Copy+Transpose = ~0.710ms per frame
vs original CPU path: ~78ms per frame
Total speedup: ~110x (DMA-BUF GPU path vs CPU path)
