# LESSONS.md — Cross-Experiment Knowledge

## Proven Insights
- Buffer-based 4-tap software bilinear works correctly on Adreno 663 (max_diff < 1e-6)
- Persistent buffer strategy: allocate once, reuse across frames (exp_1 confirmed)

## Dead Ends
- [2026-05-26 Exp 5] image2d_t + read_imagef: NOT viable on Adreno 663 OpenCL.
  read_imagef in kernel returns garbage (zeros/NaN) for all tested formats:
  CL_FLOAT, CL_UNORM_INT8, CL_UNSIGNED_INT8, CL_BGRA. clEnqueueReadImage from
  host works correctly, proving the image upload is fine. Root cause: Adreno 663
  OpenCL driver does not support in-kernel image sampling via read_imagef.
  Kernel time showed 63% potential improvement (0.241→0.090ms), but correctness
  could not be achieved. Future: revisit if Qualcomm releases driver update.
