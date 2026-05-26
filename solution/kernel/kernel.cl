// ============================================================
// kernel.cl — OpenCL Image Preprocessing Kernel
// Agent-editable file for /optimize loop
// ============================================================
// I/O Contract (DO NOT CHANGE without updating benchmark.cpp):
//   Input:  __global const uchar* src_rgb24  [src_h * src_stride]
//   Output: __global float* dst_patches      [seq_len * input_dim]
//   Config: src_w, src_h, src_stride, dst_w(448), dst_h(448)
//           mean_r/g/b, inv_std_r/g/b
//           grid_t(1), grid_h(28), grid_w(28), merge_size(2)
//           input_dim(1536), seq_len(784)
// ============================================================

// Fused resize + normalize kernel (v1 — baseline GPU port)
// NDRange: 2D over (dst_w, dst_h) = (448, 448) = 200,704 work-items
__kernel void resize_bilinear_normalized(
    __global const uchar* src,
    int src_w,
    int src_h,
    int src_stride,
    __global float* dst,
    int dst_w,
    int dst_h,
    float mean_r,
    float mean_g,
    float mean_b,
    float inv_std_r,
    float inv_std_g,
    float inv_std_b)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    if (dx >= dst_w || dy >= dst_h) return;

    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;
    float sx = (dx + 0.5f) * scale_x - 0.5f;
    float sy = (dy + 0.5f) * scale_y - 0.5f;

    int ix0 = clamp((int)sx, 0, src_w - 1);
    int iy0 = clamp((int)sy, 0, src_h - 1);
    int ix1 = clamp(ix0 + 1, 0, src_w - 1);
    int iy1 = clamp(iy0 + 1, 0, src_h - 1);

    float fx = sx - (float)ix0;
    float fy = sy - (float)iy0;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w01 = fx * (1.0f - fy);
    float w10 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    int s3 = src_stride;
    int o00 = iy0 * s3 + ix0 * 3;
    int o01 = iy0 * s3 + ix1 * 3;
    int o10 = iy1 * s3 + ix0 * 3;
    int o11 = iy1 * s3 + ix1 * 3;

    float r = w00 * (float)src[o00]   + w01 * (float)src[o01]
            + w10 * (float)src[o10]   + w11 * (float)src[o11];
    float g = w00 * (float)src[o00+1] + w01 * (float)src[o01+1]
            + w10 * (float)src[o10+1] + w11 * (float)src[o11+1];
    float b = w00 * (float)src[o00+2] + w01 * (float)src[o01+2]
            + w10 * (float)src[o10+2] + w11 * (float)src[o11+2];

    r = (r / 255.0f - mean_r) * inv_std_r;
    g = (g / 255.0f - mean_g) * inv_std_g;
    b = (b / 255.0f - mean_b) * inv_std_b;

    int out_idx = (dy * dst_w + dx) * 3;
    dst[out_idx]   = r;
    dst[out_idx+1] = g;
    dst[out_idx+2] = b;
}

// Transpose normalized image [T, H, W, C] -> patches [seq_len, input_dim]
// NDRange: 1D over seq_len = 784, local_work_size = 64
__kernel void transpose_to_patch(
    __global const float* norm_img,
    __global float* patches,
    int grid_t,
    int grid_h,
    int grid_w,
    int merge_size,
    int channels,
    int input_dim)
{
    int seq_idx = get_global_id(0);
    if (seq_idx >= grid_t * grid_h * grid_w * merge_size * merge_size) return;

    int remaining = seq_idx;
    int mw_idx     = remaining % merge_size;  remaining /= merge_size;
    int mh_idx     = remaining % merge_size;  remaining /= merge_size;
    int w_block    = remaining % grid_w;       remaining /= grid_w;
    int h_block    = remaining % grid_h;       remaining /= grid_h;
    int t_idx      = remaining;

    int patch_dim = channels * merge_size * merge_size;

    for (int c = 0; c < channels; c++) {
        for (int mh = 0; mh < merge_size; mh++) {
            for (int mw = 0; mw < merge_size; mw++) {
                int src_h = h_block * merge_size + mh;
                int src_w = w_block * merge_size + mw;
                int src_idx = ((t_idx * grid_h * merge_size + src_h)
                               * grid_w * merge_size + src_w) * channels + c;
                int dst_idx = c * merge_size * merge_size + mh * merge_size + mw;
                patches[seq_idx * input_dim + dst_idx] = norm_img[src_idx];
            }
        }
    }
}
