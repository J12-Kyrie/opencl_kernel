// ============================================================
// kernel.cl — Exp 6: float4 vectorized loads (vload4)
// Optimization: Replace 3x uchar scalar reads per corner with
//               single uchar4 vload4 (4 corners × 1 read vs 4 × 3 reads).
// Expected: 5-10% kernel time reduction on Adreno scalar arch.
// ============================================================

// Fused resize + normalize with vload4 vectorized loads
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

    // Exp 6: vload4 for coalesced 4-byte reads per corner
    int s3 = src_stride;
    int o00 = iy0 * s3 + ix0 * 3;
    int o01 = iy0 * s3 + ix1 * 3;
    int o10 = iy1 * s3 + ix0 * 3;
    int o11 = iy1 * s3 + ix1 * 3;

    uchar4 p00 = vload4(0, src + o00);
    uchar4 p01 = vload4(0, src + o01);
    uchar4 p10 = vload4(0, src + o10);
    uchar4 p11 = vload4(0, src + o11);

    float r = w00 * (float)p00.x + w01 * (float)p01.x
            + w10 * (float)p10.x + w11 * (float)p11.x;
    float g = w00 * (float)p00.y + w01 * (float)p01.y
            + w10 * (float)p10.y + w11 * (float)p11.y;
    float b = w00 * (float)p00.z + w01 * (float)p01.z
            + w10 * (float)p10.z + w11 * (float)p11.z;

    r = (r / 255.0f - mean_r) * inv_std_r;
    g = (g / 255.0f - mean_g) * inv_std_g;
    b = (b / 255.0f - mean_b) * inv_std_b;

    int out_idx = (dy * dst_w + dx) * 3;
    dst[out_idx]   = r;
    dst[out_idx+1] = g;
    dst[out_idx+2] = b;
}

// Transpose: [T, H, W, C] -> patches [seq_len, input_dim]
// Exp 8: loop unrolling + precomputed base offsets
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
    int total = grid_t * grid_h * grid_w * merge_size * merge_size;
    if (seq_idx >= total) return;

    // 6D coordinate decode
    int remaining = seq_idx;
    int mw_idx  = remaining & 1;  remaining >>= 1;  // merge_size=2
    int mh_idx  = remaining & 1;  remaining >>= 1;
    int w_block = remaining % grid_w;  remaining /= grid_w;
    int h_block = remaining % grid_h;  remaining /= grid_h;
    int t_idx   = remaining;  // grid_t=1 → always 0

    // Precompute source base offset (common to all c/mh/mw combinations)
    int merge_hw = merge_size * merge_size;  // =4
    int grid_merge_w = grid_w * merge_size;  // =56
    int row_stride = grid_merge_w * channels;  // =168
    int frame_base = t_idx * grid_h * merge_size * row_stride;
    int block_base = frame_base + h_block * merge_size * row_stride + w_block * merge_size * channels;

    // Destination base
    __global float* dst_base = patches + seq_idx * input_dim;

    // Manually unrolled: channels=3, merge=2 → 12 assignments
    int src_h0 = block_base;
    int src_h1 = block_base + row_stride;

    // Channel 0
    dst_base[0] = norm_img[src_h0];
    dst_base[1] = norm_img[src_h0 + channels];
    dst_base[2] = norm_img[src_h1];
    dst_base[3] = norm_img[src_h1 + channels];
    // Channel 1
    dst_base[4] = norm_img[src_h0 + 1];
    dst_base[5] = norm_img[src_h0 + channels + 1];
    dst_base[6] = norm_img[src_h1 + 1];
    dst_base[7] = norm_img[src_h1 + channels + 1];
    // Channel 2
    dst_base[8]  = norm_img[src_h0 + 2];
    dst_base[9]  = norm_img[src_h0 + channels + 2];
    dst_base[10] = norm_img[src_h1 + 2];
    dst_base[11] = norm_img[src_h1 + channels + 2];
}
