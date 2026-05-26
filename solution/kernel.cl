// ============================================================
// kernel.cl — Exp 5: image2d_t Hardware Bilinear via Texture Processor
// Optimization: Replace 4-tap software bilinear with single
//               read_imagef(sampler, CLK_FILTER_LINEAR) instruction
//               executed by Adreno Texture Processor (TP) hardware.
// Expected: ~40% kernel time reduction vs buffer-based baseline.
// ============================================================

// Hardware bilinear resize + normalize via image2d_t
// read_imagef with CLK_FILTER_LINEAR: single TP instruction replaces
// 4 source reads + 3 weight multiplications + 3 additions per channel.
// NDRange: 2D over (dst_w, dst_h) = (448, 448) = 200,704 work-items
__constant sampler_t bilinear_sampler = CLK_NORMALIZED_COORDS_FALSE
                                       | CLK_ADDRESS_CLAMP_TO_EDGE
                                       | CLK_FILTER_LINEAR;

__kernel void resize_bilinear_normalized(
    __read_only image2d_t src_image,
    __global float* dst,
    int src_w,
    int src_h,
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

    // Source coordinate in image-sampler space
    // CLK_NORMALIZED_COORDS_FALSE: pixel (i,j) center is at (i+0.5, j+0.5)
    // align_corners=False: pixel-center → pixel-center mapping
    float sx = (dx + 0.5f) * (float)src_w / (float)dst_w;
    float sy = (dy + 0.5f) * (float)src_h / (float)dst_h;

    // Single TP instruction: hardware bilinear interpolation
    // CL_FLOAT format: pixel values already in [0.0, 1.0]
    float4 pixel = read_imagef(src_image, bilinear_sampler, (float2)(sx, sy));

    // Normalize: pixel is already in [0,1], no /255 needed
    float r = (pixel.x - mean_r) * inv_std_r;
    float g = (pixel.y - mean_g) * inv_std_g;
    float b = (pixel.z - mean_b) * inv_std_b;

    // Write float32 normalized pixel
    int out_idx = (dy * dst_w + dx) * 3;
    dst[out_idx]   = r;
    dst[out_idx+1] = g;
    dst[out_idx+2] = b;
}

// Transpose normalized image [T, H, W, C] -> patches [seq_len, input_dim]
// (Unchanged from baseline — only resize kernel uses image2d_t optimization)
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
