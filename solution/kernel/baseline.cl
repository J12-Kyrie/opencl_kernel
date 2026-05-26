// ============================================================
// baseline.cl — CPU Reference Implementation (READ-ONLY)
// ============================================================
// CPU reference algorithm:
// 1. stbir_resize_uint8_srgb() -> scale to 448x448
// 2. For each pixel: normalize = (val/255.0 - mean) / std
// 3. Temporal replicate: copy frame 0 -> frame 1
// 4. Transpose: [T, H, W, C] -> [seq_len, input_dim]
// 5. RoPE: memcpy from precomputed tables (out of kernel scope)
//
// Expected output tolerance: max absolute diff < 1e-4 (float32)
