// ============================================================
// benchmark.cpp — OpenCL Kernel Benchmark Runner
// Build: cmake ../solution/host && make -j$(nproc)
// Run:   ./benchmark [--quick|--stride N|--full] [--output bench.log]
// ============================================================

#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <sstream>

// ---- Resolution Test Set ----
struct Resolution {
    int w, h;
    const char* name;
};
static const Resolution RES_SET[] = {
    {320,  240,  "320x240"},
    {640,  480,  "640x480"},
    {800,  600,  "800x600"},
    {1024, 768,  "1024x768"},
    {1280, 720,  "1280x720"},
    {1600, 900,  "1600x900"},
    {1920, 1080, "1920x1080"},
    {3840, 2160, "3840x2160"},
};
static const int RES_COUNT = sizeof(RES_SET) / sizeof(RES_SET[0]);

// ---- Fixed Parameters ----
static const int DST_W = 448, DST_H = 448;
static const int GRID_T = 1, GRID_H = 28, GRID_W = 28;
static const int MERGE_SIZE = 2, CHANNELS = 3;
static const int INPUT_DIM = CHANNELS * MERGE_SIZE * MERGE_SIZE;  // 1536
static const int SEQ_LEN = GRID_T * GRID_H * GRID_W * MERGE_SIZE * MERGE_SIZE; // 784
static const float MEAN_R = 0.48145f, MEAN_G = 0.45782f, MEAN_B = 0.40821f;
static const float STD_R = 0.26862f,  STD_G = 0.26130f,  STD_B = 0.27577f;

// ---- Error Checking ----
#define CL_CHECK(call, msg) do { \
    cl_int e = (call); \
    if (e != CL_SUCCESS) { fprintf(stderr, "CL Error %d: %s\n", e, msg); return false; } \
} while(0)

// ---- Helper: Read kernel source file ----
static std::string readFile(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open: %s\n", path); return ""; }
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// ---- Helper: Read GPU temperature from IQ9 thermal zones ----
static float readGpuTemp() {
    const char* zones[] = {
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone5/temp",
    };
    for (const char* z : zones) {
        std::ifstream f(z);
        if (f.is_open()) {
            int temp_mc;
            f >> temp_mc;
            return temp_mc / 1000.0f;
        }
    }
    return -1.0f;
}

// ---- OpenCL Initialization ----
static bool initOpenCL(cl_platform_id* plat, cl_device_id* dev,
                       cl_context* ctx, cl_command_queue* q) {
    cl_platform_id platforms[4]; cl_uint nplat;
    CL_CHECK(clGetPlatformIDs(4, platforms, &nplat), "getPlatformIDs");
    if (nplat == 0) { fprintf(stderr, "No OpenCL platforms\n"); return false; }
    *plat = platforms[0];

    cl_device_id devices[4]; cl_uint ndev;
    CL_CHECK(clGetDeviceIDs(*plat, CL_DEVICE_TYPE_GPU, 4, devices, &ndev), "getDeviceIDs(GPU)");
    if (ndev == 0) {
        CL_CHECK(clGetDeviceIDs(*plat, CL_DEVICE_TYPE_DEFAULT, 4, devices, &ndev), "getDeviceIDs(DEFAULT)");
    }
    if (ndev == 0) { fprintf(stderr, "No OpenCL devices\n"); return false; }
    *dev = devices[0];

    // Print device info
    char name[256]; size_t wgs;
    clGetDeviceInfo(*dev, CL_DEVICE_NAME, 256, name, NULL);
    clGetDeviceInfo(*dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &wgs, NULL);
    fprintf(stdout, "Device: %s | Max WG: %zu\n", name, wgs);

    cl_int err;
    *ctx = clCreateContext(NULL, 1, dev, NULL, NULL, &err);
    CL_CHECK(err, "createContext");

    cl_queue_properties q_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    *q = clCreateCommandQueueWithProperties(*ctx, *dev, q_props, &err);
    CL_CHECK(err, "createCommandQueue");
    return true;
}

// ---- Build OpenCL Program ----
static bool buildProgram(cl_context ctx, cl_device_id dev, cl_program* prog,
                         const char* kernel_path) {
    std::string src = readFile(kernel_path);
    if (src.empty()) return false;
    const char* csrc = src.c_str();
    size_t slen = src.size();
    cl_int err;
    *prog = clCreateProgramWithSource(ctx, 1, &csrc, &slen, &err);
    CL_CHECK(err, "createProgram");

    err = clBuildProgram(*prog, 1, &dev,
        "-cl-fast-relaxed-math -cl-mad-enable -Werror", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(*prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        std::vector<char> log(log_sz + 1);
        clGetProgramBuildInfo(*prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), NULL);
        fprintf(stderr, "Build error:\n%s\n", log.data());
        return false;
    }
    return true;
}

// ---- CPU Reference: Bilinear Resize + Normalize ----
static void cpuBaseline(const std::vector<unsigned char>& src, int sw, int sh, int sstride,
                         std::vector<float>& dst, int dw, int dh) {
    dst.resize(dw * dh * 3);
    float sx_scale = (float)sw / dw, sy_scale = (float)sh / dh;
    for (int dy = 0; dy < dh; dy++) {
        for (int dx = 0; dx < dw; dx++) {
            float sx = (dx + 0.5f) * sx_scale - 0.5f;
            float sy = (dy + 0.5f) * sy_scale - 0.5f;
            int ix0 = std::max(0, std::min((int)sx, sw - 1));
            int iy0 = std::max(0, std::min((int)sy, sh - 1));
            int ix1 = std::min(ix0 + 1, sw - 1);
            int iy1 = std::min(iy0 + 1, sh - 1);
            float fx = sx - ix0, fy = sy - iy0;
            float w00 = (1-fx)*(1-fy), w01 = fx*(1-fy), w10 = (1-fx)*fy, w11 = fx*fy;
            int s3 = sstride;
            int o00 = iy0*s3 + ix0*3, o01 = iy0*s3 + ix1*3;
            int o10 = iy1*s3 + ix0*3, o11 = iy1*s3 + ix1*3;
            float r = w00*src[o00] + w01*src[o01] + w10*src[o10] + w11*src[o11];
            float g = w00*src[o00+1] + w01*src[o01+1] + w10*src[o10+1] + w11*src[o11+1];
            float b = w00*src[o00+2] + w01*src[o01+2] + w10*src[o10+2] + w11*src[o11+2];
            int out = (dy*dw + dx)*3;
            dst[out]   = (r/255.f - MEAN_R) / STD_R;
            dst[out+1] = (g/255.f - MEAN_G) / STD_G;
            dst[out+2] = (b/255.f - MEAN_B) / STD_B;
        }
    }
}

// ---- Main Benchmark ----
int main(int argc, char** argv) {
    // Parse mode
    const char* kernel_path = KERNEL_PATH;
    int mode = 2;  // default stride 2
    const char* output_file = nullptr;
    int custom_stride = 2;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) mode = 0;
        else if (!strcmp(argv[i], "--stride")) { mode = 1; if (i+1<argc) custom_stride = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--full")) mode = 2;
        else if (!strcmp(argv[i], "--output") && i+1<argc) output_file = argv[++i];
    }

    // Select resolution indices
    std::vector<int> res_idx;
    if (mode == 0) { res_idx = {0, 6}; }  // quick: min + max
    else if (mode == 1) {
        for (int i = 0; i < RES_COUNT; i += custom_stride) res_idx.push_back(i);
    } else {
        for (int i = 0; i < RES_COUNT; i++) res_idx.push_back(i);
    }

    // Init OpenCL
    cl_platform_id plat; cl_device_id dev; cl_context ctx; cl_command_queue queue; cl_program prog;
    if (!initOpenCL(&plat, &dev, &ctx, &queue)) return 1;
    if (!buildProgram(ctx, dev, &prog, kernel_path)) return 1;

    cl_int err;
    cl_kernel k_resize = clCreateKernel(prog, "resize_bilinear_normalized", &err);
    CL_CHECK(err, "createKernel resize");
    cl_kernel k_transpose = clCreateKernel(prog, "transpose_to_patch", &err);
    CL_CHECK(err, "createKernel transpose");

    // Open output file
    FILE* out = output_file ? fopen(output_file, "w") : stdout;
    fprintf(out, "# OpenCL Kernel Benchmark\n");
    fprintf(out, "# Timestamp: %lld\n", (long long)std::chrono::system_clock::now().time_since_epoch().count());
    fprintf(out, "# Mode: %s | Resolutions: %zu\n",
            mode == 0 ? "quick" : (mode == 1 ? "stride" : "full"), res_idx.size());
    float temp_before = readGpuTemp();
    fprintf(out, "# GPU Temp Before: %.1fC\n", temp_before);
    fprintf(out, "%-12s %10s %10s %10s %10s %10s %10s %10s\n",
            "Resolution", "Upload_ms", "Kernel_ms", "Copy_ms", "Transp_ms", "Down_ms", "Total_ms", "Stdev_ms");

    bool all_pass = true;
    for (int ri : res_idx) {
        int sw = RES_SET[ri].w, sh = RES_SET[ri].h;
        int sstride = sw * 3;  // tight stride for synthetic test
        size_t src_bytes = (size_t)sh * sstride;
        size_t norm_bytes = (size_t)DST_H * DST_W * CHANNELS * sizeof(float);

        // Generate synthetic image data (gradient pattern for testing)
        std::vector<unsigned char> src_data(src_bytes);
        for (int y = 0; y < sh; y++)
            for (int x = 0; x < sw; x++) {
                int i = y * sstride + x * 3;
                src_data[i]   = (unsigned char)((x * 255) / sw);
                src_data[i+1] = (unsigned char)((y * 255) / sh);
                src_data[i+2] = (unsigned char)(128);
            }

        // Create device buffers
        cl_int cerr;
        cl_mem src_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, src_bytes, NULL, &cerr);
        CL_CHECK(cerr, "createBuffer src");
        cl_mem norm_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, norm_bytes, NULL, &cerr);
        CL_CHECK(cerr, "createBuffer norm");
        cl_mem copy_norm_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, norm_bytes, NULL, &cerr);
        CL_CHECK(cerr, "createBuffer norm_copy");
        size_t patch_bytes = (size_t)SEQ_LEN * INPUT_DIM * sizeof(float);
        cl_mem patch_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, patch_bytes, NULL, &cerr);
        CL_CHECK(cerr, "createBuffer patches");

        float inv_sr = 1.f/STD_R, inv_sg = 1.f/STD_G, inv_sb = 1.f/STD_B;

        // Set kernel args (persist across enqueues)
        clSetKernelArg(k_resize, 1, sizeof(int), &sw);
        clSetKernelArg(k_resize, 2, sizeof(int), &sh);
        clSetKernelArg(k_resize, 3, sizeof(int), &sstride);
        clSetKernelArg(k_resize, 5, sizeof(int), &DST_W);
        clSetKernelArg(k_resize, 6, sizeof(int), &DST_H);
        clSetKernelArg(k_resize, 7, sizeof(float), &MEAN_R);
        clSetKernelArg(k_resize, 8, sizeof(float), &MEAN_G);
        clSetKernelArg(k_resize, 9, sizeof(float), &MEAN_B);
        clSetKernelArg(k_resize, 10, sizeof(float), &inv_sr);
        clSetKernelArg(k_resize, 11, sizeof(float), &inv_sg);
        clSetKernelArg(k_resize, 12, sizeof(float), &inv_sb);

        int gt = 2, gh = GRID_H, gw = GRID_W, ms = MERGE_SIZE, ch = CHANNELS, idim = INPUT_DIM;
        clSetKernelArg(k_transpose, 2, sizeof(int), &gt);
        clSetKernelArg(k_transpose, 3, sizeof(int), &gh);
        clSetKernelArg(k_transpose, 4, sizeof(int), &gw);
        clSetKernelArg(k_transpose, 5, sizeof(int), &ms);
        clSetKernelArg(k_transpose, 6, sizeof(int), &ch);
        clSetKernelArg(k_transpose, 7, sizeof(int), &idim);

        size_t g2[] = {(size_t)DST_W, (size_t)DST_H};
        size_t g1[] = {(size_t)SEQ_LEN * 2};
        size_t l1[] = {64};

        // Warmup iterations
        for (int w = 0; w < 10; w++) {
            clEnqueueWriteBuffer(queue, src_buf, CL_FALSE, 0, src_bytes, src_data.data(), 0, NULL, NULL);
            clSetKernelArg(k_resize, 0, sizeof(cl_mem), &src_buf);
            clSetKernelArg(k_resize, 4, sizeof(cl_mem), &norm_buf);
            clEnqueueNDRangeKernel(queue, k_resize, 2, NULL, g2, NULL, 0, NULL, NULL);

            clSetKernelArg(k_transpose, 0, sizeof(cl_mem), &copy_norm_buf);
            clSetKernelArg(k_transpose, 1, sizeof(cl_mem), &patch_buf);
            clEnqueueCopyBuffer(queue, norm_buf, copy_norm_buf, 0, 0, norm_bytes, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_transpose, 1, NULL, g1, l1, 0, NULL, NULL);
            clFinish(queue);
        }

        // Timed iterations
        const int ITERS = 100;
        double total_upload = 0, total_kernel = 0, total_copy = 0, total_transp = 0, total_down = 0;
        std::vector<double> iter_times; iter_times.reserve(ITERS);

        for (int iter = 0; iter < ITERS; iter++) {
            cl_event upload_ev, kernel_ev, copy_ev, transp_ev, down_ev;

            clEnqueueWriteBuffer(queue, src_buf, CL_FALSE, 0, src_bytes, src_data.data(), 0, NULL, &upload_ev);

            clSetKernelArg(k_resize, 0, sizeof(cl_mem), &src_buf);
            clSetKernelArg(k_resize, 4, sizeof(cl_mem), &norm_buf);
            clEnqueueNDRangeKernel(queue, k_resize, 2, NULL, g2, NULL, 1, &upload_ev, &kernel_ev);

            clEnqueueCopyBuffer(queue, norm_buf, copy_norm_buf, 0, 0, norm_bytes, 1, &kernel_ev, &copy_ev);

            clSetKernelArg(k_transpose, 0, sizeof(cl_mem), &copy_norm_buf);
            clSetKernelArg(k_transpose, 1, sizeof(cl_mem), &patch_buf);
            clEnqueueNDRangeKernel(queue, k_transpose, 1, NULL, g1, l1, 1, &copy_ev, &transp_ev);

            std::vector<float> patches(SEQ_LEN * 2 * INPUT_DIM);
            clEnqueueReadBuffer(queue, patch_buf, CL_TRUE, 0, patch_bytes, patches.data(), 1, &transp_ev, &down_ev);

            cl_ulong start, end;
            auto get_ms = [&](cl_event ev) -> double {
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                return (end - start) / 1e6;
            };
            double u = get_ms(upload_ev), k = get_ms(kernel_ev), c = get_ms(copy_ev);
            double t = get_ms(transp_ev), d = get_ms(down_ev);
            total_upload += u; total_kernel += k; total_copy += c;
            total_transp += t; total_down += d;
            iter_times.push_back(u + k + c + t + d);

            clReleaseEvent(upload_ev); clReleaseEvent(kernel_ev); clReleaseEvent(copy_ev);
            clReleaseEvent(transp_ev); clReleaseEvent(down_ev);
        }

        double avg_up = total_upload / ITERS, avg_kern = total_kernel / ITERS;
        double avg_cp = total_copy / ITERS, avg_tr = total_transp / ITERS, avg_dn = total_down / ITERS;
        double avg_total = avg_up + avg_kern + avg_cp + avg_tr + avg_dn;

        // Compute stdev of total time
        double sum_sq = 0;
        for (double t : iter_times) {
            double d = t - avg_total;
            sum_sq += d * d;
        }
        double stdev = std::sqrt(sum_sq / ITERS);

        fprintf(out, "%-12s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                RES_SET[ri].name, avg_up, avg_kern, avg_cp, avg_tr, avg_dn, avg_total, stdev);

        // Correctness check: quick/stride=first res, full=all resolutions
        bool check_correctness = (mode == 0) ? (ri == res_idx[0]) : true;
        if (check_correctness) {
            std::vector<float> cpu_dst;
            cpuBaseline(src_data, sw, sh, sstride, cpu_dst, DST_W, DST_H);

            float max_diff = 0;
            std::vector<float> gpu_norm(norm_bytes / sizeof(float));
            clEnqueueReadBuffer(queue, norm_buf, CL_TRUE, 0, norm_bytes, gpu_norm.data(), 0, NULL, NULL);
            for (size_t i = 0; i < cpu_dst.size(); i++) {
                float d = fabsf(gpu_norm[i] - cpu_dst[i]);
                if (d > max_diff) max_diff = d;
            }
            bool pass = (max_diff < 1e-3f);
            fprintf(out, "# %s Correctness: max_diff=%.6f %s\n",
                    RES_SET[ri].name, max_diff, pass ? "PASS" : "FAIL");
            if (!pass) all_pass = false;
        }

        clReleaseMemObject(src_buf); clReleaseMemObject(norm_buf);
        clReleaseMemObject(copy_norm_buf); clReleaseMemObject(patch_buf);
    }

    float temp_after = readGpuTemp();
    float temp_delta = (temp_before > 0 && temp_after > 0) ? (temp_after - temp_before) : 0;
    fprintf(out, "# GPU Temp After: %.1fC (delta: +%.1fC)\n", temp_after, temp_delta);
    fprintf(out, "# Overall: %s\n", all_pass ? "PASS" : "FAIL");

    if (output_file) fclose(out);
    clReleaseKernel(k_resize); clReleaseKernel(k_transpose);
    clReleaseProgram(prog); clReleaseCommandQueue(queue); clReleaseContext(ctx);
    return all_pass ? 0 : 1;
}
