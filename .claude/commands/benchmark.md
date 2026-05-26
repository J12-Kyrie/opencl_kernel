# /benchmark — OpenCL Kernel Benchmark Command

## 3-Level Strategy

| Level   | Flag | Resolutions | Time | Purpose |
|---------|------|-------------|------|---------|
| quick   | --quick | 2 (320x240, 1920x1080) | ~30s | Compile check + correctness |
| stride N| --stride N | N-dependent | ~2-3 min | Default iteration |
| full    | --full | 8 (all) | ~8 min | Confirm new best |

## Resolution Test Set
{320x240, 640x480, 800x600, 1024x768, 1280x720, 1600x900, 1920x1080, 3840x2160}

## Execution Flow

### 1. Build kernel on IQ9 device
```bash
sshpass -p "Hello123" ssh -o StrictHostKeyChecking=no ubuntu@192.168.100.102 \
  "cd /mnt/workspace/opencl_kernel/build && cmake ../solution/host -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
```
If build fails: capture error, report, abort benchmark.

### 2. Run benchmark
```bash
sshpass -p "Hello123" ssh -o StrictHostKeyChecking=no ubuntu@192.168.100.102 \
  "/mnt/workspace/opencl_kernel/build/benchmark <flag> --output /tmp/bench.log"
```

### 3. Fetch results
```bash
sshpass -p "Hello123" scp -o StrictHostKeyChecking=no \
  ubuntu@192.168.100.102:/tmp/bench.log experiments/exp_N/bench.log
```

## Output Format (bench.log)
```
# OpenCL Kernel Benchmark
# Timestamp: <unix_ts>
# Mode: quick | stride N | full
Resolution   Upload_ms  Kernel_ms  Copy_ms  Transp_ms  Down_ms  Total_ms
320x240          x.xxx     x.xxx    x.xxx     x.xxx    x.xxx    x.xxx
640x480          x.xxx     x.xxx    x.xxx     x.xxx    x.xxx    x.xxx
...
# Correctness: max_diff=X.XXXXXX PASS|FAIL
# Overall: PASS|FAIL
```

## Key Metrics
- Kernel time = resize_bilinear_normalized execution time
- Transfer time = upload + download
- Total time = upload + kernel + copy + transpose + download
- Per-resolution breakdown matters — don't just report mean

## A/B Paired Test
When sub-5% difference needs confirmation:
1. Run benchmark with current kernel.cl -> save to /tmp/bench_A.log
2. Swap in previous best kernel.cl
3. Run benchmark with previous kernel.cl -> save to /tmp/bench_B.log
4. Compare per-resolution total time deltas
5. If A is worse on ALL resolutions -> regression confirmed
6. Swap back the implementation kernel
