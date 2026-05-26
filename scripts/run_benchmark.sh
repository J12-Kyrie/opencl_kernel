#!/bin/bash
# ============================================================
# run_benchmark.sh — Build + Run OpenCL Kernel Benchmark on IQ9
# Usage: ./run_benchmark.sh [--quick|--stride N|--full]
# ============================================================
set -e

MODE="${1:---stride}"
STRIDE="${2:-2}"
SSH_CMD="sshpass -p 'Hello123' ssh -o StrictHostKeyChecking=no ubuntu@192.168.100.102"
BUILD_DIR="/mnt/workspace/opencl_kernel/build"
BENCHMARK_BIN="$BUILD_DIR/benchmark"

echo "=== Building OpenCL kernel benchmark ==="
$SSH_CMD "mkdir -p $BUILD_DIR && cd $BUILD_DIR && cmake ../solution/host -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"

echo "=== Running benchmark ($MODE $STRIDE) ==="
if [ "$MODE" = "--stride" ]; then
    $SSH_CMD "$BENCHMARK_BIN --stride $STRIDE --output /tmp/bench.log"
else
    $SSH_CMD "$BENCHMARK_BIN $MODE --output /tmp/bench.log"
fi

echo "=== Fetching results ==="
sshpass -p "Hello123" scp -o StrictHostKeyChecking=no ubuntu@192.168.100.102:/tmp/bench.log ./experiments/

echo "=== Benchmark complete ==="
cat ./experiments/bench.log
