#!/bin/bash
# ============================================================
# fetch_results.sh — Pull Benchmark Results from IQ9 to Local
# Usage: ./fetch_results.sh <exp_number>
# ============================================================
set -e
EXP_NUM="${1:-latest}"

SSH_HOST="ubuntu@192.168.100.102"
SSH_PASS="Hello123"
REMOTE_DIR="/mnt/workspace/opencl_kernel"

echo "=== Fetching experiment $EXP_NUM results ==="

# Pull bench.log
sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no \
    "$SSH_HOST:$REMOTE_DIR/experiments/exp_${EXP_NUM}/bench.log" \
    "./experiments/exp_${EXP_NUM}/bench.log" 2>/dev/null && \
    echo "Fetched: bench.log"

# Pull result.md if exists
sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no \
    "$SSH_HOST:$REMOTE_DIR/experiments/exp_${EXP_NUM}/result.md" \
    "./experiments/exp_${EXP_NUM}/result.md" 2>/dev/null && \
    echo "Fetched: result.md"

# Pull summary.md for reference
sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no \
    "$SSH_HOST:$REMOTE_DIR/experiments/summary.md" \
    "./experiments/summary.md" 2>/dev/null && \
    echo "Fetched: summary.md"

echo "=== Fetch complete ==="
