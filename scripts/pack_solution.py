#!/usr/bin/env python3
"""
pack_solution.py — Package kernel solution for archival/submission.
Creates a tarball with: kernel.cl, benchmark results, experiment summary.
"""
import os, sys, tarfile, io
from datetime import datetime

def pack_solution(workspace: str, output_name: str = None):
    if output_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"opencl_kernel_{ts}.tar.gz"

    files_to_pack = [
        "CLAUDE.md", "config.toml",
        "solution/kernel/kernel.cl", "solution/kernel/baseline.cl",
        "solution/host/benchmark.cpp", "solution/host/CMakeLists.txt",
        "experiments/summary.md", "experiments/LESSONS.md",
        "experiments/profile.md", "experiments/workload_profile.md",
    ]

    with tarfile.open(output_name, "w:gz") as tar:
        for f in files_to_pack:
            path = os.path.join(workspace, f)
            if os.path.exists(path):
                tar.add(path, arcname=f)
                print(f"  Added: {f}")
            else:
                print(f"  Skipped (not found): {f}")

    size_mb = os.path.getsize(output_name) / (1024 * 1024)
    print(f"\nPacked: {output_name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/opencl_kernel"
    output = sys.argv[2] if len(sys.argv) > 2 else None
    pack_solution(workspace, output)
