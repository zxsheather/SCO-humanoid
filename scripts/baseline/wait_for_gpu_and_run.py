#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStatus:
    index: int
    memory_free_mb: int
    memory_total_mb: int
    utilization_gpu: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll one GPU until enough free memory is available, then run a command."
        )
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        required=True,
        help="Physical GPU index to watch, as reported by nvidia-smi.",
    )
    parser.add_argument(
        "--min-free-mb",
        type=int,
        required=True,
        help="Minimum free GPU memory in MiB required before launching the command.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=60.0,
        help="Seconds to sleep between nvidia-smi polls. Default: 60.",
    )
    parser.add_argument(
        "--max-gpu-util",
        type=int,
        default=None,
        help=(
            "Optional utilization ceiling (percentage). "
            "If set, the command starts only when GPU utilization is <= this value."
        ),
    )
    parser.add_argument(
        "--export-cuda-visible-devices",
        action="store_true",
        help="Export CUDA_VISIBLE_DEVICES=<gpu-index> for the launched command.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--', for example: -- env -u DISPLAY python ...",
    )
    args = parser.parse_args()
    if not args.command:
        parser.error("missing command; pass it after '--'")
    return args


def query_gpu_status(gpu_index: int) -> GpuStatus:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if not raw:
        raise RuntimeError(f"nvidia-smi returned no data for GPU {gpu_index}")
    fields = [field.strip() for field in raw.split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi output for GPU {gpu_index}: {raw!r}")
    utilization: int | None
    if fields[3] in {"[Not Supported]", "N/A"}:
        utilization = None
    else:
        utilization = int(fields[3])
    return GpuStatus(
        index=int(fields[0]),
        memory_free_mb=int(fields[1]),
        memory_total_mb=int(fields[2]),
        utilization_gpu=utilization,
    )


def is_gpu_ready(status: GpuStatus, min_free_mb: int, max_gpu_util: int | None) -> bool:
    if status.memory_free_mb < min_free_mb:
        return False
    if max_gpu_util is None or status.utilization_gpu is None:
        return True
    return status.utilization_gpu <= max_gpu_util


def render_status(status: GpuStatus) -> str:
    util_text = "n/a" if status.utilization_gpu is None else f"{status.utilization_gpu}%"
    return (
        f"gpu={status.index} free={status.memory_free_mb}MiB "
        f"total={status.memory_total_mb}MiB util={util_text}"
    )


def main() -> int:
    args = parse_args()
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found in PATH", file=sys.stderr)
        return 2

    print(
        "Waiting for GPU capacity:",
        f"gpu={args.gpu_index}",
        f"min_free_mb={args.min_free_mb}",
        f"poll_seconds={args.poll_seconds}",
        f"max_gpu_util={args.max_gpu_util if args.max_gpu_util is not None else 'disabled'}",
        flush=True,
    )

    try:
        while True:
            status = query_gpu_status(args.gpu_index)
            ready = is_gpu_ready(status, args.min_free_mb, args.max_gpu_util)
            state = "ready" if ready else "waiting"
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {state}: {render_status(status)}", flush=True)
            if ready:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Interrupted while waiting for GPU capacity.", file=sys.stderr)
        return 130

    child_env = os.environ.copy()
    if args.export_cuda_visible_devices:
        child_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    print("Launching command:", " ".join(args.command), flush=True)
    completed = subprocess.run(args.command, env=child_env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
