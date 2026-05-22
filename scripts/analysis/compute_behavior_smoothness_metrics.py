#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from _behavior_trace_metrics import compute_episode_smoothness_metrics  # noqa: E402


def summarize(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute trace-based behavior smoothness metrics from saved episode traces.")
    parser.add_argument("--trace-path", required=True, help="Path to episode_traces.json.")
    parser.add_argument("--output", default=None, help="Optional output path. Defaults next to the trace file.")
    parser.add_argument("--sparc-cutoff-hz", type=float, default=None, help="Optional SPARC frequency cutoff.")
    parser.add_argument(
        "--sparc-amplitude-threshold",
        type=float,
        default=0.05,
        help="Relative amplitude threshold for SPARC support selection.",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace_path).expanduser().resolve()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", [])

    episode_rows: list[dict[str, Any]] = []
    ldlj_values: list[float] = []
    sparc_values: list[float] = []

    for episode in episodes:
        metrics = compute_episode_smoothness_metrics(
            episode,
            sparc_cutoff_hz=args.sparc_cutoff_hz,
            sparc_amplitude_threshold=args.sparc_amplitude_threshold,
        )
        row = {
            "episode_index": episode.get("episode_index"),
            "env_id": episode.get("env_id"),
            "episode_length": episode.get("episode_length"),
            "fell": episode.get("fell"),
            **metrics,
        }
        if metrics["joint_position_ldlj_mean"] is not None:
            ldlj_values.append(float(metrics["joint_position_ldlj_mean"]))
        if metrics["joint_velocity_sparc_mean"] is not None:
            sparc_values.append(float(metrics["joint_velocity_sparc_mean"]))
        episode_rows.append(row)

    output_path = Path(args.output).expanduser().resolve() if args.output else trace_path.with_name(
        "behavior_smoothness_metrics.json"
    )
    result = {
        "trace_path": relative_to_repo(trace_path),
        "episode_count": len(episodes),
        "episodes": episode_rows,
        "summary": {
            "joint_position_ldlj_mean": summarize(ldlj_values),
            "joint_velocity_sparc_mean": summarize(sparc_values),
        },
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {relative_to_repo(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
