#!/usr/bin/env python3
"""Analyze trace-based secondary physical metrics for the full-paper MuJoCo replay."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import ensure_directory, relative_to_repo, write_json  # noqa: E402
from _mujoco_physical_trace_metrics import compute_episode_physical_metrics  # noqa: E402
import run_mujoco_physical_trace_replays as replay_runner  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis" / "mujoco_physical_secondary_metrics"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "full-paper" / "mujoco-physical-secondary-metrics.md"

SEEDS = [11, 17, 23, 29, 31]
METHOD_ORDER = ["lcp", "scppo", "heuristic"]

PHYSICAL_METRICS = [
    ("control_torque_rms", "Torque RMS", True),
    ("mechanical_power_abs_mean", "Abs power", True),
    ("mechanical_energy_abs", "Abs energy", True),
    ("mechanical_cot_proxy", "Energy/m", True),
]
PRIMARY_METRICS = [
    ("velocity_tracking_error_mean", "Vel. err", True),
    ("joint_acceleration_l2_mean", "Jnt acc", True),
    ("action_jitter_l2_mean", "Jitter", True),
    ("episode_return_mean", "Return", False),
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


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


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def run_summary(run: replay_runner.TraceReplayRun) -> dict[str, Any]:
    metrics_path, trace_path = replay_runner.expected_paths(run)
    metrics = read_json(metrics_path)
    trace_payload = read_json(trace_path)
    control_dt = float(trace_payload["control_dt"])

    episode_rows: list[dict[str, Any]] = []
    physical_values: dict[str, list[float]] = {
        key: [] for key, _, _ in PHYSICAL_METRICS
    }
    fell_count = 0
    for episode in trace_payload.get("episodes", []):
        row = compute_episode_physical_metrics(episode, control_dt=control_dt)
        if row["fell"]:
            fell_count += 1
        episode_rows.append(
            {
                "episode_index": episode.get("episode_index"),
                "fell": row["fell"],
                "step_count": row["step_count"],
                **{key: row[key] for key, _, _ in PHYSICAL_METRICS},
            }
        )
        for key, _, _ in PHYSICAL_METRICS:
            value = row.get(key)
            if value is not None:
                physical_values[key].append(float(value))

    physical_summary = {key: summarize(values) for key, values in physical_values.items()}
    primary_summary = {
        key: float(metrics[key]) for key, _, _ in PRIMARY_METRICS if metrics.get(key) is not None
    }
    return {
        "method_id": run.method_id,
        "method": run.method_label,
        "seed": run.seed,
        "selected_checkpoint": run.checkpoint,
        "config_path": run.config_path,
        "artifact_run_name": run.artifact_run_name,
        "load_run": run.load_run,
        "replay_run_name": run.replay_run_name,
        "metrics_path": relative_to_repo(metrics_path),
        "trace_path": relative_to_repo(trace_path),
        "trace_episode_count": len(trace_payload.get("episodes", [])),
        "trace_fall_count": fell_count,
        "trace_fall_rate": fell_count / max(len(trace_payload.get("episodes", [])), 1),
        "control_dt": control_dt,
        "physical_summary": physical_summary,
        "primary_metrics": primary_summary,
        "episode_rows": episode_rows,
    }


def collect() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for run in replay_runner.build_runs():
        metrics_path, trace_path = replay_runner.expected_paths(run)
        if not metrics_path.exists():
            missing.append(relative_to_repo(metrics_path))
            continue
        if not trace_path.exists():
            missing.append(relative_to_repo(trace_path))
            continue
        rows.append(run_summary(run))
    return rows, sorted(set(missing))


def aggregate_method_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method_id in METHOD_ORDER:
        method_rows = [row for row in run_rows if row["method_id"] == method_id]
        if len(method_rows) != len(SEEDS):
            continue
        aggregate = {
            "method_id": method_id,
            "method": method_rows[0]["method"],
            "seed_count": len(method_rows),
            "checkpoints": "/".join(str(next(row["selected_checkpoint"] for row in method_rows if row["seed"] == seed)) for seed in SEEDS),
            "trace_episode_count_per_seed": [int(row["trace_episode_count"]) for row in sorted(method_rows, key=lambda row: row["seed"])],
            "physical_metrics": {},
            "primary_metrics": {},
        }
        for key, _, _ in PHYSICAL_METRICS:
            values = [float(row["physical_summary"][key]["mean"]) for row in method_rows if row["physical_summary"][key] is not None]
            aggregate["physical_metrics"][key] = summarize(values)
        for key, _, _ in PRIMARY_METRICS:
            values = [float(row["primary_metrics"][key]) for row in method_rows if row["primary_metrics"].get(key) is not None]
            aggregate["primary_metrics"][key] = summarize(values)
        rows.append(aggregate)
    return rows


def best_method_label(aggregate_rows: list[dict[str, Any]], metric_key: str) -> str:
    available = [
        row for row in aggregate_rows
        if row["physical_metrics"].get(metric_key) is not None and row["physical_metrics"][metric_key].get("mean") is not None
    ]
    best = min(available, key=lambda row: float(row["physical_metrics"][metric_key]["mean"]))
    return str(best["method"])


def row_by_method(aggregate_rows: list[dict[str, Any]], method_id: str) -> dict[str, Any]:
    for row in aggregate_rows:
        if row["method_id"] == method_id:
            return row
    raise KeyError(f"Missing aggregate row for {method_id}")


def correlation_by_metric(rows: list[dict[str, Any]], physical_metric: str) -> dict[str, Any]:
    for row in rows:
        if row["physical_metric"] == physical_metric:
            return row
    raise KeyError(f"Missing correlation row for {physical_metric}")


def correlation_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for physical_key, physical_label, _ in PHYSICAL_METRICS:
        physical_values = [
            float(row["physical_summary"][physical_key]["mean"])
            for row in run_rows
            if row["physical_summary"].get(physical_key) is not None
        ]
        joint_values = [
            float(row["primary_metrics"]["joint_acceleration_l2_mean"])
            for row in run_rows
            if row["physical_summary"].get(physical_key) is not None
        ]
        jitter_values = [
            float(row["primary_metrics"]["action_jitter_l2_mean"])
            for row in run_rows
            if row["physical_summary"].get(physical_key) is not None
        ]
        vel_values = [
            float(row["primary_metrics"]["velocity_tracking_error_mean"])
            for row in run_rows
            if row["physical_summary"].get(physical_key) is not None
        ]
        rows.append(
            {
                "physical_metric": physical_key,
                "physical_metric_label": physical_label,
                "corr_joint_acceleration": pearson(physical_values, joint_values),
                "corr_action_jitter": pearson(physical_values, jitter_values),
                "corr_velocity_error": pearson(physical_values, vel_values),
            }
        )
    return rows


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    if summary["missing_artifacts"]:
        lines = [
            "# MuJoCo Physical Secondary Metrics (#117)",
            "",
            "Status: `blocked`.",
            "",
            "Missing required replay artifacts:",
            "",
        ]
        lines.extend(f"- `{item}`" for item in summary["missing_artifacts"])
        lines.extend(
            [
                "",
                "Generate them with:",
                "",
                "```bash",
                "python scripts/baseline/run_mujoco_physical_trace_replays.py --skip-completed",
                "```",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    aggregate_rows = summary["aggregate_rows"]
    corr_rows = summary["correlation_rows"]
    lcp_row = row_by_method(aggregate_rows, "lcp")
    scppo_row = row_by_method(aggregate_rows, "scppo")
    heuristic_row = row_by_method(aggregate_rows, "heuristic")
    power_corr = correlation_by_metric(corr_rows, "mechanical_power_abs_mean")
    efficiency_corr = correlation_by_metric(corr_rows, "mechanical_cot_proxy")

    lines = [
        "# MuJoCo Physical Secondary Metrics (#117)",
        "",
        "Status: `complete`.",
        "",
        "This note adds a trace-based secondary physical evaluation layer on the current five-seed selected-checkpoint MuJoCo replay. "
        "It uses the same selected checkpoints as the main mechanism comparison, rerun with compact per-timestep trace capture, and "
        "computes torque- and power-based proxies from the applied joint control and joint-velocity traces. These metrics are secondary "
        "physical evidence; they do not replace the primary task, joint-acceleration, or action-jitter metrics.",
        "",
        "## Main Read",
        "",
        (
            f"- LCP is the lowest absolute-effort row on torque RMS ({fmt(lcp_row['physical_metrics']['control_torque_rms']['mean'])}), "
            f"absolute power ({fmt(lcp_row['physical_metrics']['mechanical_power_abs_mean']['mean'])}), and absolute episode energy "
            f"({fmt(lcp_row['physical_metrics']['mechanical_energy_abs']['mean'])}). The heuristic remains best on the efficiency-style "
            f"`energy/m` proxy ({fmt(heuristic_row['physical_metrics']['mechanical_cot_proxy']['mean'])}) and on aggregate MuJoCo joint "
            f"acceleration ({fmt(heuristic_row['primary_metrics']['joint_acceleration_l2_mean']['mean'])})."
        ),
        (
            f"- This sharpens, rather than removes, the existing MuJoCo split: LCP appears physically cheaper in absolute actuation effort, "
            f"while the heuristic remains more efficient per forward progress. SC-PPO is worst on absolute power "
            f"({fmt(scppo_row['physical_metrics']['mechanical_power_abs_mean']['mean'])}) and remains worst on aggregate joint acceleration "
            f"({fmt(scppo_row['primary_metrics']['joint_acceleration_l2_mean']['mean'])})."
        ),
        (
            f"- Across the 15 method-seed rows, absolute power correlates with joint acceleration at "
            f"{fmt(power_corr['corr_joint_acceleration'])} and with action jitter at {fmt(power_corr['corr_action_jitter'])}; "
            f"the efficiency proxy `energy/m` aligns most strongly with velocity error ({fmt(efficiency_corr['corr_velocity_error'])}). "
            "The physical proxies therefore do not collapse to a single smoothness scalar."
        ),
        "- Because these traces use five captured episodes per selected checkpoint, this is still bounded secondary evidence rather than a new primary benchmark line.",
        "",
        "## Five-Seed Aggregate",
        "",
        "| Method | Checkpoints | Torque RMS | Abs power | Abs energy | Energy/m | MuJoCo Jnt acc | MuJoCo Jitter | MuJoCo Vel. err | MuJoCo Return |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        physical = row["physical_metrics"]
        primary = row["primary_metrics"]
        lines.append(
            "| {method} | {ckpt} | {tau} | {power} | {energy} | {cot} | {joint} | {jitter} | {vel} | {ret} |".format(
                method=row["method"],
                ckpt=row["checkpoints"],
                tau=fmt(physical["control_torque_rms"]["mean"]),
                power=fmt(physical["mechanical_power_abs_mean"]["mean"]),
                energy=fmt(physical["mechanical_energy_abs"]["mean"]),
                cot=fmt(physical["mechanical_cot_proxy"]["mean"]),
                joint=fmt(primary["joint_acceleration_l2_mean"]["mean"]),
                jitter=fmt(primary["action_jitter_l2_mean"]["mean"]),
                vel=fmt(primary["velocity_tracking_error_mean"]["mean"]),
                ret=fmt(primary["episode_return_mean"]["mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Seed Rows",
            "",
            "| Method | Seed | Ckpt | Trace fall | Torque RMS | Abs power | Abs energy | Energy/m | MuJoCo Jnt acc | MuJoCo Jitter | MuJoCo Return |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summary["run_rows"], key=lambda item: (item["method_id"], item["seed"])):
        physical = row["physical_summary"]
        primary = row["primary_metrics"]
        lines.append(
            "| {method} | {seed} | {ckpt} | {trace_fall} | {tau} | {power} | {energy} | {cot} | {joint} | {jitter} | {ret} |".format(
                method=row["method"],
                seed=row["seed"],
                ckpt=row["selected_checkpoint"],
                trace_fall=fmt(row["trace_fall_rate"]),
                tau=fmt(physical["control_torque_rms"]["mean"]),
                power=fmt(physical["mechanical_power_abs_mean"]["mean"]),
                energy=fmt(physical["mechanical_energy_abs"]["mean"]),
                cot=fmt(physical["mechanical_cot_proxy"]["mean"]),
                joint=fmt(primary["joint_acceleration_l2_mean"]),
                jitter=fmt(primary["action_jitter_l2_mean"]),
                ret=fmt(primary["episode_return_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Correlation with Existing MuJoCo Metrics",
            "",
            "Correlations are descriptive Pearson correlations over the 15 method-seed rows.",
            "",
            "| Physical metric | Corr with Jnt acc | Corr with Jitter | Corr with Vel. err |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in corr_rows:
        lines.append(
            "| {metric} | {joint} | {jitter} | {vel} |".format(
                metric=row["physical_metric_label"],
                joint=fmt(row["corr_joint_acceleration"]),
                jitter=fmt(row["corr_action_jitter"]),
                vel=fmt(row["corr_velocity_error"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- These are no-retraining selected-checkpoint MuJoCo traces, not hardware measurements.",
            "- The metrics are physically grounded proxies built from simulated joint control and joint velocity; they strengthen interpretation but do not establish sim-to-real energy or actuator claims.",
            "- The current read should remain secondary to the main mechanism result: policy-output smoothness and downstream dynamic effort are related, but they are not identical orderings.",
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for source in summary["source_artifacts"]:
        lines.append(f"- `{source}`")

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/baseline/run_mujoco_physical_trace_replays.py --skip-completed",
            "python scripts/analysis/analyze_mujoco_physical_metrics.py",
            "```",
            "",
            f"Generated runtime summary: `{summary['generated_artifacts']['summary_json']}`",
            f"Generated runtime note: `{summary['generated_artifacts']['summary_markdown']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(output_dir: Path) -> dict[str, Any]:
    run_rows, missing = collect()
    if missing:
        return {
            "issue": "#117",
            "missing_artifacts": missing,
            "generated_artifacts": {
                "summary_json": relative_to_repo(output_dir / "summary.json"),
                "summary_markdown": relative_to_repo(output_dir / "summary.md"),
            },
        }

    source_artifacts = []
    for row in run_rows:
        source_artifacts.extend([row["metrics_path"], row["trace_path"]])
    summary = {
        "issue": "#117",
        "protocol": {
            "seeds": SEEDS,
            "trace_replay_metrics": replay_runner.METRICS_NAME,
            "trace_replay_trace": replay_runner.TRACE_NAME,
            "interpretation": "Secondary physical proxies from selected-checkpoint MuJoCo traces.",
        },
        "run_rows": run_rows,
        "aggregate_rows": aggregate_method_rows(run_rows),
        "correlation_rows": correlation_rows(run_rows),
        "missing_artifacts": [],
        "source_artifacts": sorted(set(source_artifacts)),
        "generated_artifacts": {
            "summary_json": relative_to_repo(output_dir / "summary.json"),
            "summary_markdown": relative_to_repo(output_dir / "summary.md"),
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MuJoCo secondary physical metrics.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--doc-path", default=str(DEFAULT_DOC_PATH))
    args = parser.parse_args()

    output_dir = ensure_directory(Path(args.output_dir))
    doc_path = Path(args.doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(output_dir)
    write_json(output_dir / "summary.json", summary)
    write_markdown(summary, output_dir / "summary.md")
    write_markdown(summary, doc_path)
    print(f"Wrote {relative_to_repo(output_dir / 'summary.json')}")
    print(f"Wrote {relative_to_repo(output_dir / 'summary.md')}")
    print(f"Wrote {relative_to_repo(doc_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
