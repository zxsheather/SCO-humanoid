#!/usr/bin/env python3
"""Summarize matched MuJoCo amplification trace artifacts for issue #49."""

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

from _common import artifact_dir, ensure_directory, load_config, relative_to_repo, write_json  # noqa: E402
from run_mujoco_amplification_traces import METRICS_NAME, RUNS, TRACE_NAME  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis" / "mujoco_amplification_trace_comparison"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    return sorted_values[lower] * (upper - pos) + sorted_values[upper] * (pos - lower)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def l2(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def max_contact_force(step: dict[str, Any]) -> float:
    contacts = step.get("contacts")
    if not isinstance(contacts, list) or not contacts:
        return 0.0
    return max(float(contact.get("force_norm") or 0.0) for contact in contacts)


def max_normal_force(step: dict[str, Any]) -> float:
    contacts = step.get("contacts")
    if not isinstance(contacts, list) or not contacts:
        return 0.0
    return max(float(contact.get("normal_force") or 0.0) for contact in contacts)


def run_paths(run) -> tuple[Path, Path]:
    config = load_config(run.config)
    output_dir = artifact_dir(config, run.run_name)
    return output_dir / METRICS_NAME, output_dir / TRACE_NAME


def iter_steps(run, trace_payload: dict[str, Any]):
    for episode in trace_payload.get("episodes", []):
        for step in episode.get("trace", []):
            yield {
                "method_id": run.method_id,
                "seed": run.seed,
                "episode_index": episode.get("episode_index"),
                "fell": bool(episode.get("fell")),
                "step": int(step.get("step")),
                "time": float(step.get("time")),
                "action_jitter_l2": float(step.get("action_jitter_l2") or 0.0),
                "joint_acceleration_l2": float(step.get("joint_acceleration_l2") or 0.0),
                "control_tau_l2": l2(step.get("control_tau") or []),
                "max_contact_force": max_contact_force(step),
                "max_normal_force": max_normal_force(step),
                "contact_count": int(step.get("contact_count") or 0),
            }


def summarize_method(method_id: str, rows: list[dict[str, Any]], metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    action = [row["action_jitter_l2"] for row in rows]
    joint = [row["joint_acceleration_l2"] for row in rows]
    tau = [row["control_tau_l2"] for row in rows]
    force = [row["max_contact_force"] for row in rows]
    contacts = [row["contact_count"] for row in rows]

    joint_p95 = percentile(joint, 0.95) or 0.0
    joint_spikes = [row for row in rows if row["joint_acceleration_l2"] >= joint_p95]
    top_joint_spikes = sorted(rows, key=lambda row: row["joint_acceleration_l2"], reverse=True)[:10]

    return {
        "method_id": method_id,
        "source_count": len(metrics_rows),
        "trace_steps": len(rows),
        "trace_episode_count": len({(row["seed"], row["episode_index"]) for row in rows}),
        "fall_rate_mean_from_metrics": statistics.fmean(float(row["fall_rate"]) for row in metrics_rows),
        "episode_steps_mean_from_metrics": statistics.fmean(
            float(row["mujoco_eval"]["episode_steps_mean"]) for row in metrics_rows
        ),
        "mujoco_joint_acceleration_mean": statistics.fmean(
            float(row["joint_acceleration_l2_mean"]) for row in metrics_rows
        ),
        "mujoco_action_jitter_mean": statistics.fmean(float(row["action_jitter_l2_mean"]) for row in metrics_rows),
        "action_jitter_l2": {
            "mean": statistics.fmean(action) if action else None,
            "p95": percentile(action, 0.95),
            "max": max(action) if action else None,
        },
        "joint_acceleration_l2": {
            "mean": statistics.fmean(joint) if joint else None,
            "p95": joint_p95,
            "max": max(joint) if joint else None,
        },
        "control_tau_l2": {
            "mean": statistics.fmean(tau) if tau else None,
            "p95": percentile(tau, 0.95),
            "max": max(tau) if tau else None,
        },
        "max_contact_force": {
            "mean": statistics.fmean(force) if force else None,
            "p95": percentile(force, 0.95),
            "max": max(force) if force else None,
        },
        "contact_count": {
            "mean": statistics.fmean(contacts) if contacts else None,
            "p95": percentile([float(value) for value in contacts], 0.95),
            "max": max(contacts) if contacts else None,
        },
        "correlations": {
            "action_jitter_vs_joint_acceleration": pearson(action, joint),
            "contact_force_vs_joint_acceleration": pearson(force, joint),
            "control_tau_vs_joint_acceleration": pearson(tau, joint),
        },
        "joint_spike_p95_threshold": joint_p95,
        "joint_spike_count": len(joint_spikes),
        "joint_spike_contact_fraction": (
            sum(1 for row in joint_spikes if row["contact_count"] > 0) / len(joint_spikes) if joint_spikes else None
        ),
        "joint_spike_mean_action_jitter": (
            statistics.fmean(row["action_jitter_l2"] for row in joint_spikes) if joint_spikes else None
        ),
        "joint_spike_mean_contact_force": (
            statistics.fmean(row["max_contact_force"] for row in joint_spikes) if joint_spikes else None
        ),
        "joint_spike_mean_control_tau": (
            statistics.fmean(row["control_tau_l2"] for row in joint_spikes) if joint_spikes else None
        ),
        "top_joint_spikes": top_joint_spikes,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    rows = summary["method_summaries"]
    lines = [
        "# MuJoCo Amplification Trace Comparison",
        "",
        "Matched replay protocol: `isaac_mainline`, `20 episodes x 20s`, "
        "`joint_reset_noise = 0.1`, command `(vx=0.4, vy=0.0, dyaw=0.0)`, "
        "capturing the first 3 episodes per seed with up to 1024 control steps each.",
        "",
        "## Aggregate Trace Metrics",
        "",
        "| Method | Trace steps | Fall rate | MuJoCo jnt acc | MuJoCo jitter | Trace jitter p95 | Trace jnt acc p95 | Contact force p95 | Tau p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method_id} | {steps} | {fall} | {mja} | {mjit} | {jp95} | {ap95} | {fp95} | {tp95} |".format(
                method_id=row["method_id"],
                steps=row["trace_steps"],
                fall=fmt(row["fall_rate_mean_from_metrics"]),
                mja=fmt(row["mujoco_joint_acceleration_mean"]),
                mjit=fmt(row["mujoco_action_jitter_mean"]),
                jp95=fmt(row["action_jitter_l2"]["p95"]),
                ap95=fmt(row["joint_acceleration_l2"]["p95"]),
                fp95=fmt(row["max_contact_force"]["p95"]),
                tp95=fmt(row["control_tau_l2"]["p95"]),
            )
        )

    lines.extend(
        [
            "",
            "## Spike Coupling",
            "",
            "| Method | Joint-spike threshold | Spike contact fraction | Spike mean jitter | Spike mean contact force | Spike mean tau | corr(jitter,jnt_acc) | corr(contact,jnt_acc) | corr(tau,jnt_acc) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {method_id} | {thr} | {frac} | {jit} | {force} | {tau} | {corr1} | {corr2} | {corr3} |".format(
                method_id=row["method_id"],
                thr=fmt(row["joint_spike_p95_threshold"]),
                frac=fmt(row["joint_spike_contact_fraction"]),
                jit=fmt(row["joint_spike_mean_action_jitter"]),
                force=fmt(row["joint_spike_mean_contact_force"]),
                tau=fmt(row["joint_spike_mean_control_tau"]),
                corr1=fmt(row["correlations"]["action_jitter_vs_joint_acceleration"]),
                corr2=fmt(row["correlations"]["contact_force_vs_joint_acceleration"]),
                corr3=fmt(row["correlations"]["control_tau_vs_joint_acceleration"]),
            )
        )

    lines.extend(
        [
            "",
            "## Top Joint-Acceleration Spikes",
            "",
            "| Method | Seed | Episode | Step | Time | Fell | Joint acc | Action jitter | Contact force | Contact count | Tau |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        for spike in row["top_joint_spikes"][:3]:
            lines.append(
                "| {method_id} | {seed} | {episode} | {step} | {time} | {fell} | {joint} | {jitter} | {force} | {contacts} | {tau} |".format(
                    method_id=row["method_id"],
                    seed=spike["seed"],
                    episode=spike["episode_index"],
                    step=spike["step"],
                    time=fmt(spike["time"], 2),
                    fell=str(spike["fell"]).lower(),
                    joint=fmt(spike["joint_acceleration_l2"]),
                    jitter=fmt(spike["action_jitter_l2"]),
                    force=fmt(spike["max_contact_force"]),
                    contacts=spike["contact_count"],
                    tau=fmt(spike["control_tau_l2"]),
                )
            )

    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for artifact in summary["source_artifacts"]:
        lines.append(f"- `{artifact['method_id']}` seed `{artifact['seed']}` metrics: `{artifact['metrics_path']}`")
        lines.append(f"- `{artifact['method_id']}` seed `{artifact['seed']}` trace: `{artifact['trace_path']}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze matched MuJoCo amplification trace artifacts.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    source_artifacts: list[dict[str, Any]] = []
    method_order: list[str] = []
    metrics_by_method: dict[str, list[dict[str, Any]]] = {}
    rows_by_method: dict[str, list[dict[str, Any]]] = {}

    for run in RUNS:
        metrics_path, trace_path = run_paths(run)
        metrics = read_json(metrics_path)
        trace_payload = read_json(trace_path)
        rows = list(iter_steps(run, trace_payload))

        if run.method_id not in rows_by_method:
            method_order.append(run.method_id)
        metrics_by_method.setdefault(run.method_id, []).append(metrics)
        rows_by_method.setdefault(run.method_id, []).extend(rows)
        source_artifacts.append(
            {
                "method_id": run.method_id,
                "seed": run.seed,
                "checkpoint": run.checkpoint,
                "metrics_path": relative_to_repo(metrics_path),
                "trace_path": relative_to_repo(trace_path),
            }
        )

    method_summaries = [
        summarize_method(method_id, rows_by_method[method_id], metrics_by_method[method_id])
        for method_id in method_order
    ]
    summary = {
        "protocol": {
            "terrain_mode": "isaac_mainline",
            "episodes": 20,
            "sim_duration": 20.0,
            "joint_reset_noise": 0.1,
            "base_xy_noise": 0.0,
            "command_vx": 0.4,
            "command_vy": 0.0,
            "command_dyaw": 0.0,
            "trace_max_episodes": 3,
            "trace_max_steps": 1024,
        },
        "method_summaries": method_summaries,
        "source_artifacts": source_artifacts,
    }

    output_dir = ensure_directory(Path(args.output_dir))
    write_json(output_dir / "summary.json", summary)
    write_markdown(summary, output_dir / "summary.md")
    print(f"Wrote {relative_to_repo(output_dir / 'summary.json')}")
    print(f"Wrote {relative_to_repo(output_dir / 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
