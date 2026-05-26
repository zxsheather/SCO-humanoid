#!/usr/bin/env python3
"""Summarize MuJoCo actuator-proxy stress artifacts for issue #54."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from _common import ensure_directory, relative_to_repo, write_json  # noqa: E402
from run_mujoco_actuator_proxy_stress import METRICS_NAME, NOMINAL_METRICS_NAME, RUNS  # noqa: E402
from run_mujoco_actuator_proxy_stress import metrics_path, nominal_metrics_path  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis" / "mujoco_actuator_proxy_stress"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def nested_mean(rows: list[dict[str, Any]], outer: str, key: str) -> float:
    return statistics.fmean(float(row[outer][key]) for row in rows)


def factor(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def summarize_method(
    method_id: str,
    proxy_rows: list[dict[str, Any]],
    nominal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    nominal_joint = mean(nominal_rows, "joint_acceleration_l2_mean")
    proxy_joint = mean(proxy_rows, "joint_acceleration_l2_mean")
    nominal_jitter = mean(nominal_rows, "action_jitter_l2_mean")
    proxy_jitter = mean(proxy_rows, "action_jitter_l2_mean")
    proxy_applied_jitter = mean(proxy_rows, "applied_action_jitter_l2_mean")

    return {
        "method_id": method_id,
        "source_count": len(proxy_rows),
        "nominal": {
            "fall_rate": mean(nominal_rows, "fall_rate"),
            "velocity_tracking_error": mean(nominal_rows, "velocity_tracking_error_mean"),
            "joint_acceleration_l2": nominal_joint,
            "action_jitter_l2": nominal_jitter,
            "episode_steps": nested_mean(nominal_rows, "mujoco_eval", "episode_steps_mean"),
        },
        "proxy": {
            "fall_rate": mean(proxy_rows, "fall_rate"),
            "velocity_tracking_error": mean(proxy_rows, "velocity_tracking_error_mean"),
            "joint_acceleration_l2": proxy_joint,
            "action_jitter_l2": proxy_jitter,
            "applied_action_jitter_l2": proxy_applied_jitter,
            "action_lag_l2": mean(proxy_rows, "action_lag_l2_mean"),
            "episode_steps": nested_mean(proxy_rows, "mujoco_eval", "episode_steps_mean"),
        },
        "degradation": {
            "fall_rate_delta": mean(proxy_rows, "fall_rate") - mean(nominal_rows, "fall_rate"),
            "velocity_tracking_error_delta": (
                mean(proxy_rows, "velocity_tracking_error_mean") - mean(nominal_rows, "velocity_tracking_error_mean")
            ),
            "episode_steps_delta": (
                nested_mean(proxy_rows, "mujoco_eval", "episode_steps_mean")
                - nested_mean(nominal_rows, "mujoco_eval", "episode_steps_mean")
            ),
            "joint_acceleration_factor": factor(proxy_joint, nominal_joint),
            "raw_action_jitter_factor": factor(proxy_jitter, nominal_jitter),
            "applied_vs_raw_proxy_jitter_factor": factor(proxy_applied_jitter, proxy_jitter),
        },
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    rows = summary["method_summaries"]
    lines = [
        "# MuJoCo Actuator-Proxy Stress Summary",
        "",
        "Protocol: `isaac_mainline`, `20 episodes x 20s`, `joint_reset_noise = 0.1`, "
        "command `(vx=0.4, vy=0.0, dyaw=0.0)`, with a first-order action low-pass proxy "
        "before PD target generation.",
        "",
        "Actuator proxy:",
        "",
        f"- mode: `{summary['protocol']['actuator_proxy_mode']}`",
        f"- low-pass time constant: `{summary['protocol']['actuator_lowpass_time_constant']}` seconds",
        f"- nominal control timestep: `{summary['protocol']['control_dt']}` seconds",
        f"- low-pass alpha: `{summary['protocol']['lowpass_alpha']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Nominal fall | Proxy fall | Nominal jnt acc | Proxy jnt acc | Jnt acc factor | Nominal raw jitter | Proxy raw jitter | Proxy applied jitter | Proxy lag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method_id} | {nf} | {pf} | {nja} | {pja} | {jfac} | {njit} | {pjit} | {pajit} | {lag} |".format(
                method_id=row["method_id"],
                nf=fmt(row["nominal"]["fall_rate"]),
                pf=fmt(row["proxy"]["fall_rate"]),
                nja=fmt(row["nominal"]["joint_acceleration_l2"]),
                pja=fmt(row["proxy"]["joint_acceleration_l2"]),
                jfac=fmt(row["degradation"]["joint_acceleration_factor"]),
                njit=fmt(row["nominal"]["action_jitter_l2"]),
                pjit=fmt(row["proxy"]["action_jitter_l2"]),
                pajit=fmt(row["proxy"]["applied_action_jitter_l2"]),
                lag=fmt(row["proxy"]["action_lag_l2"]),
            )
        )

    lines.extend(
        [
            "",
            "## Task Degradation",
            "",
            "| Method | Nominal vel err | Proxy vel err | Vel err delta | Nominal steps | Proxy steps | Steps delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {method_id} | {nvel} | {pvel} | {dvel} | {nsteps} | {psteps} | {dsteps} |".format(
                method_id=row["method_id"],
                nvel=fmt(row["nominal"]["velocity_tracking_error"]),
                pvel=fmt(row["proxy"]["velocity_tracking_error"]),
                dvel=fmt(row["degradation"]["velocity_tracking_error_delta"]),
                nsteps=fmt(row["nominal"]["episode_steps"]),
                psteps=fmt(row["proxy"]["episode_steps"]),
                dsteps=fmt(row["degradation"]["episode_steps_delta"]),
            )
        )

    lines.extend(["", "## Source Artifacts", ""])
    for artifact in summary["source_artifacts"]:
        lines.append(f"- `{artifact['method_id']}` seed `{artifact['seed']}` proxy: `{artifact['proxy_metrics_path']}`")
        lines.append(f"- `{artifact['method_id']}` seed `{artifact['seed']}` nominal: `{artifact['nominal_metrics_path']}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MuJoCo actuator-proxy stress metrics.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    method_order: list[str] = []
    proxy_by_method: dict[str, list[dict[str, Any]]] = {}
    nominal_by_method: dict[str, list[dict[str, Any]]] = {}
    source_artifacts: list[dict[str, Any]] = []

    for run in RUNS:
        proxy_path = metrics_path(run)
        nominal_path = nominal_metrics_path(run)
        proxy_metrics = read_json(proxy_path)
        nominal_metrics = read_json(nominal_path)
        if run.method_id not in proxy_by_method:
            method_order.append(run.method_id)
        proxy_by_method.setdefault(run.method_id, []).append(proxy_metrics)
        nominal_by_method.setdefault(run.method_id, []).append(nominal_metrics)
        source_artifacts.append(
            {
                "method_id": run.method_id,
                "seed": run.seed,
                "checkpoint": run.checkpoint,
                "proxy_metrics_path": relative_to_repo(proxy_path),
                "nominal_metrics_path": relative_to_repo(nominal_path),
            }
        )

    method_summaries = [
        summarize_method(method_id, proxy_by_method[method_id], nominal_by_method[method_id])
        for method_id in method_order
    ]
    first_proxy = proxy_by_method[method_order[0]][0]
    actuator_proxy = first_proxy["actuator_proxy"]
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
            "actuator_proxy_mode": actuator_proxy["mode"],
            "actuator_lowpass_time_constant": actuator_proxy["lowpass_time_constant"],
            "control_dt": actuator_proxy["control_dt"],
            "lowpass_alpha": actuator_proxy["lowpass_alpha"],
        },
        "method_summaries": method_summaries,
        "source_artifacts": source_artifacts,
        "metric_files": {
            "proxy": METRICS_NAME,
            "nominal": NOMINAL_METRICS_NAME,
        },
    }

    output_dir = ensure_directory(Path(args.output_dir))
    write_json(output_dir / "summary.json", summary)
    write_markdown(summary, output_dir / "summary.md")
    print(f"Wrote {relative_to_repo(output_dir / 'summary.json')}")
    print(f"Wrote {relative_to_repo(output_dir / 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
