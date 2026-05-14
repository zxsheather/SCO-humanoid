#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    artifact_dir,
    ensure_directory,
    load_config,
    read_json,
    relative_to_repo,
    repo_root,
    write_json,
)

REPO_ROOT = repo_root()
DEFAULT_SWEEP_CONFIG = REPO_ROOT / "configs" / "sweeps" / "heuristic_action_rate_rough_terrain.json"


def load_sweep_config(config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_SWEEP_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    config_path = resolve_path(candidate["config"])
    config = load_config(config_path)
    metrics_path = artifact_dir(config, config["run_name"]) / "metrics.json"
    record: dict[str, Any] = {
        "id": candidate["id"],
        "label": candidate["label"],
        "config_path": relative_to_repo(config_path),
        "experiment_name": config["experiment_name"],
        "run_name": config["run_name"],
        "metrics_path": relative_to_repo(metrics_path),
        "action_smoothness": None,
        "status": "missing_metrics",
    }
    record["action_smoothness"] = config.get("overrides", {}).get("env", {}).get("rewards.scales.action_smoothness")

    if not metrics_path.exists():
        return record

    metrics = read_json(metrics_path)
    record.update(
        {
            "status": "complete",
            "velocity_tracking_error_mean": metrics["velocity_tracking_error_mean"],
            "fall_rate": metrics["fall_rate"],
            "joint_acceleration_l2_mean": metrics["joint_acceleration_l2_mean"],
            "action_jitter_l2_mean": metrics["action_jitter_l2_mean"],
            "episode_return_mean": metrics["episode_return_mean"],
            "policy_local_sensitivity_cost_mean": metrics["constraint_metrics"].get(
                "policy_local_sensitivity_cost_mean"
            ),
        }
    )
    return record


def reference_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        records,
        key=lambda record: (
            record["velocity_tracking_error_mean"],
            record["fall_rate"],
            record["joint_acceleration_l2_mean"],
            record["action_jitter_l2_mean"],
        ),
    )


def apply_task_floor(
    records: list[dict[str, Any]],
    tracking_tolerance: float,
    fall_tolerance: float,
) -> tuple[dict[str, Any], float, float]:
    reference = reference_candidate(records)
    tracking_threshold = reference["velocity_tracking_error_mean"] * (1.0 + tracking_tolerance)
    fall_threshold = reference["fall_rate"] + fall_tolerance
    for record in records:
        record["passes_task_floor"] = bool(
            record["velocity_tracking_error_mean"] <= tracking_threshold and record["fall_rate"] <= fall_threshold
        )
    return reference, tracking_threshold, fall_threshold


def selected_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record.get("passes_task_floor")]
    if not eligible:
        raise RuntimeError("No heuristic candidate satisfies the task-validity floor.")
    return min(
        eligible,
        key=lambda record: (
            record["joint_acceleration_l2_mean"],
            record["action_jitter_l2_mean"],
            record["velocity_tracking_error_mean"],
            record["fall_rate"],
        ),
    )


def output_path(sweep_cfg: dict[str, Any]) -> Path:
    analysis_root = sweep_cfg.get("analysis_root", "artifacts/analysis/heuristic_action_rate_rough_terrain")
    return ensure_directory(resolve_path(analysis_root))


def print_record(record: dict[str, Any]) -> None:
    if record["status"] != "complete":
        print(f"{record['id']}: missing metrics at {record['metrics_path']}")
        return
    passed = "pending"
    if "passes_task_floor" in record:
        passed = "yes" if record["passes_task_floor"] else "no"
    print(
        f"{record['id']}: vel={record['velocity_tracking_error_mean']:.6f} "
        f"fall={record['fall_rate']:.6f} "
        f"joint={record['joint_acceleration_l2_mean']:.6f} "
        f"jitter={record['action_jitter_l2_mean']:.6f} "
        f"pass={passed}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the heuristic sweep winner by task floor then smoothness.")
    parser.add_argument("--sweep-config", default=None, help="Path to the sweep JSON.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a provisional summary even if some candidate metrics are still missing.",
    )
    args = parser.parse_args()

    sweep_cfg = load_sweep_config(args.sweep_config)
    records = [summarize_candidate(candidate) for candidate in sweep_cfg["candidates"]]
    complete_records = [record for record in records if record["status"] == "complete"]
    missing_records = [record for record in records if record["status"] != "complete"]

    if missing_records and not args.allow_incomplete:
        print("Heuristic sweep status")
        for record in records:
            print_record(record)
        print("Sweep is incomplete; run the missing candidates before selecting the baseline.")
        return 2
    if not complete_records:
        print("No completed heuristic candidates found.")
        return 2

    selection_cfg = sweep_cfg["selection"]
    reference, tracking_threshold, fall_threshold = apply_task_floor(
        complete_records,
        tracking_tolerance=float(selection_cfg["tracking_error_relative_tolerance"]),
        fall_tolerance=float(selection_cfg["fall_rate_absolute_tolerance"]),
    )
    chosen = selected_candidate(complete_records)

    print("Heuristic sweep status")
    for record in records:
        print_record(record)

    print()
    print(
        "Task floor: "
        f"velocity_tracking_error_mean <= {tracking_threshold:.6f}, "
        f"fall_rate <= {fall_threshold:.6f}"
    )
    print(f"Reference task candidate: {reference['id']}")
    print(f"Selected heuristic baseline: {chosen['id']}")

    summary = {
        "sweep_name": sweep_cfg["name"],
        "selection_rule": selection_cfg,
        "selection_complete": not missing_records,
        "reference_candidate_id": reference["id"],
        "tracking_threshold": tracking_threshold,
        "fall_threshold": fall_threshold,
        "selected_candidate_id": chosen["id"],
        "selected_candidate": chosen,
        "candidates": records,
    }
    target = output_path(sweep_cfg) / "selection.json"
    write_json(target, summary)
    print(f"Wrote {relative_to_repo(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
