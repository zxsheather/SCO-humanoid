#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    artifact_dir,
    configure_runtime_env,
    ensure_directory,
    latest_checkpoint,
    load_config,
    read_json,
    relative_to_repo,
    repo_root,
    resolve_humanoid_gym_root,
    resolve_run_dir,
    write_json,
)

DEFAULT_TRACKING_TOLERANCE = 0.10
DEFAULT_FALL_TOLERANCE = 0.05
CONSTRAINT_CORRELATION_PAIRS = (
    ("train_constraint_cost_mean", "fall_rate"),
    ("train_constraint_cost_update", "fall_rate"),
    ("train_constraint_cost_mean", "joint_acceleration_l2_mean"),
    ("train_constraint_cost_mean", "action_jitter_l2_mean"),
    ("train_constraint_cost_mean", "velocity_tracking_error_mean"),
    ("train_policy_local_sensitivity_cost_mean", "fall_rate"),
    ("train_policy_local_sensitivity_cost_update", "fall_rate"),
    ("train_policy_local_sensitivity_cost_mean", "joint_acceleration_l2_mean"),
    ("train_policy_local_sensitivity_cost_mean", "action_jitter_l2_mean"),
    ("train_policy_local_sensitivity_cost_mean", "velocity_tracking_error_mean"),
    ("eval_policy_local_sensitivity_cost_mean", "fall_rate"),
    ("eval_policy_local_sensitivity_cost_mean", "joint_acceleration_l2_mean"),
)
TRAIN_CONSTRAINT_FLOAT_KEYS = (
    "lagrange_multiplier",
    "constraint_threshold",
    "constraint_cost_mean",
    "constraint_cost_update",
    "constraint_effective_cost_update",
    "constraint_cost_max",
    "constraint_cost_quantile",
    "constraint_legacy_cost_mean",
    "constraint_legacy_cost_update",
    "constraint_legacy_cost_max",
    "constraint_legacy_cost_quantile",
    "policy_local_sensitivity_cost_mean",
    "policy_local_sensitivity_cost_update",
    "policy_local_sensitivity_effective_cost_update",
    "policy_local_sensitivity_legacy_cost_mean",
    "policy_local_sensitivity_legacy_cost_update",
    "action_rate_cost_mean",
    "action_rate_cost_update",
    "action_rate_effective_cost_update",
    "action_rate_cost_max",
    "action_rate_cost_quantile",
    "action_rate_legacy_cost_mean",
    "action_rate_legacy_cost_update",
    "constraint_violation_rate",
    "constraint_legacy_violation_rate",
    "constraint_penalty_error",
    "constraint_update_error",
)
TRAIN_CONSTRAINT_STRING_KEYS = (
    "constraint_objective",
    "constraint_effective_mode",
    "constraint_penalty_mode",
    "constraint_update_error_mode",
    "constraint_legacy_guard_mode",
)
ALIGNMENT_RANGE_KEYS = (
    "fall_rate",
    "velocity_tracking_error_mean",
    "joint_acceleration_l2_mean",
    "action_jitter_l2_mean",
    "episode_return_mean",
    "eval_constraint_cost_mean",
    "eval_policy_local_sensitivity_cost_mean",
    "eval_constraint_violation_rate",
    "train_constraint_cost_mean",
    "train_constraint_cost_update",
    "train_constraint_effective_cost_update",
    "train_constraint_legacy_cost_mean",
    "train_constraint_legacy_cost_update",
    "train_policy_local_sensitivity_cost_mean",
    "train_policy_local_sensitivity_cost_update",
    "train_policy_local_sensitivity_effective_cost_update",
    "train_policy_local_sensitivity_legacy_cost_mean",
    "train_policy_local_sensitivity_legacy_cost_update",
    "train_constraint_violation_rate",
    "train_constraint_legacy_violation_rate",
    "train_lagrange_multiplier",
)


def parse_checkpoint_list(raw: str | None, run_dir: Path) -> list[int]:
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    checkpoints = sorted(int(path.stem.split("_")[-1]) for path in run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")
    return checkpoints


def composite_score(metrics: dict[str, Any]) -> float:
    return float(metrics["joint_acceleration_l2_mean"]) + 100.0 * float(metrics["velocity_tracking_error_mean"])


def all_checkpoints_collapsed(rows: list[dict[str, Any]], *, collapse_fall_rate: float = 1.0, atol: float = 1e-9) -> bool:
    fall_rates = [row.get("fall_rate") for row in rows]
    valid = [float(rate) for rate in fall_rates if rate is not None]
    return bool(valid) and all(rate >= collapse_fall_rate - atol for rate in valid)


def selection_status(rows: list[dict[str, Any]]) -> str:
    if all_checkpoints_collapsed(rows):
        return "all_checkpoints_collapsed"
    return "selected"


def task_reference_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            float(row["velocity_tracking_error_mean"]),
            float(row["fall_rate"]),
            float(row["joint_acceleration_l2_mean"]),
            float(row["action_jitter_l2_mean"]),
        ),
    )


def eligible_rows(
    rows: list[dict[str, Any]],
    *,
    tracking_tolerance: float = DEFAULT_TRACKING_TOLERANCE,
    fall_tolerance: float = DEFAULT_FALL_TOLERANCE,
) -> tuple[dict[str, Any], float, float, list[dict[str, Any]]]:
    reference = task_reference_row(rows)
    tracking_threshold = float(reference["velocity_tracking_error_mean"]) * (1.0 + tracking_tolerance)
    fall_threshold = float(reference["fall_rate"]) + fall_tolerance
    eligible = [
        row
        for row in rows
        if float(row["velocity_tracking_error_mean"]) <= tracking_threshold
        and float(row["fall_rate"]) <= fall_threshold
    ]
    return reference, tracking_threshold, fall_threshold, eligible


def selected_row(
    rows: list[dict[str, Any]],
    *,
    tracking_tolerance: float = DEFAULT_TRACKING_TOLERANCE,
    fall_tolerance: float = DEFAULT_FALL_TOLERANCE,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if all_checkpoints_collapsed(rows):
        best = min(rows, key=lambda row: float(row["composite_score"]))
        return best, None
    reference, tracking_threshold, fall_threshold, eligible = eligible_rows(
        rows,
        tracking_tolerance=tracking_tolerance,
        fall_tolerance=fall_tolerance,
    )
    best = min(
        eligible,
        key=lambda row: (
            float(row["joint_acceleration_l2_mean"]),
            float(row["action_jitter_l2_mean"]),
            float(row["velocity_tracking_error_mean"]),
            float(row["fall_rate"]),
        ),
    )
    return best, {
        "reference_checkpoint": int(reference["checkpoint"]),
        "tracking_error_relative_tolerance": tracking_tolerance,
        "fall_rate_absolute_tolerance": fall_tolerance,
        "tracking_threshold": tracking_threshold,
        "fall_threshold": fall_threshold,
        "eligible_checkpoints": [int(row["checkpoint"]) for row in eligible],
    }


def manifest_matches_checkpoint(manifest: dict[str, Any], checkpoint: int) -> bool:
    checkpoint_path = manifest.get("checkpoint_path")
    if not isinstance(checkpoint_path, str):
        return False
    return Path(checkpoint_path).name == f"model_{checkpoint}.pt"


def recover_checkpoint_metrics(output_dir: Path, checkpoint: int) -> dict[str, Any] | None:
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"
    if not metrics_path.exists() or not manifest_path.exists():
        return None

    manifest = read_json(manifest_path)
    if not manifest_matches_checkpoint(manifest, checkpoint):
        return None
    return read_json(metrics_path)


def existing_checkpoint_metrics(output_dir: Path, checkpoint: int) -> dict[str, Any] | None:
    checkpoint_metrics_path = output_dir / f"metrics_checkpoint_{checkpoint}.json"
    if not checkpoint_metrics_path.exists():
        return None
    return read_json(checkpoint_metrics_path)


def checkpoint_path(run_dir: Path, checkpoint: int) -> Path:
    return run_dir / f"model_{checkpoint}.pt"


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def first_numeric(*values: Any) -> float | None:
    for value in values:
        parsed = numeric_value(value)
        if parsed is not None:
            return parsed
    return None


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def checkpoint_train_constraint_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    alg_extra_state = loaded.get("alg_extra_state_dict")
    if not isinstance(alg_extra_state, dict):
        return {}
    latest_stats = alg_extra_state.get("latest_stats")
    if not isinstance(latest_stats, dict):
        return {}

    train_metrics: dict[str, Any] = {}
    for key in TRAIN_CONSTRAINT_FLOAT_KEYS:
        train_metrics[f"train_{key}"] = numeric_value(latest_stats.get(key))
    for key in TRAIN_CONSTRAINT_STRING_KEYS:
        value = latest_stats.get(key)
        train_metrics[f"train_{key}"] = str(value) if value is not None else None

    constraint_trace = alg_extra_state.get("constraint_trace")
    if isinstance(constraint_trace, list):
        train_metrics["train_constraint_trace_length"] = len(constraint_trace)
    else:
        train_metrics["train_constraint_trace_length"] = None
    return train_metrics


def metric_range(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values: list[tuple[int, float]] = []
    for row in rows:
        value = numeric_value(row.get(key))
        if value is None:
            continue
        values.append((int(row["checkpoint"]), value))

    if not values:
        return None

    min_checkpoint, min_value = min(values, key=lambda item: item[1])
    max_checkpoint, max_value = max(values, key=lambda item: item[1])
    return {
        "count": len(values),
        "min": min_value,
        "max": max_value,
        "delta": max_value - min_value,
        "argmin_checkpoint": min_checkpoint,
        "argmax_checkpoint": max_checkpoint,
    }


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    variance_x = sum(value * value for value in centered_x)
    variance_y = sum(value * value for value in centered_y)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    covariance = sum(x_value * y_value for x_value, y_value in zip(centered_x, centered_y))
    return covariance / math.sqrt(variance_x * variance_y)


def correlation_summary(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    pairs: list[tuple[int, float, float]] = []
    for row in rows:
        x_value = numeric_value(row.get(x_key))
        y_value = numeric_value(row.get(y_key))
        if x_value is None or y_value is None:
            continue
        pairs.append((int(row["checkpoint"]), x_value, y_value))

    if len(pairs) < 2:
        return {"pair_count": len(pairs), "pearson": None, "reason": "insufficient_pairs"}

    x_values = [item[1] for item in pairs]
    y_values = [item[2] for item in pairs]
    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)
    if x_span <= 1e-12:
        return {"pair_count": len(pairs), "pearson": None, "reason": f"{x_key}_constant"}
    if y_span <= 1e-12:
        return {"pair_count": len(pairs), "pearson": None, "reason": f"{y_key}_constant"}

    return {
        "pair_count": len(pairs),
        "pearson": pearson_correlation(x_values, y_values),
        "x_min": min(x_values),
        "x_max": max(x_values),
        "y_min": min(y_values),
        "y_max": max(y_values),
    }


def build_alignment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available": bool(rows),
        "row_count": len(rows),
        "all_eval_checkpoints_collapsed": all_checkpoints_collapsed(rows),
        "rows_with_train_constraint_metrics": sum(
            1
            for row in rows
            if first_numeric(
                row.get("train_constraint_cost_mean"),
                row.get("train_policy_local_sensitivity_cost_mean"),
                row.get("train_action_rate_cost_mean"),
            )
            is not None
        ),
        "ranges": {},
        "correlations": {},
    }
    for key in ALIGNMENT_RANGE_KEYS:
        range_summary = metric_range(rows, key)
        if range_summary is not None:
            summary["ranges"][key] = range_summary
    for x_key, y_key in CONSTRAINT_CORRELATION_PAIRS:
        summary["correlations"][f"{x_key}__vs__{y_key}"] = correlation_summary(rows, x_key, y_key)

    train_cost_range = (
        summary["ranges"].get("train_constraint_cost_mean")
        or summary["ranges"].get("train_policy_local_sensitivity_cost_mean")
        or summary["ranges"].get("train_action_rate_cost_mean")
    )
    train_update_range = (
        summary["ranges"].get("train_constraint_cost_update")
        or summary["ranges"].get("train_policy_local_sensitivity_cost_update")
        or summary["ranges"].get("train_action_rate_cost_update")
    )
    summary["collapsed_task_floor_diagnostic"] = {
        "all_eval_checkpoints_collapsed": summary["all_eval_checkpoints_collapsed"],
        "train_constraint_metrics_available": summary["rows_with_train_constraint_metrics"] > 0,
        "train_cost_mean_moves": bool(train_cost_range and train_cost_range["delta"] > 0.0),
        "train_cost_update_moves": bool(train_update_range and train_update_range["delta"] > 0.0),
    }
    return summary


def build_evaluate_policy_command(
    *,
    config_path: str | None,
    run_name: str,
    load_run: str,
    checkpoint: int,
    humanoid_gym_root: str | None,
    num_envs: int | None,
    episodes: int | None,
    rl_device: str | None,
    sim_device: str | None,
    seed: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_policy.py"),
        "--run-name",
        run_name,
        "--load-run",
        load_run,
        "--checkpoint",
        str(checkpoint),
    ]
    if config_path:
        command.extend(["--config", config_path])
    if humanoid_gym_root:
        command.extend(["--humanoid-gym-root", humanoid_gym_root])
    if num_envs is not None:
        command.extend(["--num-envs", str(num_envs)])
    if episodes is not None:
        command.extend(["--episodes", str(episodes)])
    if rl_device is not None:
        command.extend(["--rl-device", rl_device])
    if sim_device is not None:
        command.extend(["--sim-device", sim_device])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def evaluate_one_checkpoint(
    *,
    config_path: str | None,
    run_name: str,
    load_run: str,
    checkpoint: int,
    checkpoint_file: Path,
    humanoid_gym_root: str | None,
    num_envs: int | None,
    episodes: int | None,
    rl_device: str | None,
    sim_device: str | None,
    seed: int | None,
) -> dict[str, Any]:
    # Isaac Gym/PhysX cannot be safely reinitialized repeatedly inside one Python process.
    # Run each checkpoint evaluation in a fresh subprocess so longer-budget sweeps are stable.
    configure_runtime_env()
    config = load_config(config_path)
    output_dir = artifact_dir(config, run_name)
    command = build_evaluate_policy_command(
        config_path=config_path,
        run_name=run_name,
        load_run=load_run,
        checkpoint=checkpoint,
        humanoid_gym_root=humanoid_gym_root,
        num_envs=num_envs,
        episodes=episodes,
        rl_device=rl_device,
        sim_device=sim_device,
        seed=seed,
    )
    completed = subprocess.run(command, cwd=str(repo_root()), check=False)
    recovered_metrics = None
    if completed.returncode != 0:
        recovered_metrics = recover_checkpoint_metrics(output_dir, checkpoint)
        if recovered_metrics is None:
            raise RuntimeError(f"evaluate_policy failed for checkpoint {checkpoint} with exit code {completed.returncode}")
        print(
            f"Checkpoint {checkpoint} exited with code {completed.returncode} after writing artifacts; "
            "continuing with recovered metrics.",
            file=sys.stderr,
        )

    metrics = recovered_metrics if recovered_metrics is not None else read_json(output_dir / "metrics.json")
    checkpoint_metrics_path = output_dir / f"metrics_checkpoint_{checkpoint}.json"
    write_json(checkpoint_metrics_path, metrics)
    return {
        "checkpoint": checkpoint,
        "checkpoint_file": checkpoint_file,
        "metrics_path": relative_to_repo(checkpoint_metrics_path),
        "metrics": metrics,
        "composite_score": composite_score(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints for one run and write a summary.")
    parser.add_argument("--config", default=None, help="Path to the method config JSON.")
    parser.add_argument("--run-name", required=True, help="Artifact run name to update and store checkpoint metrics under.")
    parser.add_argument("--load-run", required=True, help="Explicit upstream run directory name.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Optional upstream checkout override.")
    parser.add_argument("--checkpoints", default=None, help="Comma-separated checkpoint ids. Default: all model_*.pt in load-run.")
    parser.add_argument("--episodes", type=int, default=None, help="Override the number of completed episodes.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override the evaluation environment count.")
    parser.add_argument("--rl-device", default=None, help="Override the configured RL device.")
    parser.add_argument("--sim-device", default=None, help="Override the configured sim device.")
    parser.add_argument("--seed", type=int, default=None, help="Override the evaluation seed.")
    parser.add_argument(
        "--reuse-existing-metrics",
        action="store_true",
        help="Reuse existing metrics_checkpoint_<N>.json files when present instead of reevaluating checkpoints.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    run_dir = resolve_run_dir(humanoid_gym_root, config, run_name=args.run_name, load_run=args.load_run)
    checkpoints = parse_checkpoint_list(args.checkpoints, run_dir)

    results = []
    output_dir = ensure_directory(artifact_dir(config, args.run_name))
    for checkpoint in checkpoints:
        if args.reuse_existing_metrics:
            metrics = existing_checkpoint_metrics(output_dir, checkpoint)
            if metrics is not None:
                results.append(
                    {
                        "checkpoint": checkpoint,
                        "checkpoint_file": checkpoint_path(run_dir, checkpoint),
                        "metrics_path": relative_to_repo(output_dir / f"metrics_checkpoint_{checkpoint}.json"),
                        "metrics": metrics,
                        "composite_score": composite_score(metrics),
                    }
                )
                continue
        results.append(
            evaluate_one_checkpoint(
                config_path=args.config,
                run_name=args.run_name,
                load_run=args.load_run,
                checkpoint=checkpoint,
                checkpoint_file=checkpoint_path(run_dir, checkpoint),
                humanoid_gym_root=args.humanoid_gym_root,
                num_envs=args.num_envs,
                episodes=args.episodes,
                rl_device=args.rl_device,
                sim_device=args.sim_device,
                seed=args.seed,
            )
        )

    summary_rows = []
    for item in results:
        metrics = item["metrics"]
        constraint_metrics = metrics.get("constraint_metrics", {})
        train_constraint_metrics = checkpoint_train_constraint_metrics(Path(item["checkpoint_file"]))
        eval_constraint_cost_mean = first_numeric(
            constraint_metrics.get("constraint_cost_mean"),
            constraint_metrics.get("policy_local_sensitivity_cost_mean"),
            constraint_metrics.get("action_rate_cost_mean"),
        )
        eval_constraint_violation_rate = first_numeric(
            constraint_metrics.get("constraint_violation_rate"),
        )
        summary_rows.append(
            {
                "checkpoint": item["checkpoint"],
                "composite_score": item["composite_score"],
                "velocity_tracking_error_mean": metrics.get("velocity_tracking_error_mean"),
                "joint_acceleration_l2_mean": metrics.get("joint_acceleration_l2_mean"),
                "action_jitter_l2_mean": metrics.get("action_jitter_l2_mean"),
                "episode_return_mean": metrics.get("episode_return_mean"),
                "fall_rate": metrics.get("fall_rate"),
                "policy_local_sensitivity_cost_mean": constraint_metrics.get("policy_local_sensitivity_cost_mean"),
                "constraint_violation_rate": constraint_metrics.get("constraint_violation_rate"),
                "constraint_objective": first_present(
                    constraint_metrics.get("constraint_objective"),
                    train_constraint_metrics.get("train_constraint_objective"),
                ),
                "eval_constraint_cost_mean": eval_constraint_cost_mean,
                "eval_policy_local_sensitivity_cost_mean": constraint_metrics.get("policy_local_sensitivity_cost_mean"),
                "eval_constraint_violation_rate": eval_constraint_violation_rate,
                "metrics_path": item["metrics_path"],
                **train_constraint_metrics,
            }
        )

    best, task_floor = selected_row(summary_rows)
    status = selection_status(summary_rows)
    alignment_summary = build_alignment_summary(summary_rows)
    selected_metrics_path = output_dir / "metrics_selected.json"
    alignment_summary_path = output_dir / "checkpoint_diagnostic_alignment.json"
    write_json(selected_metrics_path, next(item["metrics"] for item in results if item["checkpoint"] == best["checkpoint"]))
    write_json(alignment_summary_path, alignment_summary)
    summary = {
        "run_name": args.run_name,
        "load_run": args.load_run,
        "evaluated_checkpoints": checkpoints,
        "selection_metric": (
            "task floor then smoothness"
            if task_floor is not None
            else "joint_acceleration_l2_mean + 100 * velocity_tracking_error_mean"
        ),
        "task_floor": task_floor,
        "selection_status": status,
        "all_checkpoints_collapsed": status == "all_checkpoints_collapsed",
        "best_checkpoint": best["checkpoint"],
        "best_composite_score": best["composite_score"],
        "selected_checkpoint_metrics_path": best["metrics_path"],
        "selected_metrics_path": relative_to_repo(selected_metrics_path),
        "checkpoint_diagnostic_alignment_path": relative_to_repo(alignment_summary_path),
        "checkpoint_diagnostic_alignment": alignment_summary,
        "rows": summary_rows,
        "latest_checkpoint_path": relative_to_repo(latest_checkpoint(run_dir)),
    }
    summary_path = output_dir / "checkpoint_sweep_summary.json"
    write_json(summary_path, summary)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest["checkpoint_sweep_summary_path"] = relative_to_repo(summary_path)
        manifest["selection_status"] = status
        manifest["all_checkpoints_collapsed"] = status == "all_checkpoints_collapsed"
        manifest["selected_checkpoint"] = best["checkpoint"]
        manifest["selected_checkpoint_metrics_path"] = best["metrics_path"]
        manifest["selected_metrics_path"] = relative_to_repo(selected_metrics_path)
        manifest["checkpoint_diagnostic_alignment_path"] = relative_to_repo(alignment_summary_path)
        write_json(manifest_path, manifest)

    if status != "selected":
        print(
            "Checkpoint sweep warning: every evaluated checkpoint has fall_rate = 1.0; "
            "selected checkpoint is recorded for analysis only.",
            file=sys.stderr,
        )
    print(f"Wrote {relative_to_repo(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
