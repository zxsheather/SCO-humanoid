#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    artifact_dir,
    ensure_directory,
    latest_checkpoint,
    load_config,
    read_json,
    resolve_humanoid_gym_root,
    resolve_run_dir,
    write_json,
)
from evaluate_policy import main as evaluate_policy_main  # noqa: E402


def parse_checkpoint_list(raw: str | None, run_dir: Path) -> list[int]:
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    checkpoints = sorted(int(path.stem.split("_")[-1]) for path in run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")
    return checkpoints


def composite_score(metrics: dict[str, Any]) -> float:
    return float(metrics["joint_acceleration_l2_mean"]) + 100.0 * float(metrics["velocity_tracking_error_mean"])


def evaluate_one_checkpoint(
    *,
    config_path: str | None,
    run_name: str,
    load_run: str,
    checkpoint: int,
    num_envs: int | None,
    episodes: int | None,
    rl_device: str | None,
    sim_device: str | None,
    seed: int | None,
) -> dict[str, Any]:
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "evaluate_policy.py",
            *(["--config", config_path] if config_path else []),
            "--run-name",
            run_name,
            "--load-run",
            load_run,
            "--checkpoint",
            str(checkpoint),
        ]
        if num_envs is not None:
            sys.argv.extend(["--num-envs", str(num_envs)])
        if episodes is not None:
            sys.argv.extend(["--episodes", str(episodes)])
        if rl_device is not None:
            sys.argv.extend(["--rl-device", rl_device])
        if sim_device is not None:
            sys.argv.extend(["--sim-device", sim_device])
        if seed is not None:
            sys.argv.extend(["--seed", str(seed)])
        rc = evaluate_policy_main()
        if rc != 0:
            raise RuntimeError(f"evaluate_policy failed for checkpoint {checkpoint} with exit code {rc}")
    finally:
        sys.argv = original_argv

    config = load_config(config_path)
    output_dir = artifact_dir(config, run_name)
    metrics = read_json(output_dir / "metrics.json")
    checkpoint_metrics_path = output_dir / f"metrics_checkpoint_{checkpoint}.json"
    write_json(checkpoint_metrics_path, metrics)
    return {
        "checkpoint": checkpoint,
        "metrics_path": str(checkpoint_metrics_path),
        "metrics": metrics,
        "composite_score": composite_score(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints for one run and write a summary.")
    parser.add_argument("--config", default=None, help="Path to the method config JSON.")
    parser.add_argument("--run-name", required=True, help="Artifact run name to update and store checkpoint metrics under.")
    parser.add_argument("--load-run", required=True, help="Explicit upstream run directory name.")
    parser.add_argument("--checkpoints", default=None, help="Comma-separated checkpoint ids. Default: all model_*.pt in load-run.")
    parser.add_argument("--episodes", type=int, default=None, help="Override the number of completed episodes.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override the evaluation environment count.")
    parser.add_argument("--rl-device", default=None, help="Override the configured RL device.")
    parser.add_argument("--sim-device", default=None, help="Override the configured sim device.")
    parser.add_argument("--seed", type=int, default=None, help="Override the evaluation seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, None)
    run_dir = resolve_run_dir(humanoid_gym_root, config, run_name=args.run_name, load_run=args.load_run)
    checkpoints = parse_checkpoint_list(args.checkpoints, run_dir)

    results = []
    for checkpoint in checkpoints:
        results.append(
            evaluate_one_checkpoint(
                config_path=args.config,
                run_name=args.run_name,
                load_run=args.load_run,
                checkpoint=checkpoint,
                num_envs=args.num_envs,
                episodes=args.episodes,
                rl_device=args.rl_device,
                sim_device=args.sim_device,
                seed=args.seed,
            )
        )

    output_dir = ensure_directory(artifact_dir(config, args.run_name))
    summary_rows = []
    for item in results:
        metrics = item["metrics"]
        constraint_metrics = metrics.get("constraint_metrics", {})
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
                "metrics_path": item["metrics_path"],
            }
        )

    best = min(summary_rows, key=lambda row: row["composite_score"])
    summary = {
        "run_name": args.run_name,
        "load_run": args.load_run,
        "evaluated_checkpoints": checkpoints,
        "selection_metric": "joint_acceleration_l2_mean + 100 * velocity_tracking_error_mean",
        "best_checkpoint": best["checkpoint"],
        "best_composite_score": best["composite_score"],
        "rows": summary_rows,
        "latest_checkpoint_path": str(latest_checkpoint(run_dir)),
    }
    summary_path = output_dir / "checkpoint_sweep_summary.json"
    write_json(summary_path, summary)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
