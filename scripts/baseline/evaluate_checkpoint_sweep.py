#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def parse_checkpoint_list(raw: str | None, run_dir: Path) -> list[int]:
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    checkpoints = sorted(int(path.stem.split("_")[-1]) for path in run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")
    return checkpoints


def composite_score(metrics: dict[str, Any]) -> float:
    return float(metrics["joint_acceleration_l2_mean"]) + 100.0 * float(metrics["velocity_tracking_error_mean"])


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


def build_evaluate_policy_command(
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
        "metrics_path": relative_to_repo(checkpoint_metrics_path),
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
    selected_metrics_path = output_dir / "metrics_selected.json"
    write_json(selected_metrics_path, next(item["metrics"] for item in results if item["checkpoint"] == best["checkpoint"]))
    summary = {
        "run_name": args.run_name,
        "load_run": args.load_run,
        "evaluated_checkpoints": checkpoints,
        "selection_metric": "joint_acceleration_l2_mean + 100 * velocity_tracking_error_mean",
        "best_checkpoint": best["checkpoint"],
        "best_composite_score": best["composite_score"],
        "selected_checkpoint_metrics_path": best["metrics_path"],
        "selected_metrics_path": relative_to_repo(selected_metrics_path),
        "rows": summary_rows,
        "latest_checkpoint_path": relative_to_repo(latest_checkpoint(run_dir)),
    }
    summary_path = output_dir / "checkpoint_sweep_summary.json"
    write_json(summary_path, summary)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest["checkpoint_sweep_summary_path"] = relative_to_repo(summary_path)
        manifest["selected_checkpoint"] = best["checkpoint"]
        manifest["selected_checkpoint_metrics_path"] = best["metrics_path"]
        manifest["selected_metrics_path"] = relative_to_repo(selected_metrics_path)
        write_json(manifest_path, manifest)

    print(f"Wrote {relative_to_repo(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
