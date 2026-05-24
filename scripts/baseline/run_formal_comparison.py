#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
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
    run_command,
    write_json,
)

REPO_ROOT = repo_root()
DEFAULT_SWEEP_CONFIG = REPO_ROOT / "configs" / "sweeps" / "rough_terrain_formal_comparison.json"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "baseline" / "train_vanilla_ppo.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "baseline" / "evaluate_checkpoint_sweep.py"
COMPARISON_METRIC_KEYS = [
    "velocity_tracking_error_mean",
    "joint_acceleration_l2_mean",
    "action_jitter_l2_mean",
    "episode_return_mean",
    "fall_rate",
]


def load_sweep_config(config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_SWEEP_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def candidate_run_name(base_run_name: str, seed: int) -> str:
    return f"{base_run_name}_seed{seed}"


def train_manifest_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "manifest.json"


def eval_summary_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "checkpoint_sweep_summary.json"


def load_train_manifest(config: dict[str, Any], run_name: str) -> dict[str, Any] | None:
    manifest_path = train_manifest_path(config, run_name)
    if not manifest_path.exists():
        return None
    return read_json(manifest_path)


def train_complete(config: dict[str, Any], run_name: str) -> bool:
    manifest_path = train_manifest_path(config, run_name)
    if not manifest_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
    except json.JSONDecodeError:
        return False
    checkpoint_path = manifest.get("checkpoint_path")
    if not isinstance(checkpoint_path, str):
        return True
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    return checkpoint.exists()


def evaluate_complete(config: dict[str, Any], run_name: str) -> bool:
    return eval_summary_path(config, run_name).exists()


def stage_complete(stage: str, config: dict[str, Any], run_name: str) -> bool:
    if stage == "train":
        return train_complete(config, run_name)
    if stage == "evaluate":
        return evaluate_complete(config, run_name)
    raise ValueError("Unknown stage: {stage}".format(stage=stage))


def build_train_command(
    config_path: Path,
    run_name: str,
    seed: int,
    args: argparse.Namespace,
    *,
    train_num_envs: int | None = None,
    max_iterations: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"--config={relative_to_repo(config_path)}",
        f"--run-name={run_name}",
        f"--seed={seed}",
    ]
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    effective_train_num_envs = args.train_num_envs if args.train_num_envs is not None else train_num_envs
    effective_max_iterations = args.max_iterations if args.max_iterations is not None else max_iterations
    if effective_train_num_envs is not None:
        command.append(f"--num-envs={effective_train_num_envs}")
    if effective_max_iterations is not None:
        command.append(f"--max-iterations={effective_max_iterations}")
    if args.rl_device:
        command.append(f"--rl-device={args.rl_device}")
    if args.sim_device:
        command.append(f"--sim-device={args.sim_device}")
    return command


def build_evaluate_command(
    config_path: Path,
    run_name: str,
    load_run: str,
    seed: int,
    args: argparse.Namespace,
    *,
    eval_num_envs: int | None = None,
    episodes: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        f"--config={relative_to_repo(config_path)}",
        "--run-name",
        run_name,
        "--load-run",
        load_run,
        "--seed",
        str(seed),
    ]
    if args.humanoid_gym_root:
        command.extend(["--humanoid-gym-root", args.humanoid_gym_root])
    effective_eval_num_envs = args.eval_num_envs if args.eval_num_envs is not None else eval_num_envs
    effective_episodes = args.episodes if args.episodes is not None else episodes
    if effective_eval_num_envs is not None:
        command.extend(["--num-envs", str(effective_eval_num_envs)])
    if effective_episodes is not None:
        command.extend(["--episodes", str(effective_episodes)])
    if args.rl_device:
        command.extend(["--rl-device", args.rl_device])
    if args.sim_device:
        command.extend(["--sim-device", args.sim_device])
    return command


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for key in COMPARISON_METRIC_KEYS:
        values = [float(metrics[key]) for metrics in metrics_list]
        aggregate[f"{key}_mean"] = mean(values)
        aggregate[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return aggregate


def latest_metrics_row(rows: Any) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    typed_rows = [row for row in rows if isinstance(row, dict) and "checkpoint" in row]
    if not typed_rows:
        return None
    return max(typed_rows, key=lambda row: int(row["checkpoint"]))


def metrics_snapshot_from_row(row: dict[str, Any] | None) -> dict[str, float] | None:
    if row is None:
        return None
    if any(key not in row for key in COMPARISON_METRIC_KEYS):
        return None
    return {key: float(row[key]) for key in COMPARISON_METRIC_KEYS}


def collect_candidate_summary(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    selected_metrics_list: list[dict[str, Any]] = []
    final_metrics_list: list[dict[str, float]] = []
    selected_checkpoints: dict[str, int] = {}
    final_checkpoints: dict[str, int] = {}
    selection_statuses: dict[str, str] = {}
    missing_seeds: list[int] = []

    for seed in seeds:
        run_name = candidate_run_name(config["run_name"], seed)
        train_manifest = load_train_manifest(config, run_name)
        load_run = run_name
        if train_manifest:
            run_dir = train_manifest.get("run_dir")
            if isinstance(run_dir, str):
                load_run = run_dir
        summary_path = eval_summary_path(config, run_name)
        if not summary_path.exists():
            missing_seeds.append(seed)
            continue
        summary = read_json(summary_path)
        selected_metrics_path = summary.get("selected_metrics_path")
        if not isinstance(selected_metrics_path, str):
            missing_seeds.append(seed)
            continue
        selected_metrics = read_json(REPO_ROOT / selected_metrics_path)
        selection_status = str(summary.get("selection_status", "selected"))
        final_row = latest_metrics_row(summary.get("rows"))
        final_metrics = metrics_snapshot_from_row(final_row)
        per_seed[str(seed)] = {
            "run_name": run_name,
            "load_run": load_run,
            "checkpoint_sweep_summary_path": relative_to_repo(summary_path),
            "selection_status": selection_status,
            "selected_checkpoint": int(summary["best_checkpoint"]),
            "selected_metrics_path": selected_metrics_path,
            "selected_metrics": selected_metrics,
            "final_checkpoint": int(final_row["checkpoint"]) if final_row is not None else None,
            "final_metrics": final_metrics,
        }
        selection_statuses[str(seed)] = selection_status
        selected_checkpoints[str(seed)] = int(summary["best_checkpoint"])
        if final_row is not None:
            final_checkpoints[str(seed)] = int(final_row["checkpoint"])
        selected_metrics_list.append(selected_metrics)
        if final_metrics is not None:
            final_metrics_list.append(final_metrics)

    summary: dict[str, Any] = {
        "id": config["method"]["id"],
        "label": config["method"]["label"],
        "config_path": config["__config_path__"],
        "run_name": config["run_name"],
        "seeds": seeds,
        "missing_seeds": missing_seeds,
        "selected_checkpoints": selected_checkpoints,
        "final_checkpoints": final_checkpoints,
        "selection_statuses": selection_statuses,
        "per_seed": per_seed,
        "status": "complete" if not missing_seeds and len(selected_metrics_list) == len(seeds) else "incomplete",
    }
    if not missing_seeds and selection_statuses and all(status != "selected" for status in selection_statuses.values()):
        summary["status"] = "collapsed"
    if selected_metrics_list:
        summary["aggregate"] = aggregate_metrics(selected_metrics_list)
        summary["selected_aggregate"] = summary["aggregate"]
    if final_metrics_list:
        summary["final_aggregate"] = aggregate_metrics(final_metrics_list)
    return summary


def resolve_eval_load_run(config: dict[str, Any], run_name: str) -> str:
    train_manifest = load_train_manifest(config, run_name)
    if train_manifest:
        run_dir = train_manifest.get("run_dir")
        if isinstance(run_dir, str):
            return run_dir
    return run_name


def write_summary(sweep_cfg: dict[str, Any], candidate_summaries: list[dict[str, Any]]) -> Path:
    output_root = ensure_directory((REPO_ROOT / sweep_cfg.get("analysis_root", "artifacts/analysis/rough_terrain_formal_comparison")).resolve())
    paired_candidates = zip(sweep_cfg["candidates"], candidate_summaries)
    payload = {
        "comparison_name": sweep_cfg["name"],
        "seeds": sweep_cfg["seeds"],
        "train_num_envs": sweep_cfg["train_num_envs"],
        "eval_num_envs": sweep_cfg["eval_num_envs"],
        "max_iterations": sweep_cfg["max_iterations"],
        "episodes": sweep_cfg["episodes"],
        "candidates": candidate_summaries,
        "seed_checkpoint_sweep_paths": {
            candidate["id"]: {
                seed: summary["per_seed"][str(seed)]["checkpoint_sweep_summary_path"]
                for seed in sweep_cfg["seeds"]
                if str(seed) in summary["per_seed"]
            }
            for candidate, summary in paired_candidates
        },
    }
    output_path = output_root / "comparison_summary.json"
    write_json(output_path, payload)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and summarize the rough-terrain formal comparison.")
    parser.add_argument("--sweep-config", default=None, help="Path to the formal comparison sweep JSON.")
    parser.add_argument("--candidate", action="append", default=None, help="Optional candidate id filter.")
    parser.add_argument("--stage", choices=("plan", "train", "evaluate", "all"), default="plan")
    parser.add_argument("--skip-completed", action="store_true", help="Skip stages with completed artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Optional upstream checkout override.")
    parser.add_argument("--train-num-envs", type=int, default=None, help="Override training num_envs.")
    parser.add_argument("--eval-num-envs", type=int, default=None, help="Override evaluation num_envs.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override training max iterations.")
    parser.add_argument("--episodes", type=int, default=None, help="Override evaluation episode count.")
    parser.add_argument("--rl-device", default=None, help="Optional RL device override.")
    parser.add_argument("--sim-device", default=None, help="Optional sim device override.")
    args = parser.parse_args()

    sweep_cfg = load_sweep_config(args.sweep_config)
    candidates = sweep_cfg["candidates"]
    if args.candidate:
        requested = set(args.candidate)
        candidates = [candidate for candidate in candidates if candidate["id"] in requested]
        missing = requested - {candidate["id"] for candidate in candidates}
        if missing:
            raise SystemExit(f"Unknown candidate ids: {sorted(missing)}")

    if args.stage == "plan":
        for candidate in candidates:
            config_path = resolve_config_path(candidate["config"])
            config = load_config(config_path)
            print(f"[{candidate['id']}] {candidate['label']}")
            for seed in sweep_cfg["seeds"]:
                run_name = candidate_run_name(config["run_name"], seed)
                train_status = "complete" if train_complete(config, run_name) else "pending"
                eval_status = "complete" if evaluate_complete(config, run_name) else "pending"
                train_command = " ".join(
                    build_train_command(
                        config_path,
                        run_name,
                        seed,
                        args,
                        train_num_envs=sweep_cfg.get("train_num_envs"),
                        max_iterations=sweep_cfg.get("max_iterations"),
                    )
                )
                eval_command = " ".join(
                    build_evaluate_command(
                        config_path,
                        run_name,
                        resolve_eval_load_run(config, run_name),
                        seed,
                        args,
                        eval_num_envs=sweep_cfg.get("eval_num_envs"),
                        episodes=sweep_cfg.get("episodes"),
                    )
                )
                print(f"  seed {seed}")
                print(f"    train: {train_status}")
                print(f"    command: {train_command}")
                print(f"    evaluate: {eval_status}")
                print(f"    command: {eval_command}")
        return 0

    stages = ["train", "evaluate"] if args.stage == "all" else [args.stage]
    for candidate in candidates:
        config_path = resolve_config_path(candidate["config"])
        config = load_config(config_path)
        config["__config_path__"] = relative_to_repo(config_path)
        print(f"[{candidate['id']}] {candidate['label']}")
        for seed in sweep_cfg["seeds"]:
            run_name = candidate_run_name(config["run_name"], seed)
            for stage in stages:
                complete = stage_complete(stage, config, run_name)
                if args.skip_completed and complete:
                    marker = train_manifest_path(config, run_name) if stage == "train" else eval_summary_path(config, run_name)
                    print(f"Skipping {stage}: {relative_to_repo(marker)} already exists")
                    continue
                command = (
                    build_train_command(
                        config_path,
                        run_name,
                        seed,
                        args,
                        train_num_envs=sweep_cfg.get("train_num_envs"),
                        max_iterations=sweep_cfg.get("max_iterations"),
                    )
                    if stage == "train"
                    else build_evaluate_command(
                        config_path,
                        run_name,
                        resolve_eval_load_run(config, run_name),
                        seed,
                        args,
                        eval_num_envs=sweep_cfg.get("eval_num_envs"),
                        episodes=sweep_cfg.get("episodes"),
                    )
                )
                exit_code = run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run)
                if exit_code != 0:
                    recovered = stage_complete(stage, config, run_name)
                    if recovered:
                        marker = train_manifest_path(config, run_name) if stage == "train" else eval_summary_path(config, run_name)
                        print(
                            f"{stage} exited with code {exit_code} after writing "
                            f"{relative_to_repo(marker)}; continuing.",
                            file=sys.stderr,
                        )
                        continue
                    return exit_code
            print()
        print()

    if not args.dry_run:
        candidate_summaries = []
        for candidate in candidates:
            config_path = resolve_config_path(candidate["config"])
            config = load_config(config_path)
            config["__config_path__"] = relative_to_repo(config_path)
            candidate_summaries.append(collect_candidate_summary(config, sweep_cfg["seeds"]))
        summary_path = write_summary(sweep_cfg, candidate_summaries)
        print(f"Wrote {relative_to_repo(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
