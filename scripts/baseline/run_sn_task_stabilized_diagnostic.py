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
    run_command,
    write_json,
)
import run_sn_diagnostic as sn_diagnostic  # noqa: E402


REPO_ROOT = repo_root()
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "methods"
    / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_sn_first_hidden.json"
)
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "artifacts" / "analysis" / "sn_task_stabilized_diagnostic"
REFERENCE_CONFIG = "configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json"
PRESETS = sn_diagnostic.PRESETS


def resolve_config_path(raw: str | Path | None) -> Path:
    if raw is None:
        return DEFAULT_CONFIG
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def preset_value(args: argparse.Namespace, key: str) -> int:
    return sn_diagnostic.preset_value(args, key)


def run_name_for(config: dict[str, Any], args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    seed = preset_value(args, "seed")
    return f"{config['run_name']}_{args.preset}_seed{seed}"


def train_manifest_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "manifest.json"


def eval_summary_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "checkpoint_sweep_summary.json"


def train_complete(config: dict[str, Any], run_name: str) -> bool:
    return sn_diagnostic.train_complete(config, run_name)


def evaluate_complete(config: dict[str, Any], run_name: str) -> bool:
    return sn_diagnostic.evaluate_complete(config, run_name)


def build_train_command(config_path: Path, run_name: str, args: argparse.Namespace) -> list[str]:
    return sn_diagnostic.build_train_command(config_path, run_name, args)


def build_evaluate_command(
    config: dict[str, Any],
    config_path: Path,
    run_name: str,
    args: argparse.Namespace,
) -> list[str]:
    return sn_diagnostic.build_evaluate_command(config, config_path, run_name, args)


def collect_summary(config: dict[str, Any], run_name: str, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = eval_summary_path(config, run_name)
    manifest_path = train_manifest_path(config, run_name)
    payload: dict[str, Any] = {
        "diagnostic": "sn_task_stabilized_feasibility",
        "scope": "任务稳定化 SN 配方诊断",
        "run_name": run_name,
        "preset": args.preset,
        "train_num_envs": preset_value(args, "train_num_envs"),
        "max_iterations": preset_value(args, "max_iterations"),
        "eval_num_envs": preset_value(args, "eval_num_envs"),
        "episodes": preset_value(args, "episodes"),
        "seed": preset_value(args, "seed"),
        "reference_config": REFERENCE_CONFIG,
        "train_manifest_path": relative_to_repo(manifest_path) if manifest_path.exists() else None,
        "checkpoint_sweep_summary_path": relative_to_repo(summary_path) if summary_path.exists() else None,
        "status": "evaluated" if summary_path.exists() else "train_only" if manifest_path.exists() else "planned",
        "interpretation_boundary": [
            "This diagnostic checks whether first-hidden-layer actor SN can coexist with the current SC-PPO 3.8 recipe.",
            "It must not be read as SN replacing the SC-PPO constraint path or as a new formal mainline.",
            "Do not spend three-seed or MuJoCo terminal budget unless this single-seed diagnostic remains task-valid.",
        ],
    }
    if summary_path.exists():
        sweep = read_json(summary_path)
        payload["selection_status"] = sweep.get("selection_status")
        payload["selected_checkpoint"] = sweep.get("best_checkpoint")
        payload["selected_metrics_path"] = sweep.get("selected_metrics_path")
        rows = sweep.get("rows")
        if isinstance(rows, list) and rows:
            payload["selected_row"] = next(
                (row for row in rows if row.get("checkpoint") == sweep.get("best_checkpoint")),
                rows[0],
            )
    return payload


def write_analysis_summary(config: dict[str, Any], run_name: str, args: argparse.Namespace) -> Path:
    output_root = ensure_directory(Path(args.analysis_root).expanduser().resolve())
    output_path = output_root / f"{run_name}_summary.json"
    write_json(output_path, collect_summary(config, run_name, args))
    return output_path


def print_plan(config: dict[str, Any], config_path: Path, run_name: str, args: argparse.Namespace) -> None:
    print(f"Task-stabilized SN diagnostic preset: {args.preset}")
    print(f"config: {relative_to_repo(config_path)}")
    print(f"run_name: {run_name}")
    print(f"train: {'complete' if train_complete(config, run_name) else 'pending'}")
    print("command: " + " ".join(build_train_command(config_path, run_name, args)))
    print(f"evaluate: {'complete' if evaluate_complete(config, run_name) else 'pending'}")
    print("command: " + " ".join(build_evaluate_command(config, config_path, run_name, args)))
    print(f"analysis summary: {relative_to_repo(Path(args.analysis_root) / f'{run_name}_summary.json')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded task-stabilized SN diagnostic on top of SC-PPO 3.8.")
    parser.add_argument("--config", default=None, help="Path to the task-stabilized method config JSON.")
    parser.add_argument("--run-name", default=None, help="Override the artifact and upstream run name.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke", help="Diagnostic budget preset.")
    parser.add_argument("--stage", choices=("plan", "train", "evaluate", "all"), default="plan")
    parser.add_argument("--skip-completed", action="store_true", help="Skip completed stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Optional upstream checkout override.")
    parser.add_argument("--train-num-envs", type=int, default=None, help="Override preset training num_envs.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override preset training iterations.")
    parser.add_argument("--eval-num-envs", type=int, default=None, help="Override preset evaluation num_envs.")
    parser.add_argument("--episodes", type=int, default=None, help="Override preset evaluation episode count.")
    parser.add_argument("--seed", type=int, default=None, help="Override preset seed.")
    parser.add_argument("--rl-device", default=None, help="Optional RL device override.")
    parser.add_argument("--sim-device", default=None, help="Optional sim device override.")
    parser.add_argument(
        "--analysis-root",
        default=str(DEFAULT_ANALYSIS_ROOT),
        help="Directory for compact task-stabilized summaries.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    run_name = run_name_for(config, args)

    if args.stage == "plan":
        print_plan(config, config_path, run_name, args)
        return 0

    stages = ["train", "evaluate"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "train":
            marker = train_manifest_path(config, run_name)
            complete = train_complete(config, run_name)
            command = build_train_command(config_path, run_name, args)
        else:
            marker = eval_summary_path(config, run_name)
            complete = evaluate_complete(config, run_name)
            command = build_evaluate_command(config, config_path, run_name, args)

        if args.skip_completed and complete:
            print(f"Skipping {stage}: {relative_to_repo(marker)} already exists")
            continue

        exit_code = run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run)
        if exit_code != 0:
            recovered = train_complete(config, run_name) if stage == "train" else evaluate_complete(config, run_name)
            if recovered:
                print(
                    f"{stage} exited with code {exit_code} after writing {relative_to_repo(marker)}; continuing.",
                    file=sys.stderr,
                )
                continue
            return exit_code

    if not args.dry_run:
        summary_path = write_analysis_summary(config, run_name, args)
        print(f"Wrote {relative_to_repo(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
