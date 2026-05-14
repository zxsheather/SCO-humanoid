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

from _common import artifact_dir, load_config, read_json, repo_root, relative_to_repo, run_command  # noqa: E402

REPO_ROOT = repo_root()
DEFAULT_SWEEP_CONFIG = REPO_ROOT / "configs" / "sweeps" / "heuristic_action_rate_rough_terrain.json"
STAGE_TO_SCRIPT = {
    "train": REPO_ROOT / "scripts" / "baseline" / "train_vanilla_ppo.py",
    "evaluate": REPO_ROOT / "scripts" / "baseline" / "evaluate_policy.py",
}


def load_sweep_config(config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_SWEEP_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def selected_candidates(sweep_cfg: dict[str, Any], candidate_ids: list[str] | None) -> list[dict[str, Any]]:
    candidates = sweep_cfg["candidates"]
    if not candidate_ids:
        return candidates
    requested = set(candidate_ids)
    filtered = [candidate for candidate in candidates if candidate["id"] in requested]
    missing = requested - {candidate["id"] for candidate in filtered}
    if missing:
        raise SystemExit(f"Unknown candidate ids: {sorted(missing)}")
    return filtered


def stage_marker_path(config: dict[str, Any], stage: str) -> Path:
    output_dir = artifact_dir(config, config["run_name"])
    if stage == "train":
        return output_dir / "manifest.json"
    if stage == "evaluate":
        return output_dir / "metrics.json"
    raise ValueError(f"Unsupported stage: {stage}")


def train_stage_complete(config: dict[str, Any]) -> bool:
    manifest_path = stage_marker_path(config, "train")
    if not manifest_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
    except json.JSONDecodeError:
        return False
    checkpoint_path = manifest.get("checkpoint_path")
    if not checkpoint_path:
        return True
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    return checkpoint.exists()


def stage_complete(config: dict[str, Any], stage: str) -> bool:
    if stage == "train":
        return train_stage_complete(config)
    return stage_marker_path(config, stage).exists()


def build_command(stage: str, config_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(STAGE_TO_SCRIPT[stage]),
        f"--config={relative_to_repo(config_path)}",
    ]
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    if args.rl_device:
        command.append(f"--rl-device={args.rl_device}")
    if args.sim_device:
        command.append(f"--sim-device={args.sim_device}")
    if args.seed is not None:
        command.append(f"--seed={args.seed}")
    if stage == "train":
        if args.train_num_envs is not None:
            command.append(f"--num-envs={args.train_num_envs}")
        if args.max_iterations is not None:
            command.append(f"--max-iterations={args.max_iterations}")
    elif stage == "evaluate":
        if args.eval_num_envs is not None:
            command.append(f"--num-envs={args.eval_num_envs}")
        if args.episodes is not None:
            command.append(f"--episodes={args.episodes}")
        if args.checkpoint is not None:
            command.append(f"--checkpoint={args.checkpoint}")
    return command


def print_plan(candidate: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> None:
    config_path = resolve_config_path(candidate["config"])
    print(f"[{candidate['id']}] {candidate['label']}")
    print(f"  config: {relative_to_repo(config_path)}")
    for stage in ("train", "evaluate"):
        status = "complete" if stage_complete(config, stage) else "pending"
        command = " ".join(build_command(stage, config_path, args))
        print(f"  {stage}: {status}")
        print(f"  command: {command}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or inspect a bounded method sweep.")
    parser.add_argument("--sweep-config", default=None, help="Path to the sweep JSON.")
    parser.add_argument(
        "--stage",
        choices=("plan", "train", "evaluate", "all"),
        default="plan",
        help="Which stage to run. Use plan to inspect candidate status and commands.",
    )
    parser.add_argument("--candidate", action="append", default=None, help="Optional candidate id filter.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip stages with completed artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Optional upstream checkout override.")
    parser.add_argument("--train-num-envs", type=int, default=None, help="Override training num_envs.")
    parser.add_argument("--eval-num-envs", type=int, default=None, help="Override evaluation num_envs.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override training max iterations.")
    parser.add_argument("--episodes", type=int, default=None, help="Override evaluation episode count.")
    parser.add_argument("--checkpoint", type=int, default=None, help="Specific checkpoint for evaluation.")
    parser.add_argument("--rl-device", default=None, help="Optional RL device override.")
    parser.add_argument("--sim-device", default=None, help="Optional sim device override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional shared seed override.")
    args = parser.parse_args()

    sweep_cfg = load_sweep_config(args.sweep_config)
    candidates = selected_candidates(sweep_cfg, args.candidate)
    if args.stage == "plan":
        for candidate in candidates:
            config = load_config(resolve_config_path(candidate["config"]))
            print_plan(candidate, config, args)
        return 0

    stages = ["train", "evaluate"] if args.stage == "all" else [args.stage]
    for candidate in candidates:
        config_path = resolve_config_path(candidate["config"])
        config = load_config(config_path)
        print(f"[{candidate['id']}] {candidate['label']}")
        trained_this_round = False
        for stage in stages:
            is_complete = stage_complete(config, stage)
            if args.skip_completed and is_complete and not (stage == "evaluate" and trained_this_round):
                marker = stage_marker_path(config, stage)
                print(f"Skipping {stage}: {relative_to_repo(marker)} already exists")
                continue
            command = build_command(stage, config_path, args)
            exit_code = run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run)
            if exit_code != 0:
                return exit_code
            if stage == "train" and not args.dry_run:
                trained_this_round = True
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
