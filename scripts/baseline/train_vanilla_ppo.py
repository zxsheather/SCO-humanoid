#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    artifact_dir,
    build_upstream_command,
    default_manifest,
    ensure_directory,
    ensure_humanoid_gym_checkout,
    latest_checkpoint,
    load_config,
    relative_to_repo,
    resolve_humanoid_gym_root,
    resolve_run_dir,
    run_command,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the issue #1 Vanilla PPO baseline training command.")
    parser.add_argument("--config", default=None, help="Path to the baseline config JSON.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument("--run-name", default=None, help="Override the configured run_name.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override the configured num_envs.")
    parser.add_argument("--rl-device", default=None, help="Override the configured RL device.")
    parser.add_argument("--sim-device", default=None, help="Override the configured sim device.")
    parser.add_argument("--seed", type=int, default=None, help="Optional training seed.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Optional max learning iterations.")
    parser.add_argument("--dry-run", action="store_true", help="Print the upstream command without executing it.")
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    if not args.dry_run:
        ensure_humanoid_gym_checkout(humanoid_gym_root)

    training = config["training"]
    run_name = args.run_name or config["run_name"]
    command = build_upstream_command(
        humanoid_gym_root,
        "train.py",
        task=config["task"],
        experiment_name=config["experiment_name"],
        run_name=run_name,
        rl_device=args.rl_device or training["rl_device"],
        sim_device=args.sim_device or training["sim_device"],
        headless=bool(training["headless"]),
        num_envs=args.num_envs or training["num_envs"],
        seed=args.seed,
        max_iterations=args.max_iterations,
    )

    exit_code = run_command(command, cwd=humanoid_gym_root / "humanoid", dry_run=args.dry_run)
    if exit_code != 0 or args.dry_run:
        return exit_code

    run_dir = resolve_run_dir(humanoid_gym_root, config, run_name=run_name)
    checkpoint_path = latest_checkpoint(run_dir)
    manifest = default_manifest(config, humanoid_gym_root)
    manifest["run_name"] = run_name
    manifest["training_command"] = command
    manifest["run_dir"] = relative_to_repo(run_dir)
    manifest["checkpoint_path"] = relative_to_repo(checkpoint_path)

    output_dir = ensure_directory(artifact_dir(config, run_name))
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote manifest to {relative_to_repo(output_dir / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
