#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    artifact_dir,
    configure_runtime_env,
    default_manifest,
    ensure_directory,
    ensure_humanoid_gym_checkout,
    ensure_upstream_on_syspath,
    experiment_root,
    latest_checkpoint,
    load_config,
    relative_to_repo,
    resolve_humanoid_gym_root,
    resolve_run_dir,
    write_json,
)
from _overrides import apply_method_overrides  # noqa: E402


def build_args(
    get_args,
    config: dict,
    run_name: str,
    rl_device: str,
    sim_device: str,
    seed: int | None,
) -> object:
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "train_vanilla_ppo.py",
            f"--task={config['task']}",
            f"--experiment_name={config['experiment_name']}",
            f"--run_name={run_name}",
            f"--rl_device={rl_device}",
            f"--sim_device={sim_device}",
            "--headless",
        ]
        if seed is not None:
            sys.argv.append(f"--seed={seed}")
        return get_args()
    finally:
        sys.argv = original_argv


def rendered_invocation(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    if args.config:
        command.append(f"--config={args.config}")
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    if args.run_name:
        command.append(f"--run-name={args.run_name}")
    if args.num_envs is not None:
        command.append(f"--num-envs={args.num_envs}")
    if args.rl_device:
        command.append(f"--rl-device={args.rl_device}")
    if args.sim_device:
        command.append(f"--sim-device={args.sim_device}")
    if args.seed is not None:
        command.append(f"--seed={args.seed}")
    if args.max_iterations is not None:
        command.append(f"--max-iterations={args.max_iterations}")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one smooth-control method training command.")
    parser.add_argument("--config", default=None, help="Path to the method config JSON.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument("--run-name", default=None, help="Override the configured run_name.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override the configured num_envs.")
    parser.add_argument("--rl-device", default=None, help="Override the configured RL device.")
    parser.add_argument("--sim-device", default=None, help="Override the configured sim device.")
    parser.add_argument("--seed", type=int, default=None, help="Optional training seed.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Optional max learning iterations.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved wrapper command without executing it.")
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    training = config["training"]
    run_name = args.run_name or config["run_name"]
    command = rendered_invocation(args)
    if args.dry_run:
        print(" ".join(command))
        return 0

    ensure_humanoid_gym_checkout(humanoid_gym_root)
    configure_runtime_env()
    ensure_upstream_on_syspath(humanoid_gym_root)

    importlib.import_module("humanoid.envs")
    from humanoid.utils import get_args, task_registry

    rl_device = args.rl_device or training["rl_device"]
    sim_device = args.sim_device or training["sim_device"]
    seed = args.seed

    upstream_args = build_args(get_args, config, run_name, rl_device, sim_device, seed)
    upstream_args.num_envs = args.num_envs or training["num_envs"]

    env_cfg, train_cfg = task_registry.get_cfgs(name=config["task"])
    env_cfg, train_cfg = apply_method_overrides(env_cfg, train_cfg, config)
    env_cfg.env.num_envs = upstream_args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    train_cfg.runner.experiment_name = config["experiment_name"]
    train_cfg.runner.run_name = run_name

    env, env_cfg = task_registry.make_env(name=config["task"], args=upstream_args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=config["task"],
        args=upstream_args,
        train_cfg=train_cfg,
        log_root=str(experiment_root(humanoid_gym_root, config["experiment_name"])),
    )
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    run_dir = resolve_run_dir(humanoid_gym_root, config, run_name=run_name)
    checkpoint_path = latest_checkpoint(run_dir)
    manifest = default_manifest(config, humanoid_gym_root)
    manifest["run_name"] = run_name
    manifest["training_command"] = command
    manifest["overrides"] = config.get("overrides", {})
    manifest["run_dir"] = relative_to_repo(run_dir)
    manifest["checkpoint_path"] = relative_to_repo(checkpoint_path)

    output_dir = ensure_directory(artifact_dir(config, run_name))
    artifact_payload = ppo_runner.alg.get_artifact_payload()
    constraint_logging = config.get("evaluation", {}).get("constraint_logging", {})
    sidecar_filename = constraint_logging.get("sidecar_metrics_filename", "constraint_metrics.json")
    multiplier_trace_filename = constraint_logging.get("multiplier_trace_filename", "lagrange_multiplier_trace.json")
    constraint_metrics = artifact_payload.get("constraint_metrics")
    if isinstance(constraint_metrics, dict):
        write_json(output_dir / sidecar_filename, constraint_metrics)
        manifest["constraint_metrics_path"] = relative_to_repo(output_dir / sidecar_filename)
    lagrange_multiplier_trace = artifact_payload.get("lagrange_multiplier_trace")
    if isinstance(lagrange_multiplier_trace, list) and lagrange_multiplier_trace:
        write_json(output_dir / multiplier_trace_filename, {"trace": lagrange_multiplier_trace})
        manifest["lagrange_multiplier_trace_path"] = relative_to_repo(output_dir / multiplier_trace_filename)
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote manifest to {relative_to_repo(output_dir / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
