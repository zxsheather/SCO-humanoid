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
    default_manifest,
    ensure_directory,
    ensure_humanoid_gym_checkout,
    ensure_upstream_on_syspath,
    latest_checkpoint,
    load_config,
    read_json,
    relative_to_repo,
    resolve_humanoid_gym_root,
    resolve_run_dir,
    write_json,
)


def build_args(get_args, config: dict, rl_device: str, sim_device: str, seed: int | None) -> object:
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "export_policy.py",
            f"--task={config['task']}",
            f"--experiment_name={config['experiment_name']}",
            f"--run_name={config['run_name']}",
            f"--rl_device={rl_device}",
            f"--sim_device={sim_device}",
            "--headless",
            "--resume",
        ]
        if seed is not None:
            sys.argv.append(f"--seed={seed}")
        return get_args()
    finally:
        sys.argv = original_argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a JIT policy for the issue #1 Vanilla PPO baseline.")
    parser.add_argument("--config", default=None, help="Path to the baseline config JSON.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument("--run-name", default=None, help="Override the configured run_name.")
    parser.add_argument("--load-run", default=None, help="Explicit upstream run directory name.")
    parser.add_argument("--checkpoint", type=int, default=None, help="Specific checkpoint number to export.")
    parser.add_argument("--rl-device", default=None, help="Override the configured RL device.")
    parser.add_argument("--sim-device", default=None, help="Override the configured sim device.")
    parser.add_argument("--seed", type=int, default=None, help="Optional evaluation seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = args.run_name or config["run_name"]
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    ensure_humanoid_gym_checkout(humanoid_gym_root)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    os.environ.setdefault("WANDB_MODE", "disabled")
    Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    ensure_upstream_on_syspath(humanoid_gym_root)

    importlib.import_module("humanoid.envs")
    import torch
    from humanoid.utils import export_policy_as_jit, get_args, task_registry

    eval_cfg = config["evaluation"]
    rl_device = args.rl_device or eval_cfg["rl_device"]
    sim_device = args.sim_device or eval_cfg["sim_device"]

    upstream_args = build_args(get_args, config, rl_device, sim_device, args.seed)
    upstream_args.num_envs = 1

    env, env_cfg = task_registry.make_env(name=config["task"], args=upstream_args)
    train_cfg = task_registry.get_cfgs(name=config["task"])[1]
    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.experiment_name = config["experiment_name"]
    train_cfg.runner.run_name = run_name
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=config["task"], args=upstream_args, train_cfg=train_cfg
    )

    output_dir = ensure_directory(artifact_dir(config, run_name) / "exported" / "policies")
    export_policy_as_jit(ppo_runner.alg.actor_critic, str(output_dir))
    exported_policy = output_dir / "policy_1.pt"
    reloaded = torch.jit.load(str(exported_policy))
    del reloaded

    run_dir = resolve_run_dir(humanoid_gym_root, config, run_name=run_name, load_run=args.load_run)
    checkpoint_path = latest_checkpoint(run_dir) if args.checkpoint is None else run_dir / f"model_{args.checkpoint}.pt"

    manifest_path = artifact_dir(config, run_name) / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else default_manifest(config, humanoid_gym_root)
    manifest["run_name"] = run_name
    manifest["run_dir"] = relative_to_repo(run_dir)
    manifest["checkpoint_path"] = relative_to_repo(checkpoint_path)
    manifest["exported_policy_path"] = relative_to_repo(exported_policy)
    manifest["policy_reload_check"] = True
    manifest["num_obs"] = int(env.num_obs)
    manifest["num_actions"] = int(env.num_actions)
    write_json(manifest_path, manifest)

    print(f"Exported {relative_to_repo(exported_policy)}")
    print(f"Updated {relative_to_repo(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
