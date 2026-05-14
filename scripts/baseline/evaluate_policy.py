#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    artifact_dir,
    default_manifest,
    ensure_humanoid_gym_checkout,
    ensure_upstream_on_syspath,
    experiment_root,
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
            "evaluate_policy.py",
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
    parser = argparse.ArgumentParser(description="Evaluate the issue #1 Vanilla PPO baseline on rough terrain.")
    parser.add_argument("--config", default=None, help="Path to the baseline config JSON.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument("--run-name", default=None, help="Override the configured run_name.")
    parser.add_argument("--load-run", default=None, help="Explicit upstream run directory name.")
    parser.add_argument("--checkpoint", type=int, default=None, help="Specific checkpoint number to evaluate.")
    parser.add_argument("--episodes", type=int, default=None, help="Override the number of completed episodes.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override the evaluation environment count.")
    parser.add_argument("--rl-device", default=None, help="Override the configured RL device.")
    parser.add_argument("--sim-device", default=None, help="Override the configured sim device.")
    parser.add_argument("--seed", type=int, default=None, help="Override the evaluation seed.")
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
    from humanoid.utils import get_args, task_registry

    eval_cfg = config["evaluation"]
    rl_device = args.rl_device or eval_cfg["rl_device"]
    sim_device = args.sim_device or eval_cfg["sim_device"]
    seed = args.seed if args.seed is not None else eval_cfg.get("seed")

    upstream_args = build_args(get_args, config, rl_device, sim_device, seed)
    upstream_args.num_envs = args.num_envs or eval_cfg["num_envs"]

    env_cfg, train_cfg = task_registry.get_cfgs(name=config["task"])
    env_cfg.env.num_envs = upstream_args.num_envs
    env_cfg.terrain.curriculum = False
    env, env_cfg = task_registry.make_env(name=config["task"], args=upstream_args, env_cfg=env_cfg)

    run_dir = resolve_run_dir(humanoid_gym_root, config, run_name=run_name, load_run=args.load_run)
    train_cfg.runner.resume = True
    train_cfg.runner.experiment_name = config["experiment_name"]
    train_cfg.runner.run_name = run_name
    train_cfg.runner.load_run = run_dir.name
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=config["task"],
        args=upstream_args,
        train_cfg=train_cfg,
        log_root=str(experiment_root(humanoid_gym_root, config["experiment_name"])),
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    target_episodes = args.episodes or eval_cfg["episodes"]
    completed_episodes = 0
    episode_returns = torch.zeros(env.num_envs, device=env.device)
    episode_tracking_error = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)

    returns: list[float] = []
    tracking_errors: list[float] = []
    fell_flags: list[bool] = []

    while completed_episodes < target_episodes:
        actions = policy(obs.detach())
        obs, _, rewards, dones, infos = env.step(actions.detach())

        linear_error = torch.abs(env.commands[:, 0] - env.base_lin_vel[:, 0])
        yaw_error = torch.abs(env.commands[:, 2] - env.base_ang_vel[:, 2])
        step_tracking_error = linear_error + yaw_error

        episode_returns += rewards
        episode_tracking_error += step_tracking_error
        episode_lengths += 1

        done_ids = torch.nonzero(dones > 0, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        time_outs = infos.get("time_outs")
        for env_id in done_ids.tolist():
            returns.append(float(episode_returns[env_id].item()))
            tracking_errors.append(
                float((episode_tracking_error[env_id] / torch.clamp(episode_lengths[env_id], min=1)).item())
            )
            if time_outs is None:
                fell_flags.append(True)
            else:
                fell_flags.append(not bool(time_outs[env_id].item()))
            completed_episodes += 1
            episode_returns[env_id] = 0.0
            episode_tracking_error[env_id] = 0.0
            episode_lengths[env_id] = 0.0
            if completed_episodes >= target_episodes:
                break

    fall_count = sum(1 for flag in fell_flags if flag)
    metrics = {
        "episodes_evaluated": completed_episodes,
        "velocity_tracking_error_mean": statistics.fmean(tracking_errors),
        "velocity_tracking_error_std": statistics.pstdev(tracking_errors) if len(tracking_errors) > 1 else 0.0,
        "fall_rate": fall_count / completed_episodes,
        "episode_return_mean": statistics.fmean(returns),
        "episode_return_std": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
    }

    checkpoint_path = latest_checkpoint(run_dir) if args.checkpoint is None else run_dir / f"model_{args.checkpoint}.pt"
    output_dir = artifact_dir(config, run_name)
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)

    manifest_path = output_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else default_manifest(config, humanoid_gym_root)
    manifest["run_name"] = run_name
    manifest["run_dir"] = relative_to_repo(run_dir)
    manifest["checkpoint_path"] = relative_to_repo(checkpoint_path)
    manifest["metrics_path"] = relative_to_repo(metrics_path)
    manifest["metrics"] = metrics
    write_json(manifest_path, manifest)

    print(f"Wrote {relative_to_repo(metrics_path)}")
    print(f"Updated {relative_to_repo(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
