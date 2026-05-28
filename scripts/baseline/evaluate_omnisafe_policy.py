#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import statistics
import sys
from pathlib import Path
from typing import Any

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
    load_config,
    relative_to_repo,
    resolve_humanoid_gym_root,
    write_json,
)
from _overrides import apply_method_overrides  # noqa: E402
from evaluate_policy import compute_policy_local_sensitivity, constraint_logging_config  # noqa: E402


DEFAULT_CONFIG = "configs/methods/omnisafe_ppolag_eval_smoke.json"


def build_args(get_args, config: dict, run_name: str, rl_device: str, sim_device: str, seed: int | None) -> object:
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "evaluate_omnisafe_policy.py",
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


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def resolve_checkpoint_path(config: dict[str, Any], checkpoint: int, raw_path: str | None) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    checkpoint_dir = Path(config.get("omnisafe", {}).get("checkpoint_dir", ""))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path.cwd() / checkpoint_dir
    return checkpoint_dir / f"model_{int(checkpoint)}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an OmniSafe-style policy with shared rough-terrain metrics.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--humanoid-gym-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--create-fixture-checkpoint", action="store_true")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rl-device", default=None)
    parser.add_argument("--sim-device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_name = args.run_name or config["run_name"]
    output_dir = ensure_directory(artifact_dir(config, run_name))
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    ensure_humanoid_gym_checkout(humanoid_gym_root)
    configure_runtime_env()
    ensure_upstream_on_syspath(humanoid_gym_root)

    import isaacgym  # noqa: F401
    import torch

    from _omnisafe_policy_loader import (
        create_gaussian_actor,
        load_omnisafe_policy_checkpoint,
        save_omnisafe_policy_checkpoint,
    )

    importlib.import_module("humanoid.envs")
    from humanoid.utils import get_args, task_registry

    eval_cfg = config["evaluation"]
    method_cfg = config.get("method", {})
    constraint_cfg = constraint_logging_config(config)
    rl_device = args.rl_device or eval_cfg["rl_device"]
    sim_device = args.sim_device or eval_cfg["sim_device"]
    seed = args.seed if args.seed is not None else eval_cfg.get("seed")
    target_episodes = args.episodes or eval_cfg["episodes"]

    upstream_args = build_args(get_args, config, run_name, rl_device, sim_device, seed)
    upstream_args.num_envs = args.num_envs or eval_cfg["num_envs"]

    env_cfg, train_cfg = task_registry.get_cfgs(name=config["task"])
    env_cfg, train_cfg = apply_method_overrides(env_cfg, train_cfg, config)
    env_cfg.env.num_envs = upstream_args.num_envs
    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.curriculum = False
    if seed is not None:
        env_cfg.seed = seed
    env, _ = task_registry.make_env(name=config["task"], args=upstream_args, env_cfg=env_cfg)

    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint, args.checkpoint_path)
    omnisafe_cfg = config.get("omnisafe", {})
    if args.create_fixture_checkpoint and not checkpoint_path.exists():
        actor = create_gaussian_actor(
            obs_dim=int(env.num_obs),
            act_dim=int(env.num_actions),
            hidden_sizes=[int(v) for v in omnisafe_cfg.get("actor_hidden_sizes", [64, 64])],
            activation=str(omnisafe_cfg.get("actor_activation", "tanh")),
            device=env.device,
        )
        save_omnisafe_policy_checkpoint(
            checkpoint_path,
            actor,
            checkpoint=int(args.checkpoint),
            seed=int(seed or 0),
            cost_config=omnisafe_cfg.get("cost_config", {}),
            metadata={"fixture": True, "source": "#62 evaluation compatibility smoke"},
        )

    policy, policy_metadata = load_omnisafe_policy_checkpoint(checkpoint_path, device=env.device)

    obs = env.get_observations()
    completed_episodes = 0
    episode_returns = torch.zeros(env.num_envs, device=env.device)
    episode_tracking_error = torch.zeros(env.num_envs, device=env.device)
    episode_joint_accel = torch.zeros(env.num_envs, device=env.device)
    episode_action_jitter = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)

    returns: list[float] = []
    tracking_errors: list[float] = []
    joint_accel_values: list[float] = []
    action_jitter_values: list[float] = []
    fell_flags: list[bool] = []
    local_sensitivity_samples: list[float] = []
    step_counter = 0

    while completed_episodes < target_episodes:
        if constraint_cfg["collect_local_sensitivity"] and step_counter % max(
            int(constraint_cfg["sample_every_n_steps"]), 1
        ) == 0:
            sensitivity = compute_policy_local_sensitivity(
                policy, obs, sample_envs=int(constraint_cfg["sample_envs"])
            )
            if sensitivity is not None:
                local_sensitivity_samples.extend(float(value) for value in sensitivity.detach().cpu().tolist())

        prev_actions = env.actions.detach().clone()
        prev_dof_vel = env.dof_vel.detach().clone()
        with torch.inference_mode():
            actions = policy(obs.detach())
            obs, _, rewards, dones, infos = env.step(actions.detach())

        linear_error = torch.abs(env.commands[:, 0] - env.base_lin_vel[:, 0])
        yaw_error = torch.abs(env.commands[:, 2] - env.base_ang_vel[:, 2])
        step_tracking_error = linear_error + yaw_error
        step_joint_accel = torch.linalg.vector_norm((env.dof_vel - prev_dof_vel) / env.dt, dim=1)
        step_action_jitter = torch.linalg.vector_norm(env.actions - prev_actions, dim=1)

        episode_returns += rewards
        episode_tracking_error += step_tracking_error
        episode_joint_accel += step_joint_accel
        episode_action_jitter += step_action_jitter
        episode_lengths += 1
        step_counter += 1

        done_ids = torch.nonzero(dones > 0, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        time_outs = infos.get("time_outs")
        for env_id in done_ids.tolist():
            fell = True if time_outs is None else not bool(time_outs[env_id].item())
            returns.append(float(episode_returns[env_id].item()))
            tracking_errors.append(
                float((episode_tracking_error[env_id] / torch.clamp(episode_lengths[env_id], min=1)).item())
            )
            joint_accel_values.append(
                float((episode_joint_accel[env_id] / torch.clamp(episode_lengths[env_id], min=1)).item())
            )
            action_jitter_values.append(
                float((episode_action_jitter[env_id] / torch.clamp(episode_lengths[env_id], min=1)).item())
            )
            fell_flags.append(fell)
            completed_episodes += 1
            episode_returns[env_id] = 0.0
            episode_tracking_error[env_id] = 0.0
            episode_joint_accel[env_id] = 0.0
            episode_action_jitter[env_id] = 0.0
            episode_lengths[env_id] = 0.0
            if completed_episodes >= target_episodes:
                break

    tracking_mean, tracking_std = summarize(tracking_errors)
    return_mean, return_std = summarize(returns)
    joint_accel_mean, joint_accel_std = summarize(joint_accel_values)
    action_jitter_mean, action_jitter_std = summarize(action_jitter_values)
    fall_count = sum(1 for flag in fell_flags if flag)

    local_sensitivity_mean, local_sensitivity_std = summarize(local_sensitivity_samples)
    threshold = constraint_cfg.get("local_sensitivity_threshold")
    violation_rate = None
    if threshold is not None and local_sensitivity_samples:
        violation_rate = sum(1 for value in local_sensitivity_samples if value > float(threshold)) / len(
            local_sensitivity_samples
        )

    constraint_metrics = {
        "supported": True,
        "policy_local_sensitivity_cost_mean": local_sensitivity_mean,
        "policy_local_sensitivity_cost_std": local_sensitivity_std,
        "policy_local_sensitivity_sample_count": len(local_sensitivity_samples),
        "constraint_violation_rate": violation_rate,
        "local_sensitivity_threshold": threshold,
        "cost_config": policy_metadata.get("cost_config", {}),
    }
    metrics = {
        "metric_schema_version": int(eval_cfg.get("metric_schema_version", 2)),
        "method_id": method_cfg.get("id"),
        "method_label": method_cfg.get("label"),
        "evaluation_protocol": config.get("evaluation_protocol"),
        "episodes_evaluated": completed_episodes,
        "velocity_tracking_error_mean": tracking_mean,
        "velocity_tracking_error_std": tracking_std,
        "fall_rate": fall_count / completed_episodes,
        "joint_acceleration_l2_mean": joint_accel_mean,
        "joint_acceleration_l2_std": joint_accel_std,
        "action_jitter_l2_mean": action_jitter_mean,
        "action_jitter_l2_std": action_jitter_std,
        "episode_return_mean": return_mean,
        "episode_return_std": return_std,
        "constraint_metrics": constraint_metrics,
        "trace_capture": {"enabled": False, "episodes_captured": 0, "trace_path": None},
        "omnisafe_policy": policy_metadata,
    }

    metrics_path = output_dir / f"metrics_checkpoint_{int(args.checkpoint)}.json"
    write_json(metrics_path, metrics)
    write_json(output_dir / "metrics.json", metrics)

    manifest_path = output_dir / "manifest.json"
    manifest = default_manifest(config, humanoid_gym_root)
    manifest["run_name"] = run_name
    manifest["checkpoint"] = int(args.checkpoint)
    manifest["checkpoint_path"] = relative_to_repo(checkpoint_path)
    manifest["metrics_path"] = relative_to_repo(metrics_path)
    manifest["metrics"] = metrics
    manifest["omnisafe_policy"] = policy_metadata
    write_json(manifest_path, manifest)

    status_path = output_dir / "omnisafe_evaluation.json"
    write_json(
        status_path,
        {
            "status": "complete",
            "run_name": run_name,
            "checkpoint": int(args.checkpoint),
            "checkpoint_path": relative_to_repo(checkpoint_path),
            "metrics_path": relative_to_repo(metrics_path),
            "episodes_evaluated": completed_episodes,
        },
    )

    print(f"Wrote {relative_to_repo(metrics_path)}")
    print(f"Updated {relative_to_repo(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
