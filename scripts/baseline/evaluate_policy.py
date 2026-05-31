#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
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
from _behavior_trace_metrics import (  # noqa: E402
    should_capture_traces,
    trace_capture_config,
)
from _overrides import apply_method_overrides  # noqa: E402


def build_args(get_args, config: dict, run_name: str, rl_device: str, sim_device: str, seed: int | None) -> object:
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "evaluate_policy.py",
            f"--task={config['task']}",
            f"--experiment_name={config['experiment_name']}",
            f"--run_name={run_name}",
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


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def constraint_logging_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "collect_local_sensitivity": False,
        "local_sensitivity_threshold": None,
        "sample_envs": 4,
        "sample_every_n_steps": 20,
        "sidecar_metrics_filename": "constraint_metrics.json",
        "multiplier_trace_filename": "lagrange_multiplier_trace.json",
    }
    merged = defaults.copy()
    merged.update(config.get("evaluation", {}).get("constraint_logging", {}))
    return merged


def reset_trace_buffers(trace_buffers: list[dict[str, list[list[float]]]], env_id: int) -> None:
    trace_buffers[env_id]["dof_pos"].clear()
    trace_buffers[env_id]["dof_vel"].clear()


def append_trace_step(
    trace_buffers: list[dict[str, list[list[float]]]],
    env,
    max_steps_per_episode: int,
) -> None:
    for env_id, buffer in enumerate(trace_buffers):
        if len(buffer["dof_pos"]) >= max_steps_per_episode:
            continue
        buffer["dof_pos"].append(env.dof_pos[env_id].detach().cpu().tolist())
        buffer["dof_vel"].append(env.dof_vel[env_id].detach().cpu().tolist())


def resolve_optional_artifact_path(output_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    return candidate


def compute_policy_local_sensitivity(actor_critic, obs, sample_envs: int):
    import torch

    sample_count = min(sample_envs, obs.shape[0])
    if sample_count <= 0:
        return None

    with torch.enable_grad():
        obs_sample = obs[:sample_count].detach().clone().requires_grad_(True)
        action_mean = actor_critic.act_inference(obs_sample)
        squared_norm = torch.zeros(sample_count, device=obs_sample.device)
        for action_idx in range(action_mean.shape[1]):
            grad_outputs = torch.zeros_like(action_mean)
            grad_outputs[:, action_idx] = 1.0
            grads = torch.autograd.grad(
                outputs=action_mean,
                inputs=obs_sample,
                grad_outputs=grad_outputs,
                retain_graph=action_idx + 1 < action_mean.shape[1],
                create_graph=False,
                allow_unused=False,
            )[0]
            squared_norm += torch.sum(torch.square(grads), dim=1)
    return torch.sqrt(squared_norm)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one smooth-control method config on rough terrain.")
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
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise standard deviation added to policy observations at inference time.",
    )
    parser.add_argument(
        "--obs-noise-seed",
        type=int,
        default=20260531,
        help="Random seed for --obs-noise-std perturbations.",
    )
    parser.add_argument("--capture-traces", action="store_true", help="Persist compact episode traces for offline analysis.")
    parser.add_argument("--trace-max-episodes", type=int, default=None, help="Override the number of completed episodes to trace.")
    parser.add_argument(
        "--trace-max-steps",
        type=int,
        default=None,
        help="Override the maximum number of recorded steps per captured episode.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = args.run_name or config["run_name"]
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    ensure_humanoid_gym_checkout(humanoid_gym_root)
    configure_runtime_env()
    ensure_upstream_on_syspath(humanoid_gym_root)

    importlib.import_module("humanoid.envs")
    import torch
    from humanoid.utils import get_args, task_registry

    eval_cfg = config["evaluation"]
    method_cfg = config.get("method", {})
    constraint_cfg = constraint_logging_config(config)
    trace_cfg = trace_capture_config(config)
    if args.capture_traces:
        trace_cfg["enabled"] = True
    if args.trace_max_episodes is not None:
        trace_cfg["max_episodes"] = int(args.trace_max_episodes)
    if args.trace_max_steps is not None:
        trace_cfg["max_steps_per_episode"] = int(args.trace_max_steps)
    capture_traces = should_capture_traces(trace_cfg)
    rl_device = args.rl_device or eval_cfg["rl_device"]
    sim_device = args.sim_device or eval_cfg["sim_device"]
    seed = args.seed if args.seed is not None else eval_cfg.get("seed")

    upstream_args = build_args(get_args, config, run_name, rl_device, sim_device, seed)
    upstream_args.num_envs = args.num_envs or eval_cfg["num_envs"]

    env_cfg, train_cfg = task_registry.get_cfgs(name=config["task"])
    env_cfg, train_cfg = apply_method_overrides(env_cfg, train_cfg, config)
    env_cfg.env.num_envs = upstream_args.num_envs
    env_cfg.terrain.curriculum = False
    obs_clip = float(getattr(env_cfg.normalization, "clip_observations", 100.0))
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
    actor_critic = ppo_runner.alg.actor_critic

    obs = env.get_observations()
    obs_noise_std = max(float(args.obs_noise_std), 0.0)
    obs_noise_generator = torch.Generator(device=env.device)
    obs_noise_generator.manual_seed(int(args.obs_noise_seed))
    target_episodes = args.episodes or eval_cfg["episodes"]
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
    captured_episodes: list[dict[str, Any]] = []
    trace_buffers = [{"dof_pos": [], "dof_vel": []} for _ in range(env.num_envs)]
    step_counter = 0

    while completed_episodes < target_episodes:
        if constraint_cfg["collect_local_sensitivity"] and step_counter % max(
            int(constraint_cfg["sample_every_n_steps"]), 1
        ) == 0:
            sensitivity = compute_policy_local_sensitivity(
                actor_critic, obs, sample_envs=int(constraint_cfg["sample_envs"])
            )
            if sensitivity is not None:
                local_sensitivity_samples.extend(float(value) for value in sensitivity.detach().cpu().tolist())

        prev_actions = env.actions.detach().clone()
        prev_dof_vel = env.dof_vel.detach().clone()
        with torch.inference_mode():
            policy_obs = obs.detach()
            if obs_noise_std > 0.0:
                policy_obs = torch.clamp(
                    policy_obs
                    + torch.randn(
                        policy_obs.shape,
                        generator=obs_noise_generator,
                        device=policy_obs.device,
                        dtype=policy_obs.dtype,
                    )
                    * obs_noise_std,
                    -obs_clip,
                    obs_clip,
                )
            actions = policy(policy_obs)
            obs, _, rewards, dones, infos = env.step(actions.detach())

        if capture_traces and len(captured_episodes) < int(trace_cfg["max_episodes"]):
            append_trace_step(trace_buffers, env, int(trace_cfg["max_steps_per_episode"]))

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
            if capture_traces and len(captured_episodes) < int(trace_cfg["max_episodes"]):
                captured_episodes.append(
                    {
                        "episode_index": len(captured_episodes),
                        "env_id": env_id,
                        "episode_length": int(episode_lengths[env_id].item()),
                        "fell": fell,
                        "dt": float(env.dt),
                        "dof_pos": [list(step) for step in trace_buffers[env_id]["dof_pos"]],
                        "dof_vel": [list(step) for step in trace_buffers[env_id]["dof_vel"]],
                        "truncated": len(trace_buffers[env_id]["dof_pos"]) >= int(trace_cfg["max_steps_per_episode"]),
                    }
                )
            reset_trace_buffers(trace_buffers, env_id)
            completed_episodes += 1
            episode_returns[env_id] = 0.0
            episode_tracking_error[env_id] = 0.0
            episode_joint_accel[env_id] = 0.0
            episode_action_jitter[env_id] = 0.0
            episode_lengths[env_id] = 0.0
            if completed_episodes >= target_episodes:
                break

    fall_count = sum(1 for flag in fell_flags if flag)
    tracking_mean, tracking_std = summarize(tracking_errors)
    return_mean, return_std = summarize(returns)
    joint_accel_mean, joint_accel_std = summarize(joint_accel_values)
    action_jitter_mean, action_jitter_std = summarize(action_jitter_values)

    output_dir = artifact_dir(config, run_name)
    sidecar_path = resolve_optional_artifact_path(output_dir, constraint_cfg["sidecar_metrics_filename"])
    multiplier_trace_path = resolve_optional_artifact_path(output_dir, constraint_cfg["multiplier_trace_filename"])
    sidecar_metrics = None
    if sidecar_path is not None and sidecar_path.exists():
        sidecar_metrics = read_json(sidecar_path)

    local_sensitivity_mean, local_sensitivity_std = summarize(local_sensitivity_samples)
    local_sensitivity_threshold = constraint_cfg.get("local_sensitivity_threshold")
    if local_sensitivity_threshold is None and isinstance(sidecar_metrics, dict):
        local_sensitivity_threshold = sidecar_metrics.get("local_sensitivity_threshold")

    violation_rate = None
    if local_sensitivity_threshold is not None and local_sensitivity_samples:
        violation_count = sum(1 for value in local_sensitivity_samples if value > float(local_sensitivity_threshold))
        violation_rate = violation_count / len(local_sensitivity_samples)

    constraint_metrics: dict[str, Any] = {
        "supported": bool(constraint_cfg["collect_local_sensitivity"] or sidecar_metrics or multiplier_trace_path),
        "policy_local_sensitivity_cost_mean": local_sensitivity_mean,
        "policy_local_sensitivity_cost_std": local_sensitivity_std,
        "policy_local_sensitivity_sample_count": len(local_sensitivity_samples),
        "constraint_violation_rate": violation_rate,
        "local_sensitivity_threshold": local_sensitivity_threshold,
        "sidecar_metrics_path": relative_to_repo(sidecar_path) if sidecar_path and sidecar_path.exists() else None,
        "lagrange_multiplier_trace_path": (
            relative_to_repo(multiplier_trace_path) if multiplier_trace_path and multiplier_trace_path.exists() else None
        ),
    }
    if isinstance(sidecar_metrics, dict):
        constraint_metrics["sidecar_metrics"] = sidecar_metrics
        if constraint_metrics["policy_local_sensitivity_cost_mean"] is None:
            constraint_metrics["policy_local_sensitivity_cost_mean"] = sidecar_metrics.get(
                "policy_local_sensitivity_cost_mean"
            )
        if constraint_metrics["policy_local_sensitivity_cost_std"] is None:
            constraint_metrics["policy_local_sensitivity_cost_std"] = sidecar_metrics.get(
                "policy_local_sensitivity_cost_std"
            )
        if constraint_metrics["constraint_violation_rate"] is None:
            constraint_metrics["constraint_violation_rate"] = sidecar_metrics.get("constraint_violation_rate")

    metrics = {
        "metric_schema_version": int(eval_cfg.get("metric_schema_version", 1)),
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
        "observation_noise": {
            "std": obs_noise_std,
            "seed": int(args.obs_noise_seed),
            "applied_to": "policy_observation_input",
            "clip_observations": obs_clip,
        },
        "trace_capture": {
            "enabled": capture_traces,
            "episodes_captured": len(captured_episodes),
            "trace_path": None,
        },
    }

    checkpoint_path = latest_checkpoint(run_dir) if args.checkpoint is None else run_dir / f"model_{args.checkpoint}.pt"
    trace_path = None
    if capture_traces:
        trace_path = output_dir / str(trace_cfg["filename"])
        write_json(
            trace_path,
            {
                "metric_schema_version": int(eval_cfg.get("metric_schema_version", 1)),
                "run_name": run_name,
                "checkpoint_path": relative_to_repo(checkpoint_path),
                "episodes": captured_episodes,
            },
        )
        metrics["trace_capture"]["trace_path"] = relative_to_repo(trace_path)
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)

    manifest_path = output_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else default_manifest(config, humanoid_gym_root)
    if method_cfg:
        manifest["method"] = method_cfg
    if config.get("evaluation_protocol"):
        manifest["evaluation_protocol"] = config["evaluation_protocol"]
    manifest["run_name"] = run_name
    manifest["run_dir"] = relative_to_repo(run_dir)
    manifest["checkpoint_path"] = relative_to_repo(checkpoint_path)
    manifest["metrics_path"] = relative_to_repo(metrics_path)
    if trace_path is not None and trace_path.exists():
        manifest["trace_path"] = relative_to_repo(trace_path)
    manifest["metrics"] = metrics
    write_json(manifest_path, manifest)

    print(f"Wrote {relative_to_repo(metrics_path)}")
    if trace_path is not None and trace_path.exists():
        print(f"Wrote {relative_to_repo(trace_path)}")
    print(f"Updated {relative_to_repo(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
