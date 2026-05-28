#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import math
import random
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
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


DEFAULT_CONFIG = "configs/methods/omnisafe_ppolag_diagnostic.json"


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    critic_observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def build_critic(input_dim: int, hidden_sizes: list[int], activation: str) -> Any:
    import torch.nn as nn

    class CriticNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs).squeeze(-1)

    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden_size)))
        layers.append(activation_module(activation))
        last_dim = int(hidden_size)
    layers.append(nn.Linear(last_dim, 1))
    return CriticNetwork()


def activation_module(name: str) -> Any:
    import torch.nn as nn

    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def build_args(
    get_args,
    config: dict[str, Any],
    run_name: str,
    rl_device: str,
    sim_device: str,
    seed: int | None,
) -> object:
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "train_omnisafe_ppolag_diagnostic.py",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bounded OmniSafe PPO-Lag Jacobian diagnostic.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--humanoid-gym-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rl-device", default=None)
    parser.add_argument("--sim-device", default=None)
    parser.add_argument("--write-failure-artifact", action="store_true")
    return parser.parse_args()


def rendered_invocation(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), f"--config={args.config}"]
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    if args.run_name:
        command.append(f"--run-name={args.run_name}")
    if args.num_envs is not None:
        command.append(f"--num-envs={args.num_envs}")
    if args.max_iterations is not None:
        command.append(f"--max-iterations={args.max_iterations}")
    if args.seed is not None:
        command.append(f"--seed={args.seed}")
    if args.rl_device:
        command.append(f"--rl-device={args.rl_device}")
    if args.sim_device:
        command.append(f"--sim-device={args.sim_device}")
    if args.write_failure_artifact:
        command.append("--write-failure-artifact")
    return command


def flattened_batch(batch: RolloutBatch) -> RolloutBatch:
    return RolloutBatch(
        observations=batch.observations.flatten(0, 1),
        critic_observations=batch.critic_observations.flatten(0, 1),
        actions=batch.actions.flatten(0, 1),
        old_log_probs=batch.old_log_probs.flatten(0, 1),
        old_values=batch.old_values.flatten(0, 1),
        returns=batch.returns.flatten(0, 1),
        advantages=batch.advantages.flatten(0, 1),
    )


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(last_values)
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]
        next_non_terminal = 1.0 - dones[step].float()
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * lam * next_non_terminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return returns, normalize_advantages(advantages)


def collect_rollout(
    *,
    env: Any,
    actor: Any,
    critic: Any,
    obs: torch.Tensor,
    critic_obs: torch.Tensor,
    num_steps: int,
    gamma: float,
    lam: float,
    episode_return: torch.Tensor,
    episode_length: torch.Tensor,
    return_buffer: deque[float],
    length_buffer: deque[float],
) -> tuple[RolloutBatch, torch.Tensor, torch.Tensor, dict[str, float]]:
    observations: list[torch.Tensor] = []
    critic_observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    old_values: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []

    collection_reward_mean = []
    with torch.inference_mode():
        for _ in range(num_steps):
            distribution = actor(obs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(axis=-1)
            value = critic(critic_obs)

            next_obs, next_privileged_obs, reward, done, infos = env.step(action)
            next_critic_obs = next_privileged_obs if next_privileged_obs is not None else next_obs
            reward = reward.to(obs.device)
            done = done.to(obs.device)
            time_outs = infos.get("time_outs") if isinstance(infos, dict) else None
            adjusted_reward = reward.clone()
            if time_outs is not None:
                adjusted_reward += gamma * value * time_outs.to(obs.device).float()

            observations.append(obs.detach())
            critic_observations.append(critic_obs.detach())
            actions.append(action.detach())
            old_log_probs.append(log_prob.detach())
            old_values.append(value.detach())
            rewards.append(adjusted_reward.detach())
            dones.append(done.detach())
            collection_reward_mean.append(float(reward.mean().item()))

            episode_return += reward
            episode_length += 1
            done_ids = torch.nonzero(done > 0, as_tuple=False).flatten()
            for env_id in done_ids.tolist():
                return_buffer.append(float(episode_return[env_id].item()))
                length_buffer.append(float(episode_length[env_id].item()))
                episode_return[env_id] = 0.0
                episode_length[env_id] = 0.0

            obs = next_obs.to(obs.device)
            critic_obs = next_critic_obs.to(obs.device)

        last_values = critic(critic_obs).detach()

    rewards_tensor = torch.stack(rewards)
    dones_tensor = torch.stack(dones)
    values_tensor = torch.stack(old_values)
    returns, advantages = compute_gae(
        rewards_tensor,
        dones_tensor,
        values_tensor,
        last_values,
        gamma=gamma,
        lam=lam,
    )
    batch = RolloutBatch(
        observations=torch.stack(observations),
        critic_observations=torch.stack(critic_observations),
        actions=torch.stack(actions),
        old_log_probs=torch.stack(old_log_probs),
        old_values=values_tensor,
        returns=returns,
        advantages=advantages,
    )
    stats = {
        "rollout_reward_mean": statistics.fmean(collection_reward_mean) if collection_reward_mean else 0.0,
        "episode_return_mean_100": statistics.fmean(return_buffer) if return_buffer else math.nan,
        "episode_length_mean_100": statistics.fmean(length_buffer) if length_buffer else math.nan,
    }
    return batch, obs, critic_obs, stats


def update_policy(
    *,
    actor: Any,
    critic: Any,
    optimizer: torch.optim.Optimizer,
    hook: Any,
    batch: RolloutBatch,
    clip_param: float,
    value_loss_coef: float,
    entropy_coef: float,
    num_learning_epochs: int,
    num_mini_batches: int,
    max_grad_norm: float,
    use_clipped_value_loss: bool,
) -> dict[str, Any]:
    flat = flattened_batch(batch)
    batch_size = flat.observations.shape[0]
    mini_batch_size = max(batch_size // num_mini_batches, 1)

    value_losses: list[float] = []
    surrogate_losses: list[float] = []
    entropy_values: list[float] = []
    penalty_losses: list[float] = []
    total_losses: list[float] = []
    grad_norms: list[float] = []
    costs_for_update: list[float] = []
    cost_means: list[float] = []
    cost_maxes: list[float] = []
    cost_quantiles: list[float] = []
    violation_rates: list[float] = []
    sample_counts: list[int] = []

    for _ in range(num_learning_epochs):
        indices = torch.randperm(batch_size, device=flat.observations.device)
        for start in range(0, batch_size, mini_batch_size):
            batch_indices = indices[start : start + mini_batch_size]
            obs_batch = flat.observations.index_select(0, batch_indices)
            critic_obs_batch = flat.critic_observations.index_select(0, batch_indices)
            actions_batch = flat.actions.index_select(0, batch_indices)
            old_log_probs_batch = flat.old_log_probs.index_select(0, batch_indices)
            old_values_batch = flat.old_values.index_select(0, batch_indices)
            returns_batch = flat.returns.index_select(0, batch_indices)
            advantages_batch = flat.advantages.index_select(0, batch_indices)

            distribution = actor(obs_batch)
            actions_log_prob = distribution.log_prob(actions_batch).sum(axis=-1)
            entropy = distribution.entropy().sum(axis=-1)
            value = critic(critic_obs_batch)

            ratio = torch.exp(actions_log_prob - old_log_probs_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if use_clipped_value_loss:
                value_clipped = old_values_batch + (value - old_values_batch).clamp(-clip_param, clip_param)
                value_losses_unclipped = torch.square(value - returns_batch)
                value_losses_clipped = torch.square(value_clipped - returns_batch)
                value_loss = torch.max(value_losses_unclipped, value_losses_clipped).mean()
            else:
                value_loss = torch.square(value - returns_batch).mean()

            base_loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()
            penalty, cost = hook.penalty_loss(obs_batch)
            total_loss = base_loss + penalty

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                max_norm=max_grad_norm,
            )
            optimizer.step()

            value_losses.append(float(value_loss.detach().item()))
            surrogate_losses.append(float(surrogate_loss.detach().item()))
            entropy_values.append(float(entropy.detach().mean().item()))
            penalty_losses.append(float(penalty.detach().item()))
            total_losses.append(float(total_loss.detach().item()))
            grad_norms.append(float(grad_norm))
            costs_for_update.append(float(cost.cost_for_update))
            cost_means.append(float(cost.cost_mean))
            cost_maxes.append(float(cost.cost_max))
            cost_quantiles.append(float(cost.cost_quantile))
            violation_rates.append(float(cost.violation_rate))
            sample_counts.append(int(cost.sample_count))

    mean_cost_update = statistics.fmean(costs_for_update) if costs_for_update else 0.0
    from _omnisafe_ppolag_jacobian_hook import JacobianCostTensorResult

    aggregate_cost = JacobianCostTensorResult(
        cost_tensor=torch.tensor(mean_cost_update, device=flat.observations.device),
        cost_mean=statistics.fmean(cost_means) if cost_means else 0.0,
        cost_max=statistics.fmean(cost_maxes) if cost_maxes else 0.0,
        cost_quantile=statistics.fmean(cost_quantiles) if cost_quantiles else 0.0,
        cost_for_update=mean_cost_update,
        violation_rate=statistics.fmean(violation_rates) if violation_rates else 0.0,
        threshold=float(hook.threshold),
        sample_count=sum(sample_counts),
        per_sample_costs=[],
    )
    multiplier_before, multiplier_after = hook.update_multiplier(aggregate_cost)

    return {
        "value_loss": statistics.fmean(value_losses) if value_losses else 0.0,
        "surrogate_loss": statistics.fmean(surrogate_losses) if surrogate_losses else 0.0,
        "entropy": statistics.fmean(entropy_values) if entropy_values else 0.0,
        "constraint_penalty_loss_mean": statistics.fmean(penalty_losses) if penalty_losses else 0.0,
        "total_loss": statistics.fmean(total_losses) if total_losses else 0.0,
        "grad_norm": statistics.fmean(grad_norms) if grad_norms else 0.0,
        "lagrange_multiplier_before": multiplier_before,
        "lagrange_multiplier": multiplier_after,
        "lagrange_delta": multiplier_after - multiplier_before,
        "constraint_threshold": float(hook.threshold),
        "policy_local_sensitivity_cost_mean": aggregate_cost.cost_mean,
        "policy_local_sensitivity_cost_update": aggregate_cost.cost_for_update,
        "policy_local_sensitivity_cost_max": aggregate_cost.cost_max,
        "policy_local_sensitivity_cost_quantile": aggregate_cost.cost_quantile,
        "constraint_error": aggregate_cost.cost_for_update - float(hook.threshold),
        "constraint_violation_rate": aggregate_cost.violation_rate,
        "constraint_sample_count": aggregate_cost.sample_count,
    }


def write_failure_artifact(output_dir: Path, run_name: str, command: list[str], exc: BaseException) -> None:
    write_json(
        output_dir / "omnisafe_training.json",
        {
            "status": "failed",
            "run_name": run_name,
            "command": command,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": "Failure occurred before completing the #63 bounded OmniSafe PPO-Lag diagnostic training.",
        },
    )


def checkpoint_ids(max_iterations: int, save_interval: int) -> list[int]:
    ids = list(range(0, max_iterations, save_interval))
    if max_iterations not in ids:
        ids.append(max_iterations)
    return ids


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    run_name = args.run_name or config["run_name"]
    output_dir = ensure_directory(artifact_dir(config, run_name))
    checkpoint_dir = ensure_directory(output_dir / "checkpoints")
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    command = rendered_invocation(args)

    try:
        ensure_humanoid_gym_checkout(humanoid_gym_root)
        configure_runtime_env()
        ensure_upstream_on_syspath(humanoid_gym_root)

        import isaacgym  # noqa: F401
        global np, torch
        import numpy as np
        import torch
        from omnisafe.common.lagrange import Lagrange

        from _omnisafe_policy_loader import create_gaussian_actor, save_omnisafe_policy_checkpoint
        from _omnisafe_ppolag_jacobian_hook import HOOK_IMPLEMENTATION_BOUNDARY, JacobianPPOLagUpdateHook

        importlib.import_module("humanoid.envs")
        from humanoid.utils import get_args, task_registry

        seed = int(args.seed if args.seed is not None else training.get("seed", 1))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rl_device = args.rl_device or training["rl_device"]
        sim_device = args.sim_device or training["sim_device"]
        num_envs = int(args.num_envs if args.num_envs is not None else training["num_envs"])
        max_iterations = int(args.max_iterations if args.max_iterations is not None else training["max_iterations"])
        num_steps = int(training["num_steps_per_env"])
        save_interval = int(training["save_interval"])

        upstream_args = build_args(get_args, config, run_name, rl_device, sim_device, seed)
        upstream_args.num_envs = num_envs

        env_cfg, train_cfg = task_registry.get_cfgs(name=config["task"])
        env_cfg, train_cfg = apply_method_overrides(env_cfg, train_cfg, config)
        env_cfg.env.num_envs = num_envs
        if hasattr(env_cfg, "terrain"):
            env_cfg.terrain.curriculum = False
        env_cfg.seed = seed
        train_cfg.seed = seed

        env, _ = task_registry.make_env(name=config["task"], args=upstream_args, env_cfg=env_cfg)
        reset_obs, reset_privileged_obs = env.reset()
        obs = reset_obs if reset_obs is not None else env.get_observations()
        privileged_obs = reset_privileged_obs if reset_privileged_obs is not None else env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs = obs.to(env.device)
        critic_obs = critic_obs.to(env.device)

        actor = create_gaussian_actor(
            obs_dim=int(env.num_obs),
            act_dim=int(env.num_actions),
            hidden_sizes=[int(v) for v in training["actor_hidden_sizes"]],
            activation=str(training["activation"]),
            device=env.device,
        )
        actor.std = float(training.get("init_noise_std", 1.0))
        critic = build_critic(
            input_dim=int(critic_obs.shape[-1]),
            hidden_sizes=[int(v) for v in training["critic_hidden_sizes"]],
            activation=str(training["activation"]),
        ).to(env.device)
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=float(training["learning_rate"]),
        )

        cost_cfg = config["omnisafe"]["cost_config"]
        lagrange = Lagrange(
            cost_limit=float(cost_cfg["threshold"]),
            lagrangian_multiplier_init=float(cost_cfg.get("multiplier_init", 0.5)),
            lambda_lr=float(cost_cfg.get("lambda_lr", 0.01)),
            lambda_optimizer=str(cost_cfg.get("lambda_optimizer", "SGD")),
            lagrangian_upper_bound=float(cost_cfg.get("lagrangian_upper_bound", 5.0)),
        )
        hook = JacobianPPOLagUpdateHook(
            actor,
            lagrange,
            threshold=float(cost_cfg["threshold"]),
            subsample_obs=int(cost_cfg.get("subsample_obs", 8)),
            cost_aggregation=str(cost_cfg.get("cost_aggregation", "quantile")),
            cost_quantile=float(cost_cfg.get("cost_quantile", 0.9)),
            epsilon=float(cost_cfg.get("epsilon", 1e-6)),
        )

        episode_return = torch.zeros(env.num_envs, device=env.device)
        episode_length = torch.zeros(env.num_envs, device=env.device)
        return_buffer: deque[float] = deque(maxlen=100)
        length_buffer: deque[float] = deque(maxlen=100)
        trace: list[dict[str, Any]] = []
        saved_checkpoints: dict[str, str] = {}
        started_at = time.time()

        for iteration in range(max_iterations):
            actor.train()
            critic.train()
            collection_start = time.time()
            batch, obs, critic_obs, rollout_stats = collect_rollout(
                env=env,
                actor=actor,
                critic=critic,
                obs=obs,
                critic_obs=critic_obs,
                num_steps=num_steps,
                gamma=float(training["gamma"]),
                lam=float(training["lam"]),
                episode_return=episode_return,
                episode_length=episode_length,
                return_buffer=return_buffer,
                length_buffer=length_buffer,
            )
            collection_time = time.time() - collection_start

            update_start = time.time()
            update_stats = update_policy(
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                hook=hook,
                batch=batch,
                clip_param=float(training["clip_param"]),
                value_loss_coef=float(training["value_loss_coef"]),
                entropy_coef=float(training["entropy_coef"]),
                num_learning_epochs=int(training["num_learning_epochs"]),
                num_mini_batches=int(training["num_mini_batches"]),
                max_grad_norm=float(training["max_grad_norm"]),
                use_clipped_value_loss=bool(training["use_clipped_value_loss"]),
            )
            update_time = time.time() - update_start

            trace_entry = {
                "iteration": iteration,
                "collection_time_s": collection_time,
                "update_time_s": update_time,
                **rollout_stats,
                **update_stats,
            }
            trace.append(trace_entry)
            print(
                f"it={iteration:04d} "
                f"rew={rollout_stats['rollout_reward_mean']:.4f} "
                f"ep_ret100={rollout_stats['episode_return_mean_100']:.4f} "
                f"cost={update_stats['policy_local_sensitivity_cost_update']:.4f} "
                f"lambda={update_stats['lagrange_multiplier']:.4f} "
                f"loss={update_stats['total_loss']:.4f}",
                flush=True,
            )

            if iteration % save_interval == 0:
                path = checkpoint_dir / f"model_{iteration}.pt"
                save_omnisafe_policy_checkpoint(
                    path,
                    actor,
                    checkpoint=iteration,
                    seed=seed,
                    cost_config=cost_cfg,
                    metadata={
                        "source": "#63 bounded OmniSafe PPO-Lag diagnostic",
                        "iteration": iteration,
                        "latest_training_stats": trace_entry,
                        "boundary": HOOK_IMPLEMENTATION_BOUNDARY,
                    },
                )
                saved_checkpoints[str(iteration)] = relative_to_repo(path)

        final_checkpoint = checkpoint_dir / f"model_{max_iterations}.pt"
        save_omnisafe_policy_checkpoint(
            final_checkpoint,
            actor,
            checkpoint=max_iterations,
            seed=seed,
            cost_config=cost_cfg,
            metadata={
                "source": "#63 bounded OmniSafe PPO-Lag diagnostic",
                "iteration": max_iterations,
                "latest_training_stats": trace[-1] if trace else {},
                "boundary": HOOK_IMPLEMENTATION_BOUNDARY,
            },
        )
        saved_checkpoints[str(max_iterations)] = relative_to_repo(final_checkpoint)

        latest = trace[-1] if trace else {}
        constraint_metrics = {
            "constraint_sample_count": int(sum(entry["constraint_sample_count"] for entry in trace)),
            "constraint_cost_aggregation": cost_cfg.get("cost_aggregation"),
            "constraint_cost_quantile": float(cost_cfg.get("cost_quantile", 0.9)),
            "constraint_subsample_obs": int(cost_cfg.get("subsample_obs", 8)),
            "constraint_sampling_mode": "random_subsample",
            "constraint_violation_rate": latest.get("constraint_violation_rate"),
            "dual_update_mode": "omnisafe_lagrange_sgd",
            "lagrange_multiplier": latest.get("lagrange_multiplier"),
            "lagrange_multiplier_max": float(cost_cfg.get("lagrangian_upper_bound", 5.0)),
            "local_sensitivity_threshold": float(cost_cfg["threshold"]),
            "policy_local_sensitivity_cost_mean": latest.get("policy_local_sensitivity_cost_mean"),
            "policy_local_sensitivity_cost_update": latest.get("policy_local_sensitivity_cost_update"),
            "policy_local_sensitivity_cost_max": latest.get("policy_local_sensitivity_cost_max"),
            "policy_local_sensitivity_cost_quantile": latest.get("policy_local_sensitivity_cost_quantile"),
            "policy_local_sensitivity_cost_std": (
                statistics.pstdev(entry["policy_local_sensitivity_cost_mean"] for entry in trace)
                if len(trace) > 1
                else 0.0
            ),
        }
        write_json(output_dir / "constraint_metrics.json", constraint_metrics)
        write_json(output_dir / "lagrange_multiplier_trace.json", {"trace": trace})

        training_status = {
            "status": "complete",
            "run_name": run_name,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "issue": "#63",
            "task": config["task"],
            "seed": seed,
            "num_envs": num_envs,
            "num_steps_per_env": num_steps,
            "max_iterations": max_iterations,
            "save_interval": save_interval,
            "expected_checkpoints": checkpoint_ids(max_iterations, save_interval),
            "checkpoint_dir": relative_to_repo(checkpoint_dir),
            "checkpoint_paths": saved_checkpoints,
            "latest_checkpoint_path": relative_to_repo(final_checkpoint),
            "elapsed_time_s": time.time() - started_at,
            "latest_training_stats": latest,
            "command": command,
            "boundary": {
                "diagnostic_only": True,
                "site_packages_modified": False,
                "uses_omnisafe_components": ["GaussianLearningActor", "Lagrange"],
                "uses_project_bridge": "JacobianPPOLagUpdateHook from #65",
                "not_a_drop_in_omnisafe_ppolag_env": True,
            },
        }
        write_json(output_dir / "omnisafe_training.json", training_status)

        manifest = default_manifest(config, humanoid_gym_root)
        manifest["run_name"] = run_name
        manifest["training_command"] = command
        manifest["training_status_path"] = relative_to_repo(output_dir / "omnisafe_training.json")
        manifest["checkpoint_dir"] = relative_to_repo(checkpoint_dir)
        manifest["checkpoint_path"] = relative_to_repo(final_checkpoint)
        manifest["checkpoint_paths"] = saved_checkpoints
        manifest["constraint_metrics_path"] = relative_to_repo(output_dir / "constraint_metrics.json")
        manifest["lagrange_multiplier_trace_path"] = relative_to_repo(output_dir / "lagrange_multiplier_trace.json")
        manifest["training_status"] = training_status
        write_json(output_dir / "manifest.json", manifest)

        print(f"Wrote {relative_to_repo(output_dir / 'omnisafe_training.json')}")
        return 0
    except Exception as exc:
        if args.write_failure_artifact:
            write_failure_artifact(output_dir, run_name, command, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
