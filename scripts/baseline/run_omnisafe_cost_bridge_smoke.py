#!/usr/bin/env python3
"""One-seed Humanoid-Gym smoke for the OmniSafe Lagrange cost bridge."""

from __future__ import annotations

import argparse
import importlib
import sys
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


DEFAULT_CONFIG = "configs/methods/omnisafe_ppolag_cost_bridge_smoke.json"


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
            "run_omnisafe_cost_bridge_smoke.py",
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


def tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def tensor_summary(value: Any) -> dict[str, Any]:
    import torch

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    detached = value.detach()
    finite = torch.isfinite(detached).all().item() if detached.numel() else True
    summary: dict[str, Any] = {
        "shape": tensor_shape(detached),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite": bool(finite),
        "numel": int(detached.numel()),
    }
    if detached.numel() and detached.is_floating_point():
        summary.update(
            {
                "mean": float(detached.mean().item()),
                "min": float(detached.min().item()),
                "max": float(detached.max().item()),
            }
        )
    return summary


def finite_numbers(payload: dict[str, Any], keys: list[str]) -> bool:
    import math

    return all(math.isfinite(float(payload[key])) for key in keys)


def rendered_invocation(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    command.append(f"--config={args.config}")
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    if args.run_name:
        command.append(f"--run-name={args.run_name}")
    if args.artifacts_root:
        command.append(f"--artifacts-root={args.artifacts_root}")
    if args.num_envs is not None:
        command.append(f"--num-envs={args.num_envs}")
    if args.seed is not None:
        command.append(f"--seed={args.seed}")
    if args.rl_device:
        command.append(f"--rl-device={args.rl_device}")
    if args.sim_device:
        command.append(f"--sim-device={args.sim_device}")
    if args.threshold is not None:
        command.append(f"--threshold={args.threshold}")
    if args.subsample_obs is not None:
        command.append(f"--subsample-obs={args.subsample_obs}")
    if args.cost_aggregation:
        command.append(f"--cost-aggregation={args.cost_aggregation}")
    if args.cost_quantile is not None:
        command.append(f"--cost-quantile={args.cost_quantile}")
    if args.lambda_lr is not None:
        command.append(f"--lambda-lr={args.lambda_lr}")
    if args.lambda_optimizer:
        command.append(f"--lambda-optimizer={args.lambda_optimizer}")
    if args.multiplier_init is not None:
        command.append(f"--multiplier-init={args.multiplier_init}")
    if args.lagrangian_upper_bound is not None:
        command.append(f"--lagrangian-upper-bound={args.lagrangian_upper_bound}")
    return command


def write_failure_artifact(
    output_dir: Path,
    config: dict[str, Any],
    run_name: str,
    command: list[str],
    exc: BaseException,
) -> Path:
    payload = {
        "status": "failed",
        "run_name": run_name,
        "command": command,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "cost_source": config.get("smoke", {}).get("cost_source"),
        "cost_is_canonical": bool(config.get("smoke", {}).get("cost_is_canonical", True)),
        "note": "Failure occurred before completing the real Humanoid-Gym cost-bridge smoke.",
    }
    output_path = output_dir / "omnisafe_cost_bridge_smoke.json"
    write_json(output_path, payload)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real one-seed OmniSafe Lagrange cost-bridge smoke (#61).")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to method config JSON.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument("--run-name", default=None, help="Override the configured run_name.")
    parser.add_argument("--artifacts-root", default=None, help="Override artifacts root dir.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override smoke num_envs.")
    parser.add_argument("--seed", type=int, default=None, help="Override smoke seed.")
    parser.add_argument("--rl-device", default=None, help="Override RL device.")
    parser.add_argument("--sim-device", default=None, help="Override sim device.")
    parser.add_argument("--threshold", type=float, default=None, help="Override local-sensitivity threshold.")
    parser.add_argument("--subsample-obs", type=int, default=None, help="Override Jacobian observation subsample.")
    parser.add_argument("--cost-aggregation", default=None, choices=["mean", "max", "quantile"])
    parser.add_argument("--cost-quantile", type=float, default=None)
    parser.add_argument("--lambda-lr", type=float, default=None)
    parser.add_argument("--lambda-optimizer", default=None)
    parser.add_argument("--multiplier-init", type=float, default=None)
    parser.add_argument("--lagrangian-upper-bound", type=float, default=None)
    parser.add_argument("--write-failure-artifact", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(args.config)
    if args.artifacts_root:
        config["artifacts_root"] = args.artifacts_root
    smoke_cfg = config.get("smoke", {})
    run_name = args.run_name or config["run_name"]
    output_dir = ensure_directory(artifact_dir(config, run_name))
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    command = rendered_invocation(args)

    try:
        ensure_humanoid_gym_checkout(humanoid_gym_root)
        configure_runtime_env()
        ensure_upstream_on_syspath(humanoid_gym_root)

        import isaacgym  # noqa: F401
        import torch
        from humanoid.algo.ppo.actor_critic import ActorCritic

        from _omnisafe_bridge import run_bridge_smoke

        importlib.import_module("humanoid.envs")
        from humanoid.utils import class_to_dict, get_args, task_registry

        training = config["training"]
        rl_device = args.rl_device or training["rl_device"]
        sim_device = args.sim_device or training["sim_device"]
        seed = args.seed if args.seed is not None else int(smoke_cfg.get("seed", 23))
        num_envs = args.num_envs if args.num_envs is not None else int(training["num_envs"])

        upstream_args = build_args(get_args, config, run_name, rl_device, sim_device, seed)
        upstream_args.num_envs = num_envs

        env_cfg, train_cfg = task_registry.get_cfgs(name=config["task"])
        env_cfg, train_cfg = apply_method_overrides(env_cfg, train_cfg, config)
        env_cfg.env.num_envs = num_envs
        if hasattr(env_cfg, "terrain"):
            env_cfg.terrain.curriculum = False
        env_cfg.seed = seed

        env, _ = task_registry.make_env(name=config["task"], args=upstream_args, env_cfg=env_cfg)
        reset_obs, _ = env.reset()
        obs = reset_obs if reset_obs is not None else env.get_observations()

        policy_cfg = class_to_dict(train_cfg.policy)
        critic_obs_dim = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs
        actor_critic = ActorCritic(env.num_obs, critic_obs_dim, env.num_actions, **policy_cfg).to(obs.device)
        actor_critic.eval()

        with torch.no_grad():
            action = actor_critic.act_inference(obs)
            next_obs, privileged_obs, reward, done, upstream_info = env.step(action)

        threshold = args.threshold if args.threshold is not None else float(smoke_cfg.get("threshold", 3.8))
        subsample_obs = args.subsample_obs if args.subsample_obs is not None else int(smoke_cfg.get("subsample_obs", 8))
        cost_aggregation = args.cost_aggregation or str(smoke_cfg.get("cost_aggregation", "quantile"))
        cost_quantile = args.cost_quantile if args.cost_quantile is not None else float(smoke_cfg.get("cost_quantile", 0.9))
        lambda_lr = args.lambda_lr if args.lambda_lr is not None else float(smoke_cfg.get("lambda_lr", 0.01))
        lambda_optimizer = args.lambda_optimizer or str(smoke_cfg.get("lambda_optimizer", "SGD"))
        multiplier_init = (
            args.multiplier_init if args.multiplier_init is not None else float(smoke_cfg.get("multiplier_init", 0.5))
        )
        upper_bound = (
            args.lagrangian_upper_bound
            if args.lagrangian_upper_bound is not None
            else float(smoke_cfg.get("lagrangian_upper_bound", 5.0))
        )
        epsilon = float(smoke_cfg.get("epsilon", 1e-6))

        bridge_result = run_bridge_smoke(
            actor_critic,
            obs,
            threshold=threshold,
            multiplier_init=multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
            lagrangian_upper_bound=upper_bound,
            subsample_obs=subsample_obs,
            cost_aggregation=cost_aggregation,
            cost_quantile=cost_quantile,
            epsilon=epsilon,
        )

        cost = bridge_result["cost"]
        update = bridge_result["update"]
        required_cost_keys = [
            "cost_mean",
            "cost_max",
            "cost_quantile",
            "cost_for_update",
            "violation_rate",
        ]
        required_update_keys = ["multiplier", "multiplier_before", "lambda_loss", "constraint_error"]
        if not finite_numbers(cost, required_cost_keys):
            raise RuntimeError(f"Non-finite cost diagnostics: {cost}")
        if not finite_numbers(update, required_update_keys):
            raise RuntimeError(f"Non-finite multiplier diagnostics: {update}")
        if not bool(tensor_summary(reward)["finite"]):
            raise RuntimeError("Non-finite reward tensor in cost-bridge smoke.")

        payload = {
            "status": "complete",
            "bridge_status": "lagrange_component_smoke_ok",
            "run_name": run_name,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": command,
            "issue": "#61",
            "task": config["task"],
            "seed": seed,
            "num_envs": num_envs,
            "rl_device": rl_device,
            "sim_device": sim_device,
            "actor_critic": {
                "obs_dim": int(env.num_obs),
                "critic_obs_dim": int(critic_obs_dim),
                "act_dim": int(env.num_actions),
                "num_params": int(sum(p.numel() for p in actor_critic.parameters())),
            },
            "rollout": {
                "observation": tensor_summary(obs),
                "next_observation": tensor_summary(next_obs),
                "privileged_observation": tensor_summary(privileged_obs) if privileged_obs is not None else None,
                "action": tensor_summary(action),
                "reward": tensor_summary(reward),
                "done": tensor_summary(done),
                "humanoid_gym_info_keys": sorted(upstream_info.keys()) if isinstance(upstream_info, dict) else [],
            },
            "bridge": bridge_result,
            "boundary": {
                "full_omnisafe_ppolag_training_bridge": "not_implemented",
                "reason": (
                    "OmniSafe PPOLag consumes environment-side episode costs and cost advantages. "
                    "The SC-PPO cost is actor-internal and computed on policy observations during the PPO update. "
                    "A faithful full PPOLag baseline would require an algorithm/update hook rather than a pure env adapter."
                ),
            },
        }

        output_path = output_dir / "omnisafe_cost_bridge_smoke.json"
        write_json(output_path, payload)

        manifest = default_manifest(config, humanoid_gym_root)
        manifest["run_name"] = run_name
        manifest["cost_bridge_smoke_path"] = relative_to_repo(output_path)
        manifest["smoke"] = payload
        write_json(output_dir / "manifest.json", manifest)

        print(f"Wrote {relative_to_repo(output_path)}")
        print(
            "SMOKE OK: "
            f"reward_mean={payload['rollout']['reward']['mean']:.4f}, "
            f"cost={cost['cost_for_update']:.4f}, "
            f"violation_rate={cost['violation_rate']:.4f}, "
            f"lambda={update['multiplier_before']:.4f}->{update['multiplier']:.4f}"
        )
        return 0
    except Exception as exc:
        if args.write_failure_artifact:
            output_path = write_failure_artifact(output_dir, config, run_name, command, exc)
            print(f"Wrote failure artifact to {relative_to_repo(output_path)}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
