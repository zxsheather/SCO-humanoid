#!/usr/bin/env python3
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


DEFAULT_CONFIG = "configs/methods/omnisafe_ppolag_update_hook_smoke.json"


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
            "run_omnisafe_update_hook_smoke.py",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the OmniSafe PPO-Lag Jacobian update hook.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--humanoid-gym-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
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
    return command


def write_failure_artifact(output_dir: Path, run_name: str, command: list[str], exc: BaseException) -> Path:
    output_path = output_dir / "omnisafe_update_hook_smoke.json"
    write_json(
        output_path,
        {
            "status": "failed",
            "run_name": run_name,
            "command": command,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": "Failure occurred before completing the #65 update-hook smoke.",
        },
    )
    return output_path


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
        import numpy as np
        import torch
        from gymnasium import spaces
        from omnisafe.common.lagrange import Lagrange
        from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor

        from _omnisafe_ppolag_jacobian_hook import HOOK_IMPLEMENTATION_BOUNDARY, JacobianPPOLagUpdateHook

        importlib.import_module("humanoid.envs")
        from humanoid.utils import get_args, task_registry

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

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(env.num_obs),), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(int(env.num_actions),), dtype=np.float32)
        actor = GaussianLearningActor(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=list(smoke_cfg.get("actor_hidden_sizes", [64, 64])),
            activation=str(smoke_cfg.get("actor_activation", "tanh")),
        ).to(obs.device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=float(smoke_cfg.get("actor_lr", 3e-4)))

        lagrange = Lagrange(
            cost_limit=float(smoke_cfg.get("threshold", 3.8)),
            lagrangian_multiplier_init=float(smoke_cfg.get("multiplier_init", 0.5)),
            lambda_lr=float(smoke_cfg.get("lambda_lr", 0.01)),
            lambda_optimizer=str(smoke_cfg.get("lambda_optimizer", "SGD")),
            lagrangian_upper_bound=float(smoke_cfg.get("lagrangian_upper_bound", 5.0)),
        )
        hook = JacobianPPOLagUpdateHook(
            actor,
            lagrange,
            threshold=float(smoke_cfg.get("threshold", 3.8)),
            subsample_obs=int(smoke_cfg.get("subsample_obs", 1)),
            cost_aggregation=str(smoke_cfg.get("cost_aggregation", "quantile")),
            cost_quantile=float(smoke_cfg.get("cost_quantile", 0.9)),
            epsilon=float(smoke_cfg.get("epsilon", 1e-6)),
        )

        with torch.no_grad():
            action = actor.predict(obs, deterministic=True)
            next_obs, privileged_obs, reward, done, upstream_info = env.step(action)

        action_for_loss = action.detach()
        old_distribution = actor(obs.detach())
        old_logp = old_distribution.log_prob(action_for_loss).sum(axis=-1).detach()
        distribution = actor(obs.detach())
        logp = actor.log_prob(action_for_loss)
        ratio = torch.exp(logp - old_logp)
        advantage = torch.ones_like(ratio)
        base_policy_loss = -(ratio * advantage).mean()

        update = hook.actor_update(
            optimizer=optimizer,
            base_policy_loss=base_policy_loss,
            obs_batch=obs,
        )

        if not torch.isfinite(reward).all().item():
            raise RuntimeError("Non-finite reward tensor in #65 update-hook smoke.")
        if not np.isfinite(update.total_loss):
            raise RuntimeError("Non-finite total loss in #65 update-hook smoke.")
        if not np.isfinite(update.grad_norm):
            raise RuntimeError("Non-finite actor gradient norm in #65 update-hook smoke.")

        payload = {
            "status": "complete",
            "run_name": run_name,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": command,
            "issue": "#65",
            "task": config["task"],
            "seed": seed,
            "num_envs": num_envs,
            "rl_device": rl_device,
            "sim_device": sim_device,
            "rollout": {
                "observation": tensor_summary(obs),
                "next_observation": tensor_summary(next_obs),
                "privileged_observation": tensor_summary(privileged_obs) if privileged_obs is not None else None,
                "action": tensor_summary(action),
                "reward": tensor_summary(reward),
                "done": tensor_summary(done),
                "humanoid_gym_info_keys": sorted(upstream_info.keys()) if isinstance(upstream_info, dict) else [],
            },
            "update_hook": {
                "base_loss": update.base_loss,
                "penalty_loss": update.penalty_loss,
                "total_loss": update.total_loss,
                "grad_norm": update.grad_norm,
                "multiplier_before": update.multiplier_before,
                "multiplier_after": update.multiplier_after,
                "constraint_error": update.constraint_error,
                "cost": {
                    "cost_mean": update.cost.cost_mean,
                    "cost_max": update.cost.cost_max,
                    "cost_quantile": update.cost.cost_quantile,
                    "cost_for_update": update.cost.cost_for_update,
                    "violation_rate": update.cost.violation_rate,
                    "threshold": update.cost.threshold,
                    "sample_count": update.cost.sample_count,
                    "per_sample_costs": update.cost.per_sample_costs,
                },
            },
            "boundary": HOOK_IMPLEMENTATION_BOUNDARY,
        }

        output_path = output_dir / "omnisafe_update_hook_smoke.json"
        write_json(output_path, payload)
        manifest = default_manifest(config, humanoid_gym_root)
        manifest["run_name"] = run_name
        manifest["update_hook_smoke_path"] = relative_to_repo(output_path)
        manifest["smoke"] = payload
        write_json(output_dir / "manifest.json", manifest)

        print(f"Wrote {relative_to_repo(output_path)}")
        print(
            "SMOKE OK: "
            f"reward_mean={payload['rollout']['reward']['mean']:.4f}, "
            f"cost={update.cost.cost_for_update:.4f}, "
            f"violation_rate={update.cost.violation_rate:.4f}, "
            f"lambda={update.multiplier_before:.4f}->{update.multiplier_after:.4f}, "
            f"total_loss={update.total_loss:.4f}"
        )
        return 0
    except Exception as exc:
        if args.write_failure_artifact:
            output_path = write_failure_artifact(output_dir, run_name, command, exc)
            print(f"Wrote failure artifact to {relative_to_repo(output_path)}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
