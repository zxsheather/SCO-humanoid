#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
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


DEFAULT_CONFIG = "configs/methods/omnisafe_ppolag_adapter_smoke.json"


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
            "run_omnisafe_adapter_smoke.py",
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


def tensor_dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else None


def tensor_device(value: Any) -> str | None:
    device = getattr(value, "device", None)
    return str(device) if device is not None else None


def bool_tensor(value: Any, *, like: Any | None = None):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=torch.bool)
    else:
        tensor = torch.as_tensor(value, dtype=torch.bool)
    device = getattr(like, "device", None)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


class OmniSafeHumanoidGymAdapter:
    """Thin adapter that exposes Humanoid-Gym steps in OmniSafe CMDP tuple form."""

    def __init__(self, env: Any, *, cost_source: str, cost_is_canonical: bool) -> None:
        self.env = env
        self.cost_source = cost_source
        self.cost_is_canonical = cost_is_canonical

    @property
    def num_envs(self) -> int:
        return int(self.env.num_envs)

    @property
    def num_actions(self) -> int:
        return int(self.env.num_actions)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        reset = getattr(self.env, "reset", None)
        if callable(reset):
            reset()
        obs = self.env.get_observations()
        return obs, {
            "adapter": "omnisafe_humanoid_gym_smoke",
            "cost_source": self.cost_source,
            "cost_is_canonical": self.cost_is_canonical,
        }

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        import torch

        obs, _, reward, done, upstream_info = self.env.step(action)
        done_bool = bool_tensor(done)
        time_outs = upstream_info.get("time_outs") if isinstance(upstream_info, dict) else None
        if time_outs is None:
            truncated = torch.zeros_like(done_bool, dtype=torch.bool)
        else:
            truncated = bool_tensor(time_outs, like=done_bool) & done_bool
        terminated = done_bool & ~truncated
        cost = torch.zeros_like(reward, dtype=reward.dtype, device=reward.device)
        info = {
            "adapter": "omnisafe_humanoid_gym_smoke",
            "cost_source": self.cost_source,
            "cost_is_canonical": self.cost_is_canonical,
            "humanoid_gym_info_keys": sorted(upstream_info.keys()) if isinstance(upstream_info, dict) else [],
            "final_observation": obs,
        }
        return obs, reward, cost, terminated, truncated, info


def summarize_step(step_index: int, action: Any, result: tuple[Any, Any, Any, Any, Any, dict[str, Any]]) -> dict[str, Any]:
    obs, reward, cost, terminated, truncated, info = result
    return {
        "step": step_index,
        "action_shape": tensor_shape(action),
        "action_dtype": tensor_dtype(action),
        "action_device": tensor_device(action),
        "observation_shape": tensor_shape(obs),
        "observation_dtype": tensor_dtype(obs),
        "observation_device": tensor_device(obs),
        "reward_shape": tensor_shape(reward),
        "reward_dtype": tensor_dtype(reward),
        "cost_shape": tensor_shape(cost),
        "cost_dtype": tensor_dtype(cost),
        "terminated_shape": tensor_shape(terminated),
        "truncated_shape": tensor_shape(truncated),
        "cost_source": info.get("cost_source"),
        "cost_is_canonical": bool(info.get("cost_is_canonical")),
        "humanoid_gym_info_keys": info.get("humanoid_gym_info_keys", []),
    }


def make_zero_action(adapter: OmniSafeHumanoidGymAdapter, obs: Any):
    import torch

    return torch.zeros(adapter.num_envs, adapter.num_actions, dtype=obs.dtype, device=obs.device)


def write_failure_artifact(output_dir: Path, config: dict[str, Any], run_name: str, command: list[str], exc: BaseException) -> Path:
    payload = {
        "status": "failed",
        "run_name": run_name,
        "command": command,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "cost_source": config.get("smoke", {}).get("cost_source"),
        "cost_is_canonical": bool(config.get("smoke", {}).get("cost_is_canonical", False)),
        "note": "Failure occurred before completing the Humanoid-Gym to OmniSafe tuple smoke.",
    }
    output_path = output_dir / "omnisafe_adapter_smoke.json"
    write_json(output_path, payload)
    return output_path


def rendered_invocation(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    command.append(f"--config={args.config}")
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    if args.run_name:
        command.append(f"--run-name={args.run_name}")
    if args.num_envs is not None:
        command.append(f"--num-envs={args.num_envs}")
    if args.steps is not None:
        command.append(f"--steps={args.steps}")
    if args.rl_device:
        command.append(f"--rl-device={args.rl_device}")
    if args.sim_device:
        command.append(f"--sim-device={args.sim_device}")
    if args.seed is not None:
        command.append(f"--seed={args.seed}")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the Humanoid-Gym to OmniSafe CMDP tuple adapter.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the smoke config JSON.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument("--run-name", default=None, help="Override the configured run_name.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override smoke num_envs.")
    parser.add_argument("--steps", type=int, default=None, help="Number of adapter steps to execute.")
    parser.add_argument("--rl-device", default=None, help="Override RL device.")
    parser.add_argument("--sim-device", default=None, help="Override sim device.")
    parser.add_argument("--seed", type=int, default=None, help="Override env seed.")
    parser.add_argument("--write-failure-artifact", action="store_true", help="Write a failure artifact before re-raising.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = args.run_name or config["run_name"]
    command = rendered_invocation(args)
    output_dir = ensure_directory(artifact_dir(config, run_name))
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    smoke_cfg = config.get("smoke", {})

    try:
        ensure_humanoid_gym_checkout(humanoid_gym_root)
        configure_runtime_env()
        ensure_upstream_on_syspath(humanoid_gym_root)

        importlib.import_module("humanoid.envs")
        from humanoid.utils import get_args, task_registry

        training = config["training"]
        rl_device = args.rl_device or training["rl_device"]
        sim_device = args.sim_device or training["sim_device"]
        seed = args.seed
        upstream_args = build_args(get_args, config, run_name, rl_device, sim_device, seed)
        upstream_args.num_envs = args.num_envs or int(training["num_envs"])

        env_cfg, train_cfg = task_registry.get_cfgs(name=config["task"])
        env_cfg, train_cfg = apply_method_overrides(env_cfg, train_cfg, config)
        env_cfg.env.num_envs = upstream_args.num_envs
        if hasattr(env_cfg, "terrain"):
            env_cfg.terrain.curriculum = False
        if seed is not None:
            env_cfg.seed = seed

        env, _ = task_registry.make_env(name=config["task"], args=upstream_args, env_cfg=env_cfg)
        adapter = OmniSafeHumanoidGymAdapter(
            env,
            cost_source=str(smoke_cfg.get("cost_source", "non_canonical_zero_smoke")),
            cost_is_canonical=bool(smoke_cfg.get("cost_is_canonical", False)),
        )

        obs, reset_info = adapter.reset()
        steps = args.steps if args.steps is not None else int(smoke_cfg.get("steps", 1))
        step_summaries: list[dict[str, Any]] = []
        for step_index in range(steps):
            action = make_zero_action(adapter, obs)
            result = adapter.step(action)
            step_summaries.append(summarize_step(step_index, action, result))
            obs = result[0]

        payload = {
            "status": "complete",
            "run_name": run_name,
            "command": command,
            "task": config["task"],
            "num_envs": adapter.num_envs,
            "num_actions": adapter.num_actions,
            "rl_device": rl_device,
            "sim_device": sim_device,
            "seed": seed,
            "reset": {
                "observation_shape": tensor_shape(obs),
                "observation_dtype": tensor_dtype(obs),
                "observation_device": tensor_device(obs),
                "info": {key: value for key, value in reset_info.items() if key != "final_observation"},
            },
            "steps": step_summaries,
            "cost_source": adapter.cost_source,
            "cost_is_canonical": adapter.cost_is_canonical,
            "warning": smoke_cfg.get("note"),
        }
        output_path = output_dir / "omnisafe_adapter_smoke.json"
        write_json(output_path, payload)

        manifest = default_manifest(config, humanoid_gym_root)
        manifest["run_name"] = run_name
        manifest["smoke_path"] = relative_to_repo(output_path)
        manifest["smoke"] = payload
        write_json(output_dir / "manifest.json", manifest)
        print(f"Wrote {relative_to_repo(output_path)}")
        print(f"Wrote {relative_to_repo(output_dir / 'manifest.json')}")
        return 0
    except Exception as exc:
        if args.write_failure_artifact:
            output_path = write_failure_artifact(output_dir, config, run_name, command, exc)
            print(f"Wrote failure artifact to {relative_to_repo(output_path)}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
