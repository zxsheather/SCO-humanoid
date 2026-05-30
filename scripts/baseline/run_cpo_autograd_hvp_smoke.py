#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    artifact_dir,
    ensure_directory,
    load_config,
    relative_to_repo,
    resolve_humanoid_gym_root,
    write_json,
)


DEFAULT_CONFIG = "configs/methods/cpo_autograd_hvp_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test CPO autograd/HVP compatibility for the SC-PPO Jacobian cost.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--humanoid-gym-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def rendered_invocation(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), f"--config={args.config}"]
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    if args.run_name:
        command.append(f"--run-name={args.run_name}")
    if args.artifacts_root:
        command.append(f"--artifacts-root={args.artifacts_root}")
    if args.device:
        command.append(f"--device={args.device}")
    if args.seed is not None:
        command.append(f"--seed={args.seed}")
    if args.batch_size is not None:
        command.append(f"--batch-size={args.batch_size}")
    return command


def load_actor_critic_class(humanoid_gym_root: Path) -> type[Any]:
    module_path = humanoid_gym_root / "humanoid" / "algo" / "ppo" / "actor_critic.py"
    if not module_path.exists():
        raise FileNotFoundError(f"ActorCritic source not found: {module_path}")
    spec = importlib.util.spec_from_file_location("cpo_smoke_actor_critic", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ActorCritic from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ActorCritic


def activation_from_name(name: str):
    import torch.nn as nn

    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[key]()


def select_device(raw_device: str) -> str:
    import torch

    if raw_device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if raw_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {raw_device}, but CUDA is not available.")
    return raw_device


def synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def reset_peak_memory(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def memory_snapshot(device: str) -> dict[str, Any]:
    import torch

    # Linux reports ru_maxrss in KiB.
    max_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    payload: dict[str, Any] = {"process_max_rss_mb": max_rss_mb}
    if device.startswith("cuda"):
        cuda_device = torch.device(device)
        payload.update(
            {
                "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated(cuda_device)) / (1024.0**2),
                "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved(cuda_device)) / (1024.0**2),
            }
        )
    else:
        payload.update({"cuda_peak_allocated_mb": None, "cuda_peak_reserved_mb": None})
    return payload


def timed_step(device: str, fn) -> tuple[Any, dict[str, Any]]:
    reset_peak_memory(device)
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    metrics = {"wall_time_s": time.perf_counter() - start, **memory_snapshot(device)}
    return result, metrics


def tensor_summary(value) -> dict[str, Any]:
    import torch

    detached = value.detach()
    finite = bool(torch.isfinite(detached).all().item()) if detached.numel() else True
    payload: dict[str, Any] = {
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite": finite,
        "numel": int(detached.numel()),
    }
    if detached.numel() and detached.is_floating_point():
        payload.update(
            {
                "mean": float(detached.mean().item()),
                "min": float(detached.min().item()),
                "max": float(detached.max().item()),
            }
        )
    return payload


def policy_named_parameters(actor_critic) -> list[tuple[str, Any]]:
    return [
        (name, param)
        for name, param in actor_critic.named_parameters()
        if name == "std" or name.startswith("actor.")
    ]


def gradient_summary(named_params: list[tuple[str, Any]], grads: tuple[Any, ...]) -> dict[str, Any]:
    import torch

    total_tensors = len(named_params)
    none_names: list[str] = []
    zero_names: list[str] = []
    nonzero_names: list[str] = []
    finite = True
    sq_norm = 0.0
    covered_numel = 0
    total_numel = 0

    for (name, param), grad in zip(named_params, grads):
        total_numel += int(param.numel())
        if grad is None:
            none_names.append(name)
            continue
        detached = grad.detach()
        covered_numel += int(detached.numel())
        finite = finite and bool(torch.isfinite(detached).all().item())
        grad_norm = float(torch.linalg.vector_norm(detached).item())
        sq_norm += grad_norm**2
        if grad_norm > 0.0:
            nonzero_names.append(name)
        else:
            zero_names.append(name)

    return {
        "finite": bool(finite),
        "global_norm": math.sqrt(sq_norm),
        "total_param_tensors": total_tensors,
        "none_param_tensors": len(none_names),
        "zero_param_tensors": len(zero_names),
        "nonzero_param_tensors": len(nonzero_names),
        "covered_numel": covered_numel,
        "total_numel": total_numel,
        "none_param_names": none_names,
        "zero_param_names": zero_names,
        "nonzero_param_names_sample": nonzero_names[:8],
    }


def zero_like_param(param):
    import torch

    return torch.zeros_like(param, memory_format=torch.preserve_format)


def flatten_tensors(tensors: list[Any]) -> Any:
    import torch

    if not tensors:
        return torch.empty(0)
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def compute_jacobian_cost_tensor(
    actor_critic,
    obs_batch,
    *,
    subsample_obs: int,
    threshold: float,
    cost_aggregation: str,
    cost_quantile: float,
    epsilon: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if subsample_obs <= 0 or obs_batch.shape[0] <= subsample_obs:
        sampled_obs = obs_batch.detach().clone().requires_grad_(True)
    else:
        indices = torch.randperm(obs_batch.shape[0], device=obs_batch.device)[:subsample_obs]
        sampled_obs = obs_batch.index_select(0, indices).detach().clone().requires_grad_(True)

    action_mean = actor_critic.act_inference(sampled_obs)
    squared_norm = torch.zeros(sampled_obs.shape[0], device=obs_batch.device)
    for action_idx in range(action_mean.shape[1]):
        grad_outputs = torch.zeros_like(action_mean)
        grad_outputs[:, action_idx] = 1.0
        grads = torch.autograd.grad(
            outputs=action_mean,
            inputs=sampled_obs,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        squared_norm = squared_norm + torch.sum(torch.square(grads), dim=1)

    local_sensitivity = torch.sqrt(torch.clamp(squared_norm, min=epsilon))
    cost_mean = local_sensitivity.mean()
    cost_max = local_sensitivity.max()
    cost_quantile_tensor = torch.quantile(local_sensitivity, cost_quantile)

    if cost_aggregation == "mean":
        cost_for_update = cost_mean
    elif cost_aggregation == "max":
        cost_for_update = cost_max
    elif cost_aggregation == "quantile":
        cost_for_update = cost_quantile_tensor
    else:
        raise ValueError(f"Unsupported cost_aggregation: {cost_aggregation}")

    detached = local_sensitivity.detach()
    return cost_for_update, {
        "cost_mean": float(cost_mean.detach().item()),
        "cost_max": float(cost_max.detach().item()),
        "cost_quantile": float(cost_quantile_tensor.detach().item()),
        "cost_for_update": float(cost_for_update.detach().item()),
        "violation_rate": float((detached > threshold).float().mean().item()),
        "threshold": float(threshold),
        "sample_count": int(sampled_obs.shape[0]),
        "action_dim": int(action_mean.shape[1]),
        "per_sample_costs": detached.cpu().tolist(),
    }


def run_cost_gradient_smoke(actor_critic, obs_batch, policy_params, smoke_cfg: dict[str, Any], subsample_obs: int):
    import torch

    actor_critic.zero_grad(set_to_none=True)
    cost_tensor, cost_payload = compute_jacobian_cost_tensor(
        actor_critic,
        obs_batch,
        subsample_obs=subsample_obs,
        threshold=float(smoke_cfg["threshold"]),
        cost_aggregation=str(smoke_cfg["cost_aggregation"]),
        cost_quantile=float(smoke_cfg["cost_quantile"]),
        epsilon=float(smoke_cfg["epsilon"]),
    )
    grads = torch.autograd.grad(cost_tensor, [param for _, param in policy_params], allow_unused=True)
    payload = {
        "subsample_obs": int(subsample_obs),
        "cost": cost_payload,
        "grad_theta_J_C": gradient_summary(policy_params, grads),
    }
    if not payload["grad_theta_J_C"]["finite"]:
        raise RuntimeError(f"Non-finite cost gradient for subsample_obs={subsample_obs}.")
    return payload


def run_kl_hvp_smoke(actor_critic, obs_batch, policy_params):
    import torch
    from torch.distributions import Normal, kl_divergence

    actor_critic.zero_grad(set_to_none=True)
    hvp_obs = obs_batch.detach()
    with torch.no_grad():
        old_mean = actor_critic.act_inference(hvp_obs).detach()
        old_std = (actor_critic.std * actor_critic.action_std_scale).detach().expand_as(old_mean)

    new_mean = actor_critic.act_inference(hvp_obs)
    new_std = (actor_critic.std * actor_critic.action_std_scale).expand_as(new_mean)
    kl = kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1).mean()

    raw_params = [param for _, param in policy_params]
    kl_grads = torch.autograd.grad(kl, raw_params, create_graph=True, allow_unused=True)
    flat_grad_parts = [grad if grad is not None else zero_like_param(param) for grad, param in zip(kl_grads, raw_params)]
    flat_grad = flatten_tensors(flat_grad_parts)
    if flat_grad.numel() == 0:
        raise RuntimeError("KL gradient has zero parameters; cannot compute HVP.")

    vector = torch.linspace(1.0, 2.0, flat_grad.numel(), device=flat_grad.device, dtype=flat_grad.dtype)
    vector = vector / torch.linalg.vector_norm(vector)
    grad_vector_dot = torch.dot(flat_grad, vector)
    hvp_grads = torch.autograd.grad(grad_vector_dot, raw_params, allow_unused=True)
    hvp_parts = [grad if grad is not None else zero_like_param(param) for grad, param in zip(hvp_grads, raw_params)]
    flat_hvp = flatten_tensors(hvp_parts)

    finite = bool(torch.isfinite(flat_hvp).all().item()) if flat_hvp.numel() else False
    if not finite:
        raise RuntimeError("Non-finite KL Fisher-vector product.")
    return {
        "kl_value": float(kl.detach().item()),
        "flat_gradient_numel": int(flat_grad.numel()),
        "probe_vector_numel": int(vector.numel()),
        "hvp_numel": int(flat_hvp.numel()),
        "shape_consistent": bool(flat_hvp.numel() == vector.numel()),
        "finite": finite,
        "hvp_norm": float(torch.linalg.vector_norm(flat_hvp.detach()).item()),
        "hvp_abs_max": float(torch.max(torch.abs(flat_hvp.detach())).item()) if flat_hvp.numel() else 0.0,
        "kl_gradient": gradient_summary(policy_params, kl_grads),
        "hvp_gradient": gradient_summary(policy_params, hvp_grads),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.artifacts_root:
        config["artifacts_root"] = args.artifacts_root
    smoke_cfg = config["smoke"]
    run_name = args.run_name or config["run_name"]
    output_dir = ensure_directory(artifact_dir(config, run_name))
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    command = rendered_invocation(args)

    import torch

    seed = args.seed if args.seed is not None else int(smoke_cfg["seed"])
    torch.manual_seed(seed)
    device = select_device(args.device or str(smoke_cfg["device"]))
    batch_size = args.batch_size if args.batch_size is not None else int(smoke_cfg["batch_size"])

    ActorCritic = load_actor_critic_class(humanoid_gym_root)
    actor_critic = ActorCritic(
        int(smoke_cfg["obs_dim"]),
        int(smoke_cfg["critic_obs_dim"]),
        int(smoke_cfg["num_actions"]),
        actor_hidden_dims=list(smoke_cfg["actor_hidden_dims"]),
        critic_hidden_dims=list(smoke_cfg["critic_hidden_dims"]),
        init_noise_std=float(smoke_cfg["init_noise_std"]),
        activation=activation_from_name(str(smoke_cfg["activation"])),
    ).to(device)
    actor_critic.train()

    obs = torch.randn(batch_size, int(smoke_cfg["obs_dim"]), device=device)
    hvp_batch_size = min(int(smoke_cfg["hvp_batch_size"]), batch_size)
    hvp_obs = obs[:hvp_batch_size]
    policy_params = policy_named_parameters(actor_critic)

    if device.startswith("cuda"):
        # Keep first-use CUDA/autograd initialization out of the reported timings.
        warmup_subsample = int(smoke_cfg["subsample_obs_values"][0])
        run_cost_gradient_smoke(actor_critic, obs, policy_params, smoke_cfg, warmup_subsample)
        run_kl_hvp_smoke(actor_critic, hvp_obs, policy_params)
        torch.cuda.empty_cache()

    cost_results: list[dict[str, Any]] = []
    for subsample_obs in smoke_cfg["subsample_obs_values"]:
        result, metrics = timed_step(
            device,
            lambda subsample_obs=subsample_obs: run_cost_gradient_smoke(
                actor_critic,
                obs,
                policy_params,
                smoke_cfg,
                int(subsample_obs),
            ),
        )
        result["runtime"] = metrics
        cost_results.append(result)

    hvp_result, hvp_metrics = timed_step(device, lambda: run_kl_hvp_smoke(actor_critic, hvp_obs, policy_params))
    hvp_result["runtime"] = hvp_metrics

    proceed = all(item["grad_theta_J_C"]["finite"] for item in cost_results) and hvp_result["finite"]
    recommendation = "proceed_to_one_update_prototype" if proceed else "stop_before_one_update_prototype"

    payload = {
        "status": "complete",
        "issue": smoke_cfg["issue"],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        "run_name": run_name,
        "device": device,
        "seed": seed,
        "observation_source": smoke_cfg["observation_source"],
        "cost_is_canonical": bool(smoke_cfg["cost_is_canonical"]),
        "cost_source": smoke_cfg["cost_source"],
        "model": {
            "actor_class": "Humanoid-Gym ActorCritic",
            "obs_dim": int(smoke_cfg["obs_dim"]),
            "critic_obs_dim": int(smoke_cfg["critic_obs_dim"]),
            "num_actions": int(smoke_cfg["num_actions"]),
            "actor_hidden_dims": list(smoke_cfg["actor_hidden_dims"]),
            "critic_hidden_dims": list(smoke_cfg["critic_hidden_dims"]),
            "policy_param_tensors": len(policy_params),
            "policy_param_numel": sum(int(param.numel()) for _, param in policy_params),
        },
        "observation": tensor_summary(obs),
        "cost_gradient_smokes": cost_results,
        "kl_hvp_smoke": hvp_result,
        "timing_note": "CUDA first-use warm-up is excluded from reported timings when CUDA is used.",
        "recommendation": recommendation,
        "boundary": "Autograd/HVP feasibility only; no rollout, no training, no official CPO parity claim.",
    }

    output_path = output_dir / "cpo_autograd_hvp_smoke.json"
    write_json(output_path, payload)
    print(relative_to_repo(output_path))
    print(recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
