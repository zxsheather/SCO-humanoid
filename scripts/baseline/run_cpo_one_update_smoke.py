#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
from run_cpo_autograd_hvp_smoke import (  # noqa: E402
    activation_from_name,
    compute_jacobian_cost_tensor,
    flatten_tensors,
    gradient_summary,
    load_actor_critic_class,
    memory_snapshot,
    policy_named_parameters,
    reset_peak_memory,
    select_device,
    synchronize,
    tensor_summary,
    zero_like_param,
)


DEFAULT_CONFIG = "configs/methods/cpo_one_update_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local CPO-style update smoke with the SC-PPO Jacobian cost.")
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


def flat_params(named_params: list[tuple[str, Any]]):
    return flatten_tensors([param.detach().reshape(-1) for _, param in named_params])


def set_flat_params(named_params: list[tuple[str, Any]], vector) -> None:
    offset = 0
    for _, param in named_params:
        numel = param.numel()
        param.data.copy_(vector[offset : offset + numel].view_as(param))
        offset += numel
    if offset != vector.numel():
        raise ValueError(f"Flat vector length mismatch: consumed {offset}, got {vector.numel()}")


def flat_grad(named_params: list[tuple[str, Any]], grads: tuple[Any, ...]):
    return flatten_tensors(
        [grad.detach() if grad is not None else zero_like_param(param) for (_, param), grad in zip(named_params, grads)]
    )


def distribution_stats(actor_critic, obs_batch):
    import torch
    from torch.distributions import Normal

    mean = actor_critic.act_inference(obs_batch)
    std = (actor_critic.std * actor_critic.action_std_scale).expand_as(mean)
    return Normal(mean, std)


def surrogate_objective(actor_critic, obs_batch, actions, old_log_prob, advantages):
    import torch

    dist = distribution_stats(actor_critic, obs_batch)
    log_prob = dist.log_prob(actions).sum(dim=-1)
    ratio = torch.exp(log_prob - old_log_prob)
    return (ratio * advantages).mean()


def mean_kl_from_old(actor_critic, obs_batch, old_mean, old_std):
    from torch.distributions import Normal, kl_divergence

    new_dist = distribution_stats(actor_critic, obs_batch)
    old_dist = Normal(old_mean, old_std)
    return kl_divergence(old_dist, new_dist).sum(dim=-1).mean()


def compute_cost_scalar(actor_critic, obs_batch, smoke_cfg: dict[str, Any], seed: int):
    import torch

    # Make line-search cost checks comparable by using the same subsample each time.
    torch.manual_seed(seed)
    return compute_jacobian_cost_tensor(
        actor_critic,
        obs_batch,
        subsample_obs=int(smoke_cfg["subsample_obs"]),
        threshold=float(smoke_cfg["threshold"]),
        cost_aggregation=str(smoke_cfg["cost_aggregation"]),
        cost_quantile=float(smoke_cfg["cost_quantile"]),
        epsilon=float(smoke_cfg["epsilon"]),
    )


def conjugate_gradient(fvp_fn, b, *, max_iters: int, residual_tol: float) -> tuple[Any, dict[str, Any]]:
    import torch

    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)
    residuals = [float(torch.sqrt(torch.clamp(rdotr, min=0.0)).item())]
    finite = bool(torch.isfinite(r).all().item())

    completed_iters = 0
    for idx in range(max_iters):
        Ap = fvp_fn(p)
        finite = finite and bool(torch.isfinite(Ap).all().item())
        denom = torch.dot(p, Ap)
        if abs(float(denom.detach().item())) < 1e-20:
            break
        alpha = rdotr / denom
        x = x + alpha * p
        r = r - alpha * Ap
        new_rdotr = torch.dot(r, r)
        residuals.append(float(torch.sqrt(torch.clamp(new_rdotr, min=0.0)).item()))
        completed_iters = idx + 1
        if float(new_rdotr.detach().item()) < residual_tol:
            rdotr = new_rdotr
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr

    return x, {
        "iterations": completed_iters,
        "initial_residual": residuals[0],
        "final_residual": residuals[-1],
        "residuals": residuals,
        "finite": bool(finite),
    }


def make_fvp(actor_critic, obs_batch, named_params, old_mean, old_std, damping: float):
    import torch

    raw_params = [param for _, param in named_params]

    def fvp(vector):
        actor_critic.zero_grad(set_to_none=True)
        kl = mean_kl_from_old(actor_critic, obs_batch, old_mean, old_std)
        kl_grads = torch.autograd.grad(kl, raw_params, create_graph=True, allow_unused=True)
        flat_kl_grad = flatten_tensors(
            [grad if grad is not None else zero_like_param(param) for grad, param in zip(kl_grads, raw_params)]
        )
        grad_vector_dot = torch.dot(flat_kl_grad, vector)
        hvp_grads = torch.autograd.grad(grad_vector_dot, raw_params, allow_unused=True)
        flat_hvp = flatten_tensors(
            [grad if grad is not None else zero_like_param(param) for grad, param in zip(hvp_grads, raw_params)]
        )
        return flat_hvp + float(damping) * vector

    return fvp


def solve_dual_step(g, b, x_g, x_b, fvp_fn, *, constraint_value: float, max_kl: float):
    import torch

    eps = 1e-12
    q = float(torch.dot(g, x_g).detach().item())
    r = float(torch.dot(g, x_b).detach().item())
    s = float(torch.dot(b, x_b).detach().item())
    lambda_star = math.sqrt(max(q, eps) / max(2.0 * max_kl, eps))
    reward_step = x_g / max(lambda_star, eps)
    linearized_constraint = float((torch.dot(b, reward_step).detach().item()) + constraint_value)

    if constraint_value <= 0.0 and linearized_constraint <= 0.0:
        return reward_step, {
            "case": "reward_only_feasible",
            "lambda": lambda_star,
            "nu": 0.0,
            "q": q,
            "r": r,
            "s": s,
            "linearized_constraint": linearized_constraint,
            "trust_region_quadratic": float(0.5 * torch.dot(reward_step, fvp_fn(reward_step)).detach().item()),
        }

    nu = max(0.0, (linearized_constraint / max(s, eps)))
    raw_step = x_g - nu * x_b
    raw_quad = float(torch.dot(raw_step, fvp_fn(raw_step)).detach().item())
    if raw_quad <= eps:
        raw_step = -x_b
        raw_quad = float(torch.dot(raw_step, fvp_fn(raw_step)).detach().item())
    scale = math.sqrt(max(2.0 * max_kl, eps) / max(raw_quad, eps))
    step = scale * raw_step
    return step, {
        "case": "projected_constraint_fallback",
        "lambda": 1.0 / max(scale, eps),
        "nu": nu,
        "q": q,
        "r": r,
        "s": s,
        "linearized_constraint": float((torch.dot(b, step).detach().item()) + constraint_value),
        "trust_region_quadratic": float(0.5 * torch.dot(step, fvp_fn(step)).detach().item()),
    }


def evaluate_candidate(
    actor_critic,
    obs_batch,
    actions,
    old_log_prob,
    advantages,
    old_mean,
    old_std,
    smoke_cfg,
    seed: int,
) -> dict[str, Any]:
    import torch

    surrogate = surrogate_objective(actor_critic, obs_batch, actions, old_log_prob, advantages)
    kl = mean_kl_from_old(actor_critic, obs_batch, old_mean, old_std)
    cost_tensor, cost_payload = compute_cost_scalar(actor_critic, obs_batch, smoke_cfg, seed)
    finite = bool(torch.isfinite(surrogate).all().item() and torch.isfinite(kl).all().item() and torch.isfinite(cost_tensor).all().item())
    return {
        "surrogate": float(surrogate.detach().item()),
        "kl": float(kl.detach().item()),
        "cost": cost_payload,
        "constraint_value": float(cost_payload["cost_for_update"] - float(smoke_cfg["threshold"])),
        "finite": finite,
    }


def run_line_search(
    actor_critic,
    named_params,
    step,
    old_flat,
    *,
    obs_batch,
    actions,
    old_log_prob,
    advantages,
    old_mean,
    old_std,
    old_eval,
    smoke_cfg,
    seed: int,
) -> dict[str, Any]:
    max_kl = float(smoke_cfg["max_kl"])
    constraint_tol = float(smoke_cfg["constraint_tolerance"])
    surrogate_tol = float(smoke_cfg["surrogate_tolerance"])
    shrink = float(smoke_cfg["line_search_shrink"])
    max_backtracks = int(smoke_cfg["line_search_backtracks"])
    attempts: list[dict[str, Any]] = []

    accepted_attempt: dict[str, Any] | None = None
    for idx in range(max_backtracks):
        fraction = shrink**idx
        set_flat_params(named_params, old_flat + fraction * step)
        candidate = evaluate_candidate(
            actor_critic,
            obs_batch,
            actions,
            old_log_prob,
            advantages,
            old_mean,
            old_std,
            smoke_cfg,
            seed,
        )
        candidate.update(
            {
                "backtrack": idx,
                "step_fraction": fraction,
                "accepted": bool(
                    candidate["finite"]
                    and candidate["kl"] <= max_kl
                    and candidate["constraint_value"] <= constraint_tol
                    and candidate["surrogate"] >= old_eval["surrogate"] - surrogate_tol
                ),
            }
        )
        attempts.append(candidate)
        if candidate["accepted"]:
            accepted_attempt = candidate
            break

    set_flat_params(named_params, old_flat)
    return {
        "accepted": accepted_attempt is not None,
        "accepted_attempt": accepted_attempt,
        "attempts": attempts,
        "parameters_restored": True,
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
    named_params = policy_named_parameters(actor_critic)
    raw_params = [param for _, param in named_params]

    old_dist = distribution_stats(actor_critic, obs)
    actions = old_dist.sample().detach()
    old_log_prob = old_dist.log_prob(actions).sum(dim=-1).detach()
    old_mean = old_dist.mean.detach()
    old_std = old_dist.stddev.detach()
    advantages = torch.linspace(-1.0, 1.0, batch_size, device=device)
    advantages = (advantages - advantages.mean()) / torch.clamp(advantages.std(unbiased=False), min=1e-6)

    if device.startswith("cuda"):
        # Exclude first-use CUDA/autograd setup from the measured one-update timing.
        cost_tensor, _ = compute_cost_scalar(actor_critic, obs, smoke_cfg, seed)
        torch.autograd.grad(cost_tensor, raw_params, allow_unused=True)
        warmup_fvp = make_fvp(actor_critic, obs, named_params, old_mean, old_std, float(smoke_cfg["hvp_damping"]))
        warmup_fvp(torch.zeros(sum(param.numel() for _, param in named_params), device=device))
        torch.cuda.empty_cache()

    reset_peak_memory(device)
    synchronize(device)
    start = time.perf_counter()

    old_flat = flat_params(named_params)
    old_eval = evaluate_candidate(
        actor_critic,
        obs,
        actions,
        old_log_prob,
        advantages,
        old_mean,
        old_std,
        smoke_cfg,
        seed,
    )

    reward_objective = surrogate_objective(actor_critic, obs, actions, old_log_prob, advantages)
    reward_grads = torch.autograd.grad(reward_objective, raw_params, allow_unused=True)
    g = flat_grad(named_params, reward_grads)

    cost_tensor, cost_payload = compute_cost_scalar(actor_critic, obs, smoke_cfg, seed)
    cost_grads = torch.autograd.grad(cost_tensor, raw_params, allow_unused=True)
    b = flat_grad(named_params, cost_grads)
    constraint_value = float(cost_payload["cost_for_update"] - float(smoke_cfg["threshold"]))

    fvp_fn = make_fvp(actor_critic, obs, named_params, old_mean, old_std, float(smoke_cfg["hvp_damping"]))
    x_g, cg_reward = conjugate_gradient(
        fvp_fn,
        g,
        max_iters=int(smoke_cfg["cg_iters"]),
        residual_tol=float(smoke_cfg["cg_residual_tol"]),
    )
    x_b, cg_cost = conjugate_gradient(
        fvp_fn,
        b,
        max_iters=int(smoke_cfg["cg_iters"]),
        residual_tol=float(smoke_cfg["cg_residual_tol"]),
    )
    step, dual = solve_dual_step(
        g,
        b,
        x_g,
        x_b,
        fvp_fn,
        constraint_value=constraint_value,
        max_kl=float(smoke_cfg["max_kl"]),
    )
    line_search = run_line_search(
        actor_critic,
        named_params,
        step,
        old_flat,
        obs_batch=obs,
        actions=actions,
        old_log_prob=old_log_prob,
        advantages=advantages,
        old_mean=old_mean,
        old_std=old_std,
        old_eval=old_eval,
        smoke_cfg=smoke_cfg,
        seed=seed,
    )

    synchronize(device)
    runtime = {"wall_time_s": time.perf_counter() - start, **memory_snapshot(device)}

    finite = bool(
        old_eval["finite"]
        and torch.isfinite(g).all().item()
        and torch.isfinite(b).all().item()
        and torch.isfinite(step).all().item()
        and cg_reward["finite"]
        and cg_cost["finite"]
    )
    status = "complete" if finite else "failed"
    recommendation = "proceed_to_bounded_one_seed_diagnostic" if finite and line_search["accepted"] else "stop_before_training"

    payload = {
        "status": status,
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
            "policy_param_tensors": len(named_params),
            "policy_param_numel": sum(int(param.numel()) for _, param in named_params),
        },
        "observation": tensor_summary(obs),
        "old_eval": old_eval,
        "reward_gradient": {
            "surrogate": float(reward_objective.detach().item()),
            "summary": gradient_summary(named_params, reward_grads),
            "flat_norm": float(torch.linalg.vector_norm(g.detach()).item()),
        },
        "cost_gradient": {
            "cost": cost_payload,
            "constraint_value": constraint_value,
            "summary": gradient_summary(named_params, cost_grads),
            "flat_norm": float(torch.linalg.vector_norm(b.detach()).item()),
        },
        "conjugate_gradient": {
            "reward": cg_reward,
            "cost": cg_cost,
        },
        "dual_solve": dual,
        "step": {
            "flat_norm": float(torch.linalg.vector_norm(step.detach()).item()),
            "finite": bool(torch.isfinite(step).all().item()),
        },
        "line_search": line_search,
        "runtime": runtime,
        "finite": finite,
        "recommendation": recommendation,
        "boundary": "One local CPO-style update attempt only; no rollout, no training, no official CPO parity claim.",
    }

    output_path = output_dir / "cpo_one_update_smoke.json"
    write_json(output_path, payload)
    print(relative_to_repo(output_path))
    print(recommendation)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
