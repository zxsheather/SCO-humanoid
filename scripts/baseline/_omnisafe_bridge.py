"""
OmniSafe PPO-Lag bridge: Lagrange multiplier adapter for SC-PPO.

This module provides a faithful bridge between OmniSafe's Lagrange multiplier
class and SC-PPO's Jacobian local-sensitivity cost, satisfying the #61
acceptance criteria without an invasive algorithm fork.

Key finding (documented in #59 feasibility audit):
  OmniSafe PPO-Lag expects per-step environment-side costs, but the SC-PPO
  Jacobian/local-sensitivity cost is computed inside the PPO update from the
  actor's weights. Full integration into OmniSafe's rollout→buffer→update
  pipeline is structurally infeasible without replacing most of PPOLag.

  However, OmniSafe's `Lagrange` multiplier class can be used as a standalone
  module. Its update rule (gradient ascent on λ) is mathematically equivalent
  to SC-PPO's plain dual ascent mode. The value of this bridge is therefore:

  1. Proving OmniSafe's Lagrange class is importable and functional.
  2. Demonstrating equivalence to the existing plain-dual comparison.
  3. Providing a faithful cost bridge (same Jacobian cost, OmniSafe multiplier).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from omnisafe.common.lagrange import Lagrange as OmniSafeLagrange

logger = logging.getLogger(__name__)


# ── Bridge result dataclass ──────────────────────────────────────

@dataclass
class OmniSafeBridgeResult:
    """Result of one OmniSafe Lagrange multiplier update step."""

    multiplier: float
    multiplier_before: float
    lambda_loss: float
    cost_update: float
    threshold: float
    constraint_error: float


# ── Lagrange adapter ─────────────────────────────────────────────

def _create_lagrange(
    cost_limit: float,
    lagrangian_multiplier_init: float = 0.5,
    lambda_lr: float = 0.01,
    lambda_optimizer: str = "SGD",
    lagrangian_upper_bound: Optional[float] = 5.0,
) -> OmniSafeLagrange:
    """Create an OmniSafe Lagrange multiplier with SC-PPO-compatible defaults.

    Args:
        cost_limit: Same semantics as SC-PPO's ``threshold``.
        lagrangian_multiplier_init: Same as SC-PPO's ``lambda_init``.
        lambda_lr: Learning rate for multiplier (≈ SC-PPO ``eta``).
        lambda_optimizer: Optimizer name (``"SGD"`` → plain dual ascent;
            ``"Adam"`` → adaptive).
        lagrangian_upper_bound: Upper clamp bound (≈ SC-PPO ``lambda_max``).

    Returns:
        Configured OmniSafe Lagrange instance.
    """
    return OmniSafeLagrange(
        cost_limit=cost_limit,
        lagrangian_multiplier_init=lagrangian_multiplier_init,
        lambda_lr=lambda_lr,
        lambda_optimizer=lambda_optimizer,
        lagrangian_upper_bound=lagrangian_upper_bound,
    )


def update_omnisafe_multiplier(
    lagrange: OmniSafeLagrange,
    cost_update: float,
    threshold: float,
) -> OmniSafeBridgeResult:
    """Execute one OmniSafe Lagrange multiplier update step.

    Args:
        lagrange: An initialized OmniSafe ``Lagrange`` instance.
        cost_update: The current batch cost value ``J_C`` (same as SC-PPO's
            ``cost_for_update``).
        threshold: The sensitivity threshold ``d``.

    Returns:
        ``OmniSafeBridgeResult`` with pre/post multiplier values and diagnostics.
    """
    multiplier_before = float(lagrange.lagrangian_multiplier.item())

    # Update λ via gradient ascent:  λ ← λ + lr * (J_C - d), then clamp to [0, upper]
    lagrange.update_lagrange_multiplier(cost_update)

    multiplier_after = float(lagrange.lagrangian_multiplier.item())
    constraint_error = cost_update - threshold

    return OmniSafeBridgeResult(
        multiplier=multiplier_after,
        multiplier_before=multiplier_before,
        lambda_loss=float(-multiplier_before * constraint_error),
        cost_update=cost_update,
        threshold=threshold,
        constraint_error=constraint_error,
    )


# ── Jacobian cost computation (mirrors SC-PPO _local_sensitivity_metrics) ──

def compute_jacobian_cost(
    actor_critic: Any,
    obs_batch: torch.Tensor,
    *,
    threshold: float = 3.8,
    subsample_obs: int = 8,
    cost_aggregation: str = "quantile",
    cost_quantile: float = 0.9,
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """Compute the policy Jacobian Frobenius-norm cost on an observation batch.

    Mirrors ``SCPPO._local_sensitivity_metrics()`` in
    ``.external/humanoid-gym/humanoid/algo/ppo/sc_ppo.py``.

    Args:
        actor_critic: The ``ActorCritic`` module (must expose ``act_inference``).
        obs_batch: Observation tensor, shape ``[N, obs_dim]``.
        threshold: Sensitivity threshold for violation-rate diagnostics.
        subsample_obs: Maximum observations to subsample for cost (≤ N).
        cost_aggregation: ``"mean"``, ``"max"``, or ``"quantile"``.
        cost_quantile: Quantile for aggregation (only if ``quantile``).
        epsilon: Numerical floor for sqrt.

    Returns:
        Dict with keys: ``cost_mean``, ``cost_max``, ``cost_quantile``,
        ``cost_for_update``, ``violation_rate``, ``sample_count``,
        ``per_sample_costs``.
    """
    device = obs_batch.device
    N = obs_batch.shape[0]
    n_sample = min(subsample_obs, N) if subsample_obs > 0 else N

    if n_sample < N:
        indices = torch.randperm(N, device=device)[:n_sample]
        sampled_obs = obs_batch[indices].detach().clone().requires_grad_(True)
    else:
        sampled_obs = obs_batch.detach().clone().requires_grad_(True)

    # Forward pass: deterministic action mean
    action_mean = actor_critic.act_inference(sampled_obs)
    num_actions = action_mean.shape[1]

    # Per-action-dimension Jacobian → squared norm
    squared_norm = torch.zeros(sampled_obs.shape[0], device=device)
    for action_idx in range(num_actions):
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
        squared_norm += torch.sum(torch.square(grads), dim=1)

    local_sensitivity = torch.sqrt(torch.clamp(squared_norm, min=epsilon))

    cost_mean = float(local_sensitivity.mean().item())
    cost_max = float(local_sensitivity.max().item())
    cost_quantile_val = float(
        torch.quantile(local_sensitivity, cost_quantile).item()
    )

    if cost_aggregation == "mean":
        cost_for_update = cost_mean
    elif cost_aggregation == "max":
        cost_for_update = cost_max
    elif cost_aggregation == "quantile":
        cost_for_update = cost_quantile_val
    else:
        raise ValueError(f"Unsupported cost_aggregation: {cost_aggregation}")

    violation_rate = float((local_sensitivity > threshold).float().mean().item())

    return {
        "cost_mean": cost_mean,
        "cost_max": cost_max,
        "cost_quantile": cost_quantile_val,
        "cost_for_update": cost_for_update,
        "violation_rate": violation_rate,
        "threshold": float(threshold),
        "sample_count": int(sampled_obs.shape[0]),
        "per_sample_costs": local_sensitivity.detach().cpu().tolist(),
    }


# ── Smoke test entrypoint ────────────────────────────────────────

def run_bridge_smoke(
    actor_critic: Any,
    obs_batch: torch.Tensor,
    threshold: float = 3.8,
    multiplier_init: float = 0.5,
    lambda_lr: float = 0.01,
    lambda_optimizer: str = "SGD",
    lagrangian_upper_bound: float = 5.0,
    subsample_obs: int = 8,
    cost_aggregation: str = "quantile",
    cost_quantile: float = 0.9,
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """Run a single-cycle smoke test of the full bridge.

    1. Compute Jacobian cost on the provided observation batch.
    2. Create an OmniSafe Lagrange instance.
    3. Execute one multiplier update step.
    4. Return all diagnostics.

    Args:
        actor_critic: ``ActorCritic`` module.
        obs_batch: Observation tensor.
        threshold: Cost threshold (≈ SC-PPO d).
        multiplier_init: Initial λ value.
        lambda_lr: Multiplier learning rate.
        lambda_optimizer: ``"SGD"`` or ``"Adam"``.
        lagrangian_upper_bound: Upper clamp bound for λ.
        subsample_obs: Maximum observations to sample for the Jacobian cost.
        cost_aggregation: ``"mean"``, ``"max"``, or ``"quantile"``.
        cost_quantile: Quantile for the update statistic.
        epsilon: Numerical floor used by the local-sensitivity norm.

    Returns:
        Dict with cost metrics, multiplier diagnostics, and bridge status.
    """
    cost_result = compute_jacobian_cost(
        actor_critic,
        obs_batch,
        threshold=threshold,
        subsample_obs=subsample_obs,
        cost_aggregation=cost_aggregation,
        cost_quantile=cost_quantile,
        epsilon=epsilon,
    )

    lagrange = _create_lagrange(
        cost_limit=threshold,
        lagrangian_multiplier_init=multiplier_init,
        lambda_lr=lambda_lr,
        lambda_optimizer=lambda_optimizer,
        lagrangian_upper_bound=lagrangian_upper_bound,
    )

    update_result = update_omnisafe_multiplier(
        lagrange,
        cost_result["cost_for_update"],
        threshold,
    )

    return {
        "bridge_status": "smoke_ok",
        "cost_is_canonical": True,
        "cost_source": "policy_local_sensitivity_jacobian",
        "cost": cost_result,
        "update": {
            "multiplier": update_result.multiplier,
            "multiplier_before": update_result.multiplier_before,
            "lambda_loss": update_result.lambda_loss,
            "constraint_error": update_result.constraint_error,
        },
    }
