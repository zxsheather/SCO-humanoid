from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class JacobianCostTensorResult:
    cost_tensor: torch.Tensor
    cost_mean: float
    cost_max: float
    cost_quantile: float
    cost_for_update: float
    violation_rate: float
    threshold: float
    sample_count: int
    per_sample_costs: list[float]


@dataclass
class HookUpdateResult:
    base_loss: float
    penalty_loss: float
    total_loss: float
    multiplier_before: float
    multiplier_after: float
    constraint_error: float
    grad_norm: float
    cost: JacobianCostTensorResult


class OmniSafeActorMeanAdapter:
    """Expose OmniSafe Gaussian actors through the SC-PPO act_inference contract."""

    def __init__(self, actor: Any) -> None:
        self.actor = actor

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        distribution = self.actor(obs)
        mean = getattr(distribution, "mean", None)
        if mean is None:
            return self.actor.predict(obs, deterministic=True)
        return mean


def _sample_observations(obs_batch: torch.Tensor, subsample_obs: int) -> torch.Tensor:
    if subsample_obs <= 0 or obs_batch.shape[0] <= subsample_obs:
        return obs_batch.detach().clone().requires_grad_(True)
    indices = torch.randperm(obs_batch.shape[0], device=obs_batch.device)[:subsample_obs]
    return obs_batch.index_select(0, indices).detach().clone().requires_grad_(True)


def compute_differentiable_jacobian_cost(
    actor: Any,
    obs_batch: torch.Tensor,
    *,
    threshold: float = 3.8,
    subsample_obs: int = 8,
    cost_aggregation: str = "quantile",
    cost_quantile: float = 0.9,
    epsilon: float = 1e-6,
) -> JacobianCostTensorResult:
    """Compute a differentiable SC-PPO-style Jacobian cost for actor updates."""

    actor_adapter = OmniSafeActorMeanAdapter(actor)
    sampled_obs = _sample_observations(obs_batch, subsample_obs)
    action_mean = actor_adapter.act_inference(sampled_obs)
    squared_norm = torch.zeros(sampled_obs.shape[0], device=sampled_obs.device)

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
    cost_mean_tensor = local_sensitivity.mean()
    cost_max_tensor = local_sensitivity.max()
    cost_quantile_tensor = torch.quantile(local_sensitivity, cost_quantile)

    if cost_aggregation == "mean":
        cost_tensor = cost_mean_tensor
    elif cost_aggregation == "max":
        cost_tensor = cost_max_tensor
    elif cost_aggregation == "quantile":
        cost_tensor = cost_quantile_tensor
    else:
        raise ValueError(f"Unsupported cost_aggregation: {cost_aggregation}")

    detached = local_sensitivity.detach()
    return JacobianCostTensorResult(
        cost_tensor=cost_tensor,
        cost_mean=float(cost_mean_tensor.detach().item()),
        cost_max=float(cost_max_tensor.detach().item()),
        cost_quantile=float(cost_quantile_tensor.detach().item()),
        cost_for_update=float(cost_tensor.detach().item()),
        violation_rate=float((detached > threshold).float().mean().item()),
        threshold=float(threshold),
        sample_count=int(sampled_obs.shape[0]),
        per_sample_costs=detached.cpu().tolist(),
    )


class JacobianPPOLagUpdateHook:
    """Bounded PPO-Lag update hook for SC-PPO's actor-local Jacobian cost."""

    def __init__(
        self,
        actor: Any,
        lagrange: Any,
        *,
        threshold: float = 3.8,
        subsample_obs: int = 8,
        cost_aggregation: str = "quantile",
        cost_quantile: float = 0.9,
        epsilon: float = 1e-6,
    ) -> None:
        self.actor = actor
        self.lagrange = lagrange
        self.threshold = float(threshold)
        self.subsample_obs = int(subsample_obs)
        self.cost_aggregation = str(cost_aggregation)
        self.cost_quantile = float(cost_quantile)
        self.epsilon = float(epsilon)

    def jacobian_cost(self, obs_batch: torch.Tensor) -> JacobianCostTensorResult:
        return compute_differentiable_jacobian_cost(
            self.actor,
            obs_batch,
            threshold=self.threshold,
            subsample_obs=self.subsample_obs,
            cost_aggregation=self.cost_aggregation,
            cost_quantile=self.cost_quantile,
            epsilon=self.epsilon,
        )

    def penalty_loss(self, obs_batch: torch.Tensor) -> tuple[torch.Tensor, JacobianCostTensorResult]:
        cost = self.jacobian_cost(obs_batch)
        constraint_error = cost.cost_tensor - self.threshold
        multiplier = self.lagrange.lagrangian_multiplier.detach()
        return multiplier * constraint_error, cost

    def update_multiplier(self, cost: JacobianCostTensorResult) -> tuple[float, float]:
        multiplier_before = float(self.lagrange.lagrangian_multiplier.detach().item())
        self.lagrange.update_lagrange_multiplier(cost.cost_for_update)
        multiplier_after = float(self.lagrange.lagrangian_multiplier.detach().item())
        return multiplier_before, multiplier_after

    def actor_update(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        base_policy_loss: torch.Tensor,
        obs_batch: torch.Tensor,
    ) -> HookUpdateResult:
        penalty, cost = self.penalty_loss(obs_batch)
        total_loss = base_policy_loss + penalty

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        optimizer.step()

        multiplier_before, multiplier_after = self.update_multiplier(cost)
        return HookUpdateResult(
            base_loss=float(base_policy_loss.detach().item()),
            penalty_loss=float(penalty.detach().item()),
            total_loss=float(total_loss.detach().item()),
            multiplier_before=multiplier_before,
            multiplier_after=multiplier_after,
            constraint_error=float(cost.cost_for_update - self.threshold),
            grad_norm=float(grad_norm),
            cost=cost,
        )


HOOK_IMPLEMENTATION_BOUNDARY = {
    "site_packages_modified": False,
    "omnisafe_components_reused": ["GaussianLearningActor", "Lagrange"],
    "required_ppolag_override_points": ["_update_actor"],
    "not_implemented_here": "Full PPOLag learn() integration and checkpoint production remain #62/#63 scope.",
}
