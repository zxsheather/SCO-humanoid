# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY, LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import statistics
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .ppo import PPO


class CPOStylePPO(PPO):
    """Bounded local CPO-style diagnostic for actor-internal Jacobian costs.

    This is intentionally not an official CPO implementation. It keeps the
    SC-PPO policy-local-sensitivity cost and replaces the PPO actor step with a
    local CPO-style constrained update: reward gradient, cost gradient, damped
    KL Fisher-vector products, conjugate gradient, a small dual decision, and
    backtracking line search. The critic remains a standard Adam value update.
    """

    def __init__(self, actor_critic, constraint=None, device="cpu", **kwargs):
        super().__init__(actor_critic, device=device, **kwargs)

        cfg = constraint or {}
        self.constraint_enabled = bool(cfg.get("enabled", True))
        self.constraint_threshold = float(cfg.get("threshold", 3.8))
        self.constraint_subsample_obs = self._parse_constraint_subsample_obs(cfg.get("subsample_obs", 8))
        self.constraint_cost_aggregation = str(cfg.get("cost_aggregation", "quantile")).lower()
        self.constraint_cost_quantile = min(max(float(cfg.get("cost_quantile", 0.9)), 0.0), 1.0)
        self.constraint_epsilon = float(cfg.get("epsilon", 1e-6))

        self.cpo_max_kl = float(self.desired_kl if self.desired_kl is not None else cfg.get("cpo_max_kl", 0.01))
        self.cpo_cg_iters = int(cfg.get("cpo_cg_iters", 5))
        self.cpo_cg_residual_tol = float(cfg.get("cpo_cg_residual_tol", 1e-10))
        self.cpo_hvp_damping = float(cfg.get("cpo_hvp_damping", 0.1))
        self.cpo_line_search_backtracks = int(cfg.get("cpo_line_search_backtracks", 5))
        self.cpo_line_search_shrink = float(cfg.get("cpo_line_search_shrink", 0.5))
        self.cpo_surrogate_tolerance = float(cfg.get("cpo_surrogate_tolerance", 1e-8))
        self.cpo_constraint_tolerance = float(cfg.get("cpo_constraint_tolerance", 1e-6))

        # CPO-style actor updates are manual; Adam is retained for the critic and
        # for checkpoint compatibility with the runner's save/load path.
        self.optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=self.learning_rate)

        self.constraint_trace = []
        self.latest_stats = {}

    def _parse_constraint_subsample_obs(self, raw_value):
        if raw_value is None:
            return 0
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"all", "full", "full_batch", "full-batch"}:
                return 0
            raw_value = normalized
        parsed = int(raw_value)
        return max(parsed, 0)

    def _policy_named_parameters(self):
        return [
            (name, param)
            for name, param in self.actor_critic.named_parameters()
            if name == "std" or name.startswith("actor.")
        ]

    def _flat_params(self, named_params):
        return torch.cat([param.detach().reshape(-1) for _, param in named_params])

    def _set_flat_params(self, named_params, vector):
        offset = 0
        for _, param in named_params:
            numel = param.numel()
            param.data.copy_(vector[offset : offset + numel].view_as(param))
            offset += numel
        if offset != vector.numel():
            raise ValueError(f"Flat vector length mismatch: consumed {offset}, got {vector.numel()}")

    def _flatten_grads(self, named_params, grads):
        pieces = []
        for (_, param), grad in zip(named_params, grads):
            if grad is None:
                pieces.append(torch.zeros_like(param).reshape(-1))
            else:
                pieces.append(grad.reshape(-1))
        return torch.cat(pieces)

    def _gradient_summary(self, named_params, grads):
        total_tensors = len(named_params)
        none_names = []
        zero_names = []
        nonzero_names = []
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
        }

    def _constraint_indices(self, obs_batch):
        if self.constraint_subsample_obs <= 0 or obs_batch.shape[0] <= self.constraint_subsample_obs:
            return None
        return torch.randperm(obs_batch.shape[0], device=obs_batch.device)[: self.constraint_subsample_obs]

    def _constraint_obs_batch(self, obs_batch, indices=None):
        if indices is None:
            return obs_batch
        return obs_batch.index_select(0, indices)

    def _local_sensitivity_metrics(self, obs_batch, indices=None):
        sampled_obs = self._constraint_obs_batch(obs_batch, indices).detach().clone().requires_grad_(True)
        action_mean = self.actor_critic.act_inference(sampled_obs)
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
            squared_norm += torch.sum(torch.square(grads), dim=1)

        local_sensitivity = torch.sqrt(torch.clamp(squared_norm, min=self.constraint_epsilon))
        cost_mean = local_sensitivity.mean()
        cost_max = local_sensitivity.max()
        cost_quantile = torch.quantile(local_sensitivity, self.constraint_cost_quantile)
        cost_for_update = self._aggregate_constraint_cost(cost_mean, cost_max, cost_quantile)
        violation_rate = torch.mean((local_sensitivity > self.constraint_threshold).float())
        return {
            "cost_mean": cost_mean,
            "cost_max": cost_max,
            "cost_quantile": cost_quantile,
            "cost_for_update": cost_for_update,
            "violation_rate": violation_rate,
            "sample_count": local_sensitivity.shape[0],
        }

    def _aggregate_constraint_cost(self, cost_mean, cost_max, cost_quantile):
        if self.constraint_cost_aggregation == "mean":
            return cost_mean
        if self.constraint_cost_aggregation == "max":
            return cost_max
        if self.constraint_cost_aggregation == "quantile":
            return cost_quantile
        raise ValueError(f"Unsupported constraint cost_aggregation: {self.constraint_cost_aggregation}")

    def _surrogate_objective(self, obs_batch, actions_batch, old_actions_log_prob_batch, advantages_batch):
        self.actor_critic.update_distribution(obs_batch)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        objective = (torch.squeeze(advantages_batch) * ratio).mean()
        if self.entropy_coef:
            objective = objective + self.entropy_coef * self.actor_critic.entropy.mean()
        return objective

    def _mean_kl(self, obs_batch, old_mu_batch, old_sigma_batch):
        self.actor_critic.update_distribution(obs_batch)
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
        )
        return torch.mean(kl)

    def _fvp(self, vector, named_params, obs_batch, old_mu_batch, old_sigma_batch):
        raw_params = [param for _, param in named_params]
        self.actor_critic.zero_grad(set_to_none=True)
        kl = self._mean_kl(obs_batch, old_mu_batch, old_sigma_batch)
        kl_grads = torch.autograd.grad(kl, raw_params, create_graph=True, allow_unused=True)
        flat_kl_grad = self._flatten_grads(named_params, kl_grads)
        grad_vector_dot = torch.dot(flat_kl_grad, vector)
        hvp_grads = torch.autograd.grad(grad_vector_dot, raw_params, allow_unused=True)
        flat_hvp = self._flatten_grads(named_params, hvp_grads)
        return flat_hvp + self.cpo_hvp_damping * vector

    def _conjugate_gradient(self, fvp_fn, b):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        residuals = [float(torch.sqrt(torch.clamp(rdotr, min=0.0)).item())]
        finite = bool(torch.isfinite(r).all().item())
        completed_iters = 0

        for idx in range(self.cpo_cg_iters):
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
            if float(new_rdotr.detach().item()) < self.cpo_cg_residual_tol:
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

    def _solve_dual_step(self, g, b, x_g, x_b, fvp_fn, constraint_value):
        eps = 1e-12
        q = float(torch.dot(g, x_g).detach().item())
        r = float(torch.dot(g, x_b).detach().item())
        s = float(torch.dot(b, x_b).detach().item())
        lambda_star = math.sqrt(max(q, eps) / max(2.0 * self.cpo_max_kl, eps))
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

        nu = max(0.0, linearized_constraint / max(s, eps))
        raw_step = x_g - nu * x_b
        raw_quad = float(torch.dot(raw_step, fvp_fn(raw_step)).detach().item())
        if raw_quad <= eps:
            raw_step = -x_b
            raw_quad = float(torch.dot(raw_step, fvp_fn(raw_step)).detach().item())
        scale = math.sqrt(max(2.0 * self.cpo_max_kl, eps) / max(raw_quad, eps))
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

    def _evaluate_candidate(
        self,
        obs_batch,
        actions_batch,
        old_actions_log_prob_batch,
        advantages_batch,
        old_mu_batch,
        old_sigma_batch,
        constraint_indices,
    ):
        surrogate = self._surrogate_objective(obs_batch, actions_batch, old_actions_log_prob_batch, advantages_batch)
        kl = self._mean_kl(obs_batch, old_mu_batch, old_sigma_batch)
        constraint_stats = self._local_sensitivity_metrics(obs_batch, constraint_indices)
        cost_update = constraint_stats["cost_for_update"]
        finite = bool(
            torch.isfinite(surrogate).all().item()
            and torch.isfinite(kl).all().item()
            and torch.isfinite(cost_update).all().item()
        )
        return {
            "surrogate": float(surrogate.detach().item()),
            "kl": float(kl.detach().item()),
            "constraint_value": float((cost_update - self.constraint_threshold).detach().item()),
            "cost_mean": float(constraint_stats["cost_mean"].detach().item()),
            "cost_update": float(cost_update.detach().item()),
            "cost_max": float(constraint_stats["cost_max"].detach().item()),
            "cost_quantile": float(constraint_stats["cost_quantile"].detach().item()),
            "violation_rate": float(constraint_stats["violation_rate"].detach().item()),
            "sample_count": int(constraint_stats["sample_count"]),
            "finite": finite,
        }

    def _line_search(
        self,
        named_params,
        step,
        old_flat,
        obs_batch,
        actions_batch,
        old_actions_log_prob_batch,
        advantages_batch,
        old_mu_batch,
        old_sigma_batch,
        old_eval,
        constraint_indices,
    ):
        attempts = []
        accepted_attempt = None
        for idx in range(self.cpo_line_search_backtracks):
            fraction = self.cpo_line_search_shrink**idx
            self._set_flat_params(named_params, old_flat + fraction * step)
            candidate = self._evaluate_candidate(
                obs_batch,
                actions_batch,
                old_actions_log_prob_batch,
                advantages_batch,
                old_mu_batch,
                old_sigma_batch,
                constraint_indices,
            )
            candidate.update(
                {
                    "backtrack": idx,
                    "step_fraction": fraction,
                    "accepted": bool(
                        candidate["finite"]
                        and candidate["kl"] <= self.cpo_max_kl
                        and candidate["constraint_value"] <= self.cpo_constraint_tolerance
                        and candidate["surrogate"] >= old_eval["surrogate"] - self.cpo_surrogate_tolerance
                    ),
                }
            )
            attempts.append(candidate)
            if candidate["accepted"]:
                accepted_attempt = candidate
                break

        if accepted_attempt is None:
            self._set_flat_params(named_params, old_flat)
        return {
            "accepted": accepted_attempt is not None,
            "accepted_attempt": accepted_attempt,
            "attempts": attempts,
        }

    def _critic_update(self, critic_obs_batch, target_values_batch, returns_batch):
        value_batch = self.actor_critic.evaluate(critic_obs_batch)
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param,
                self.clip_param,
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        self.optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return value_loss

    def _actor_cpo_update(
        self,
        obs_batch,
        actions_batch,
        advantages_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
    ):
        named_params = self._policy_named_parameters()
        raw_params = [param for _, param in named_params]
        old_flat = self._flat_params(named_params)
        constraint_indices = self._constraint_indices(obs_batch)

        old_eval = self._evaluate_candidate(
            obs_batch,
            actions_batch,
            old_actions_log_prob_batch,
            advantages_batch,
            old_mu_batch,
            old_sigma_batch,
            constraint_indices,
        )

        reward_objective = self._surrogate_objective(
            obs_batch,
            actions_batch,
            old_actions_log_prob_batch,
            advantages_batch,
        )
        reward_grads = torch.autograd.grad(reward_objective, raw_params, allow_unused=True)
        g = self._flatten_grads(named_params, reward_grads).detach()

        constraint_stats = self._local_sensitivity_metrics(obs_batch, constraint_indices)
        cost_tensor = constraint_stats["cost_for_update"]
        cost_grads = torch.autograd.grad(cost_tensor, raw_params, allow_unused=True)
        b = self._flatten_grads(named_params, cost_grads).detach()
        constraint_value = float((cost_tensor - self.constraint_threshold).detach().item())

        fvp_fn = lambda vector: self._fvp(vector, named_params, obs_batch, old_mu_batch, old_sigma_batch)
        x_g, cg_reward = self._conjugate_gradient(fvp_fn, g)
        x_b, cg_cost = self._conjugate_gradient(fvp_fn, b)
        step, dual = self._solve_dual_step(g, b, x_g, x_b, fvp_fn, constraint_value)
        line_search = self._line_search(
            named_params,
            step,
            old_flat,
            obs_batch,
            actions_batch,
            old_actions_log_prob_batch,
            advantages_batch,
            old_mu_batch,
            old_sigma_batch,
            old_eval,
            constraint_indices,
        )

        final_eval = line_search["accepted_attempt"] or old_eval
        finite = bool(
            old_eval["finite"]
            and torch.isfinite(g).all().item()
            and torch.isfinite(b).all().item()
            and torch.isfinite(step).all().item()
            and cg_reward["finite"]
            and cg_cost["finite"]
        )
        return {
            "finite": finite,
            "old_eval": old_eval,
            "final_eval": final_eval,
            "reward_gradient": self._gradient_summary(named_params, reward_grads),
            "cost_gradient": self._gradient_summary(named_params, cost_grads),
            "reward_gradient_norm": float(torch.linalg.vector_norm(g).item()),
            "cost_gradient_norm": float(torch.linalg.vector_norm(b).item()),
            "cg_reward": cg_reward,
            "cg_cost": cg_cost,
            "dual": dual,
            "step_norm": float(torch.linalg.vector_norm(step.detach()).item()),
            "line_search": line_search,
        }

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        batch_traces = []

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.device))
                torch.cuda.reset_peak_memory_stats(torch.device(self.device))
            update_start = time.perf_counter()

            value_loss = self._critic_update(critic_obs_batch, target_values_batch, returns_batch)
            actor_result = self._actor_cpo_update(
                obs_batch,
                actions_batch,
                advantages_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            )

            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.device))
                cuda_peak_allocated_mb = float(torch.cuda.max_memory_allocated(torch.device(self.device))) / (1024.0**2)
                cuda_peak_reserved_mb = float(torch.cuda.max_memory_reserved(torch.device(self.device))) / (1024.0**2)
            else:
                cuda_peak_allocated_mb = None
                cuda_peak_reserved_mb = None
            update_wall_time_s = time.perf_counter() - update_start

            final_eval = actor_result["final_eval"]
            trace_entry = {
                "iteration": len(self.constraint_trace),
                "constraint_effective_mode": "local_cpo_style",
                "constraint_penalty_mode": "linearized_cpo_trust_region",
                "constraint_update_error_mode": "jacobian_minus_threshold",
                "constraint_legacy_guard_mode": "none",
                "constraint_cost_aggregation": self.constraint_cost_aggregation,
                "constraint_cost_quantile": float(self.constraint_cost_quantile),
                "constraint_subsample_obs": int(self.constraint_subsample_obs),
                "constraint_sampling_mode": "full_batch" if self.constraint_subsample_obs <= 0 else "fixed_subsample",
                "constraint_threshold": self.constraint_threshold,
                "policy_local_sensitivity_cost_mean": final_eval["cost_mean"],
                "policy_local_sensitivity_cost_update": final_eval["cost_update"],
                "policy_local_sensitivity_cost_max": final_eval["cost_max"],
                "policy_local_sensitivity_cost_quantile": final_eval["cost_quantile"],
                "policy_local_sensitivity_effective_cost_update": final_eval["cost_update"],
                "policy_local_sensitivity_legacy_cost_mean": final_eval["cost_mean"],
                "policy_local_sensitivity_legacy_cost_update": final_eval["cost_update"],
                "constraint_violation_rate": final_eval["violation_rate"],
                "constraint_legacy_violation_rate": final_eval["violation_rate"],
                "constraint_update_error": final_eval["constraint_value"],
                "constraint_penalty_error": final_eval["constraint_value"],
                "constraint_sample_count": final_eval["sample_count"],
                "lagrange_multiplier": 0.0,
                "cpo_finite": float(1.0 if actor_result["finite"] else 0.0),
                "cpo_line_search_accepted": float(1.0 if actor_result["line_search"]["accepted"] else 0.0),
                "cpo_line_search_backtrack": (
                    float(actor_result["line_search"]["accepted_attempt"]["backtrack"])
                    if actor_result["line_search"]["accepted_attempt"] is not None
                    else None
                ),
                "cpo_kl": final_eval["kl"],
                "cpo_surrogate": final_eval["surrogate"],
                "cpo_old_surrogate": actor_result["old_eval"]["surrogate"],
                "cpo_dual_case": actor_result["dual"]["case"],
                "cpo_dual_lambda": actor_result["dual"]["lambda"],
                "cpo_dual_nu": actor_result["dual"]["nu"],
                "cpo_trust_region_quadratic": actor_result["dual"]["trust_region_quadratic"],
                "cpo_reward_gradient_norm": actor_result["reward_gradient_norm"],
                "cpo_cost_gradient_norm": actor_result["cost_gradient_norm"],
                "cpo_reward_cg_final_residual": actor_result["cg_reward"]["final_residual"],
                "cpo_cost_cg_final_residual": actor_result["cg_cost"]["final_residual"],
                "cpo_step_norm": actor_result["step_norm"],
                "cpo_update_wall_time_s": float(update_wall_time_s),
                "cpo_cuda_peak_allocated_mb": cuda_peak_allocated_mb,
                "cpo_cuda_peak_reserved_mb": cuda_peak_reserved_mb,
                "value_loss": float(value_loss.detach().item()),
            }
            self.constraint_trace.append(trace_entry)
            batch_traces.append(trace_entry)

            mean_value_loss += float(value_loss.detach().item())
            mean_surrogate_loss += -float(final_eval["surrogate"])

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        if batch_traces:
            self.latest_stats = batch_traces[-1].copy()
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss

    def get_logging_stats(self):
        if not self.latest_stats:
            return {}
        return {
            "Constraint/policy_local_sensitivity_cost_mean": self.latest_stats["policy_local_sensitivity_cost_mean"],
            "Constraint/policy_local_sensitivity_cost_update": self.latest_stats["policy_local_sensitivity_cost_update"],
            "Constraint/constraint_threshold": self.latest_stats["constraint_threshold"],
            "Constraint/constraint_error": self.latest_stats["constraint_update_error"],
            "Constraint/constraint_violation_rate": self.latest_stats["constraint_violation_rate"],
            "CPO/line_search_accepted": self.latest_stats["cpo_line_search_accepted"],
            "CPO/kl": self.latest_stats["cpo_kl"],
            "CPO/surrogate": self.latest_stats["cpo_surrogate"],
            "CPO/reward_gradient_norm": self.latest_stats["cpo_reward_gradient_norm"],
            "CPO/cost_gradient_norm": self.latest_stats["cpo_cost_gradient_norm"],
        }

    def extra_state_dict(self):
        return {
            "constraint_trace": list(self.constraint_trace),
            "latest_stats": dict(self.latest_stats),
            "boundary": "CPO-style local diagnostic only; not official CPO parity.",
        }

    def load_extra_state_dict(self, state_dict):
        if not state_dict:
            return
        self.constraint_trace = list(state_dict.get("constraint_trace", []))
        self.latest_stats = dict(state_dict.get("latest_stats", {}))

    def get_artifact_payload(self):
        if not self.constraint_trace:
            return {}
        cost_history = [entry["policy_local_sensitivity_cost_mean"] for entry in self.constraint_trace]
        accepted = [entry["cpo_line_search_accepted"] for entry in self.constraint_trace]
        payload = {
            "constraint_metrics": {
                "constraint_sample_count": int(sum(entry["constraint_sample_count"] for entry in self.constraint_trace)),
                "constraint_cost_aggregation": self.constraint_cost_aggregation,
                "constraint_cost_quantile": float(self.constraint_cost_quantile),
                "constraint_subsample_obs": int(self.constraint_subsample_obs),
                "constraint_sampling_mode": "full_batch" if self.constraint_subsample_obs <= 0 else "fixed_subsample",
                "constraint_violation_rate": self.latest_stats.get("constraint_violation_rate"),
                "dual_update_mode": "cpo_style_trust_region",
                "lagrange_multiplier": 0.0,
                "local_sensitivity_threshold": self.constraint_threshold,
                "policy_local_sensitivity_cost_mean": self.latest_stats.get("policy_local_sensitivity_cost_mean"),
                "policy_local_sensitivity_cost_update": self.latest_stats.get("policy_local_sensitivity_cost_update"),
                "policy_local_sensitivity_cost_max": self.latest_stats.get("policy_local_sensitivity_cost_max"),
                "policy_local_sensitivity_cost_quantile": self.latest_stats.get(
                    "policy_local_sensitivity_cost_quantile"
                ),
                "policy_local_sensitivity_cost_std": (
                    statistics.pstdev(cost_history) if len(cost_history) > 1 else 0.0
                ),
                "cpo_line_search_accept_rate": statistics.fmean(accepted) if accepted else None,
                "cpo_latest_kl": self.latest_stats.get("cpo_kl"),
                "cpo_latest_dual_case": self.latest_stats.get("cpo_dual_case"),
            },
            "lagrange_multiplier_trace": self.constraint_trace,
        }
        return payload
