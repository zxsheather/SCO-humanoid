# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import statistics

import torch
import torch.nn as nn

from .ppo import PPO


class SCPPO(PPO):
    def __init__(self, actor_critic, constraint=None, device="cpu", **kwargs):
        super().__init__(actor_critic, device=device, **kwargs)

        cfg = constraint or {}
        self.constraint_enabled = bool(cfg.get("enabled", True))
        self.constraint_threshold = float(cfg.get("threshold", 5.5))
        self.constraint_subsample_obs = self._parse_constraint_subsample_obs(cfg.get("subsample_obs", 8))
        self.constraint_update_mode = str(cfg.get("update_mode", "pid")).lower()
        self.constraint_dual_lr = float(cfg.get("dual_lr", 0.01))
        self.constraint_cost_aggregation = str(cfg.get("cost_aggregation", "mean")).lower()
        self.constraint_cost_quantile = min(max(float(cfg.get("cost_quantile", 0.9)), 0.0), 1.0)
        self.constraint_pid_kp = float(cfg.get("pid_kp", 0.05))
        self.constraint_pid_ki = float(cfg.get("pid_ki", 0.001))
        self.constraint_pid_kd = float(cfg.get("pid_kd", 0.01))
        self.constraint_pid_integral_mode = str(cfg.get("pid_integral_mode", "standard")).lower()
        self.constraint_pid_integral_decay = min(max(float(cfg.get("pid_integral_decay", 1.0)), 0.0), 1.0)
        self.constraint_lambda_min = float(cfg.get("lambda_min", 0.0))
        self.constraint_lambda_max = float(cfg.get("lambda_max", 5.0))
        self.constraint_integral_min = float(cfg.get("integral_min", -10.0))
        self.constraint_integral_max = float(cfg.get("integral_max", 10.0))
        self.constraint_epsilon = float(cfg.get("epsilon", 1e-6))

        self.lagrange_multiplier = torch.tensor(
            float(cfg.get("lambda_init", 0.0)),
            device=self.device,
        )
        self.constraint_integral_error = 0.0
        self.previous_constraint_error = 0.0
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

    def _constraint_obs_batch(self, obs_batch):
        if self.constraint_subsample_obs <= 0 or obs_batch.shape[0] <= self.constraint_subsample_obs:
            return obs_batch
        indices = torch.randperm(obs_batch.shape[0], device=obs_batch.device)[: self.constraint_subsample_obs]
        return obs_batch.index_select(0, indices)

    def _constraint_sampling_mode(self):
        if self.constraint_subsample_obs <= 0:
            return "full_batch"
        return "random_subsample"

    def _local_sensitivity_metrics(self, obs_batch):
        sampled_obs = self._constraint_obs_batch(obs_batch).detach().clone().requires_grad_(True)
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

    def _integrate_constraint_error(self, error):
        integral = float(self.constraint_integral_error)
        lower_bound_active = float(self.lagrange_multiplier.item()) <= (self.constraint_lambda_min + self.constraint_epsilon)

        if self.constraint_pid_integral_mode == "standard":
            integral += error
        elif self.constraint_pid_integral_mode == "lower_bound_clamp":
            if lower_bound_active:
                integral = max(integral, 0.0)
            integral += error
        elif self.constraint_pid_integral_mode == "lower_bound_decay":
            if lower_bound_active and error < 0.0:
                integral *= self.constraint_pid_integral_decay
            else:
                integral += error
        else:
            raise ValueError(f"Unsupported pid_integral_mode: {self.constraint_pid_integral_mode}")

        self.constraint_integral_error = min(
            max(integral, self.constraint_integral_min),
            self.constraint_integral_max,
        )

    def _update_lagrange_multiplier(self, constraint_error):
        error = float(constraint_error)
        if self.constraint_update_mode == "pid":
            self._integrate_constraint_error(error)
            derivative = error - self.previous_constraint_error
            delta = (
                self.constraint_pid_kp * error
                + self.constraint_pid_ki * self.constraint_integral_error
                + self.constraint_pid_kd * derivative
            )
            self.previous_constraint_error = error
        else:
            delta = self.constraint_dual_lr * error

        with torch.no_grad():
            updated = float(self.lagrange_multiplier.item()) + delta
            updated = min(max(updated, self.constraint_lambda_min), self.constraint_lambda_max)
            self.lagrange_multiplier.fill_(updated)
        return delta

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_constraint_cost = 0
        mean_constraint_cost_update = 0
        mean_constraint_cost_max = 0
        mean_constraint_cost_quantile = 0
        mean_constraint_penalty = 0
        mean_constraint_violation_rate = 0
        total_constraint_samples = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                constraint_cost = torch.tensor(0.0, device=self.device)
                constraint_cost_update = torch.tensor(0.0, device=self.device)
                constraint_cost_max = torch.tensor(0.0, device=self.device)
                constraint_cost_quantile = torch.tensor(0.0, device=self.device)
                constraint_violation_rate = torch.tensor(0.0, device=self.device)
                constraint_penalty = torch.tensor(0.0, device=self.device)
                constraint_sample_count = 0
                if self.constraint_enabled:
                    constraint_stats = self._local_sensitivity_metrics(obs_batch)
                    constraint_cost = constraint_stats["cost_mean"]
                    constraint_cost_update = constraint_stats["cost_for_update"]
                    constraint_cost_max = constraint_stats["cost_max"]
                    constraint_cost_quantile = constraint_stats["cost_quantile"]
                    constraint_violation_rate = constraint_stats["violation_rate"]
                    constraint_sample_count = constraint_stats["sample_count"]
                    constraint_error = constraint_cost_update - self.constraint_threshold
                    constraint_penalty = self.lagrange_multiplier.detach() * constraint_error

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + constraint_penalty
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_constraint_cost += constraint_cost.item()
                mean_constraint_cost_update += constraint_cost_update.item()
                mean_constraint_cost_max += constraint_cost_max.item()
                mean_constraint_cost_quantile += constraint_cost_quantile.item()
                mean_constraint_penalty += constraint_penalty.item()
                mean_constraint_violation_rate += constraint_violation_rate.item()
                total_constraint_samples += constraint_sample_count

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_constraint_cost /= num_updates
        mean_constraint_cost_update /= num_updates
        mean_constraint_cost_max /= num_updates
        mean_constraint_cost_quantile /= num_updates
        mean_constraint_penalty /= num_updates
        mean_constraint_violation_rate /= num_updates

        lagrange_delta = 0.0
        if self.constraint_enabled:
            lagrange_delta = self._update_lagrange_multiplier(mean_constraint_cost_update - self.constraint_threshold)

        trace_entry = {
            "iteration": len(self.constraint_trace),
            "lagrange_multiplier": float(self.lagrange_multiplier.item()),
            "lagrange_delta": float(lagrange_delta),
            "constraint_cost_aggregation": self.constraint_cost_aggregation,
            "constraint_cost_quantile": float(self.constraint_cost_quantile),
            "constraint_subsample_obs": int(self.constraint_subsample_obs),
            "constraint_sampling_mode": self._constraint_sampling_mode(),
            "constraint_threshold": self.constraint_threshold,
            "policy_local_sensitivity_cost_mean": float(mean_constraint_cost),
            "policy_local_sensitivity_cost_update": float(mean_constraint_cost_update),
            "policy_local_sensitivity_cost_max": float(mean_constraint_cost_max),
            "policy_local_sensitivity_cost_quantile": float(mean_constraint_cost_quantile),
            "pid_integral_mode": self.constraint_pid_integral_mode,
            "pid_integral_decay": float(self.constraint_pid_integral_decay),
            "constraint_integral_error": float(self.constraint_integral_error),
            "constraint_violation_rate": float(mean_constraint_violation_rate),
            "constraint_penalty_loss_mean": float(mean_constraint_penalty),
            "constraint_sample_count": int(total_constraint_samples),
        }
        self.constraint_trace.append(trace_entry)
        self.latest_stats = trace_entry.copy()
        self.latest_stats["constraint_error"] = float(mean_constraint_cost_update - self.constraint_threshold)

        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss

    def get_logging_stats(self):
        if not self.latest_stats:
            return {}
        return {
            "Constraint/lagrange_multiplier": self.latest_stats["lagrange_multiplier"],
            "Constraint/lagrange_delta": self.latest_stats["lagrange_delta"],
            "Constraint/policy_local_sensitivity_cost_mean": self.latest_stats["policy_local_sensitivity_cost_mean"],
            "Constraint/policy_local_sensitivity_cost_update": self.latest_stats["policy_local_sensitivity_cost_update"],
            "Constraint/policy_local_sensitivity_cost_max": self.latest_stats["policy_local_sensitivity_cost_max"],
            "Constraint/policy_local_sensitivity_cost_quantile": self.latest_stats[
                "policy_local_sensitivity_cost_quantile"
            ],
            "Constraint/constraint_threshold": self.latest_stats["constraint_threshold"],
            "Constraint/constraint_error": self.latest_stats["constraint_error"],
            "Constraint/constraint_integral_error": self.latest_stats["constraint_integral_error"],
            "Constraint/constraint_violation_rate": self.latest_stats["constraint_violation_rate"],
            "Loss/constraint_penalty": self.latest_stats["constraint_penalty_loss_mean"],
        }

    def extra_state_dict(self):
        return {
            "lagrange_multiplier": float(self.lagrange_multiplier.item()),
            "constraint_integral_error": float(self.constraint_integral_error),
            "previous_constraint_error": float(self.previous_constraint_error),
            "constraint_trace": list(self.constraint_trace),
            "latest_stats": dict(self.latest_stats),
        }

    def load_extra_state_dict(self, state_dict):
        if not state_dict:
            return
        with torch.no_grad():
            self.lagrange_multiplier.fill_(float(state_dict.get("lagrange_multiplier", self.lagrange_multiplier.item())))
        self.constraint_integral_error = float(state_dict.get("constraint_integral_error", 0.0))
        self.previous_constraint_error = float(state_dict.get("previous_constraint_error", 0.0))
        self.constraint_trace = list(state_dict.get("constraint_trace", []))
        self.latest_stats = dict(state_dict.get("latest_stats", {}))

    def get_artifact_payload(self):
        if not self.constraint_trace:
            return {}
        cost_history = [entry["policy_local_sensitivity_cost_mean"] for entry in self.constraint_trace]
        payload = {
            "constraint_metrics": {
                "constraint_sample_count": int(sum(entry["constraint_sample_count"] for entry in self.constraint_trace)),
                "constraint_cost_aggregation": self.constraint_cost_aggregation,
                "constraint_cost_quantile": float(self.constraint_cost_quantile),
                "constraint_subsample_obs": int(self.constraint_subsample_obs),
                "constraint_sampling_mode": self._constraint_sampling_mode(),
                "constraint_violation_rate": self.latest_stats.get("constraint_violation_rate"),
                "dual_update_mode": self.constraint_update_mode,
                "lagrange_multiplier": self.latest_stats.get("lagrange_multiplier"),
                "lagrange_multiplier_max": self.constraint_lambda_max,
                "local_sensitivity_threshold": self.constraint_threshold,
                "pid_integral_mode": self.constraint_pid_integral_mode,
                "pid_integral_decay": float(self.constraint_pid_integral_decay),
                "policy_local_sensitivity_cost_mean": self.latest_stats.get("policy_local_sensitivity_cost_mean"),
                "policy_local_sensitivity_cost_update": self.latest_stats.get("policy_local_sensitivity_cost_update"),
                "policy_local_sensitivity_cost_max": self.latest_stats.get("policy_local_sensitivity_cost_max"),
                "policy_local_sensitivity_cost_quantile": self.latest_stats.get(
                    "policy_local_sensitivity_cost_quantile"
                ),
                "policy_local_sensitivity_cost_std": (
                    statistics.pstdev(cost_history) if len(cost_history) > 1 else 0.0
                ),
            },
            "lagrange_multiplier_trace": self.constraint_trace,
        }
        return payload
