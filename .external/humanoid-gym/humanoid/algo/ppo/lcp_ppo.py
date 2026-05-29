# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import statistics

import torch
import torch.nn as nn

from .ppo import PPO


class LCPPPO(PPO):
    """PPO with a fixed LCP-style gradient penalty on actor log-probability."""

    def __init__(self, actor_critic, lcp=None, device="cpu", **kwargs):
        super().__init__(actor_critic, device=device, **kwargs)

        cfg = lcp or {}
        self.lcp_enabled = bool(cfg.get("enabled", True))
        self.lcp_weight = float(cfg.get("lcp_weight", cfg.get("weight", 0.002)))
        self.lcp_subsample_obs = self._parse_lcp_subsample_obs(cfg.get("subsample_obs", 64))
        self.lcp_epsilon = float(cfg.get("epsilon", 1e-12))
        self.lcp_local_sensitivity_threshold = cfg.get("local_sensitivity_threshold", 3.8)
        if self.lcp_local_sensitivity_threshold is not None:
            self.lcp_local_sensitivity_threshold = float(self.lcp_local_sensitivity_threshold)
        self.lcp_trace = []
        self.latest_stats = {}

    def _parse_lcp_subsample_obs(self, raw_value):
        if raw_value is None:
            return 0
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"all", "full", "full_batch", "full-batch"}:
                return 0
            raw_value = normalized
        return max(int(raw_value), 0)

    def _sample_lcp_batch(self, obs_batch, actions_batch):
        if self.lcp_subsample_obs <= 0 or obs_batch.shape[0] <= self.lcp_subsample_obs:
            return obs_batch, actions_batch
        indices = torch.randperm(obs_batch.shape[0], device=obs_batch.device)[: self.lcp_subsample_obs]
        return obs_batch.index_select(0, indices), actions_batch.index_select(0, indices)

    def _lcp_sampling_mode(self):
        if self.lcp_subsample_obs <= 0:
            return "full_batch"
        return "random_subsample"

    def _lcp_gradient_penalty(self, obs_batch, actions_batch):
        sampled_obs, sampled_actions = self._sample_lcp_batch(obs_batch, actions_batch)
        sampled_obs = sampled_obs.detach().clone().requires_grad_(True)
        sampled_actions = sampled_actions.detach()

        self.actor_critic.act(sampled_obs)
        log_prob = self.actor_critic.get_actions_log_prob(sampled_actions)
        grads = torch.autograd.grad(
            outputs=log_prob.sum(),
            inputs=sampled_obs,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        squared_norm = torch.sum(torch.square(grads), dim=1)
        penalty = squared_norm.mean()
        grad_norm = torch.sqrt(torch.clamp(squared_norm, min=self.lcp_epsilon))
        return {
            "penalty": penalty,
            "grad_norm_mean": grad_norm.mean(),
            "grad_norm_max": grad_norm.max(),
            "sample_count": int(sampled_obs.shape[0]),
        }

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_lcp_gradient_penalty = 0
        mean_lcp_penalty_loss = 0
        mean_lcp_grad_norm = 0
        mean_lcp_grad_norm_max = 0
        total_lcp_samples = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            lcp_gradient_penalty = torch.tensor(0.0, device=self.device)
            lcp_penalty_loss = torch.tensor(0.0, device=self.device)
            lcp_grad_norm_mean = torch.tensor(0.0, device=self.device)
            lcp_grad_norm_max = torch.tensor(0.0, device=self.device)
            lcp_sample_count = 0
            if self.lcp_enabled and self.lcp_weight > 0.0:
                lcp_stats = self._lcp_gradient_penalty(obs_batch, actions_batch)
                lcp_gradient_penalty = lcp_stats["penalty"]
                lcp_penalty_loss = self.lcp_weight * lcp_gradient_penalty
                lcp_grad_norm_mean = lcp_stats["grad_norm_mean"]
                lcp_grad_norm_max = lcp_stats["grad_norm_max"]
                lcp_sample_count = lcp_stats["sample_count"]

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + lcp_penalty_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_lcp_gradient_penalty += lcp_gradient_penalty.item()
            mean_lcp_penalty_loss += lcp_penalty_loss.item()
            mean_lcp_grad_norm += lcp_grad_norm_mean.item()
            mean_lcp_grad_norm_max += lcp_grad_norm_max.item()
            total_lcp_samples += lcp_sample_count

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_lcp_gradient_penalty /= num_updates
        mean_lcp_penalty_loss /= num_updates
        mean_lcp_grad_norm /= num_updates
        mean_lcp_grad_norm_max /= num_updates

        trace_entry = {
            "iteration": len(self.lcp_trace),
            "lcp_enabled": bool(self.lcp_enabled),
            "lcp_weight": float(self.lcp_weight),
            "lcp_penalty_form": "mean(||grad_obs log pi(a_batch | obs_batch)||^2)",
            "lcp_gradient_penalty_mean": float(mean_lcp_gradient_penalty),
            "lcp_penalty_loss_mean": float(mean_lcp_penalty_loss),
            "lcp_grad_norm_mean": float(mean_lcp_grad_norm),
            "lcp_grad_norm_max": float(mean_lcp_grad_norm_max),
            "lcp_subsample_obs": int(self.lcp_subsample_obs),
            "lcp_sampling_mode": self._lcp_sampling_mode(),
            "lcp_sample_count": int(total_lcp_samples),
            "local_sensitivity_threshold": self.lcp_local_sensitivity_threshold,
        }
        self.lcp_trace.append(trace_entry)
        self.latest_stats = trace_entry.copy()

        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss

    def get_logging_stats(self):
        if not self.latest_stats:
            return {}
        return {
            "LCP/lcp_weight": self.latest_stats["lcp_weight"],
            "LCP/lcp_gradient_penalty_mean": self.latest_stats["lcp_gradient_penalty_mean"],
            "LCP/lcp_grad_norm_mean": self.latest_stats["lcp_grad_norm_mean"],
            "LCP/lcp_grad_norm_max": self.latest_stats["lcp_grad_norm_max"],
            "Loss/lcp_penalty": self.latest_stats["lcp_penalty_loss_mean"],
        }

    def extra_state_dict(self):
        return {
            "lcp_trace": list(self.lcp_trace),
            "latest_stats": dict(self.latest_stats),
            "lcp_weight": float(self.lcp_weight),
            "lcp_subsample_obs": int(self.lcp_subsample_obs),
        }

    def load_extra_state_dict(self, state_dict):
        if not state_dict:
            return
        self.lcp_trace = list(state_dict.get("lcp_trace", []))
        self.latest_stats = dict(state_dict.get("latest_stats", {}))

    def get_artifact_payload(self):
        if not self.lcp_trace:
            return {}
        penalty_history = [entry["lcp_gradient_penalty_mean"] for entry in self.lcp_trace]
        return {
            "constraint_metrics": {
                "constraint_source": "lcp_soft_gradient_penalty",
                "dual_update_mode": "fixed_soft_penalty",
                "lcp_enabled": bool(self.lcp_enabled),
                "lcp_weight": float(self.lcp_weight),
                "lcp_penalty_form": "mean(||grad_obs log pi(a_batch | obs_batch)||^2)",
                "lcp_gradient_penalty_mean": self.latest_stats.get("lcp_gradient_penalty_mean"),
                "lcp_gradient_penalty_std": (
                    statistics.pstdev(penalty_history) if len(penalty_history) > 1 else 0.0
                ),
                "lcp_penalty_loss_mean": self.latest_stats.get("lcp_penalty_loss_mean"),
                "lcp_grad_norm_mean": self.latest_stats.get("lcp_grad_norm_mean"),
                "lcp_grad_norm_max": self.latest_stats.get("lcp_grad_norm_max"),
                "lcp_subsample_obs": int(self.lcp_subsample_obs),
                "lcp_sampling_mode": self._lcp_sampling_mode(),
                "constraint_sample_count": int(sum(entry["lcp_sample_count"] for entry in self.lcp_trace)),
                "local_sensitivity_threshold": self.lcp_local_sensitivity_threshold,
            },
            "lagrange_multiplier_trace": self.lcp_trace,
        }
