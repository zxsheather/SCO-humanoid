from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn
from torch.distributions import Normal


REPO_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_GYM_ROOT = REPO_ROOT / ".external" / "humanoid-gym"


def load_logprob_constraint_module():
    root_str = str(HUMANOID_GYM_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("humanoid.algo.ppo.logprob_constraint_ppo")


class DummyActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        self.std = nn.Parameter(torch.full((2,), 0.5))
        self.distribution = None

    def to(self, device):
        return self

    def update_distribution(self, observations):
        mean = self.linear(observations)
        std = torch.ones_like(mean) * self.std
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.rsample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)


class LogProbConstraintPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_logprob_constraint_module()

    def build_algo(self, **constraint_overrides):
        constraint = {
            "enabled": True,
            "threshold": 6.0,
            "subsample_obs": 4,
            "cost_aggregation": "mean",
            "lambda_init": 0.0,
        }
        constraint.update(constraint_overrides)
        return self.module.LogProbConstraintPPO(
            DummyActorCritic(),
            constraint=constraint,
            device="cpu",
        )

    def test_logprob_gradient_metrics_build_higher_order_graph(self):
        algo = self.build_algo()
        obs_batch = torch.randn(8, 4)
        actions_batch = torch.randn(8, 2)

        metrics = algo._logprob_gradient_metrics(obs_batch, actions_batch)

        self.assertTrue(metrics["cost_mean"].requires_grad)
        self.assertTrue(metrics["cost_for_update"].requires_grad)
        self.assertTrue(metrics["grad_norm_mean"].requires_grad)

    def test_all_alias_means_full_batch(self):
        algo = self.build_algo(subsample_obs="all")
        obs_batch = torch.randn(8, 4)
        actions_batch = torch.randn(8, 2)

        selected_obs, selected_actions = algo._sample_constraint_batch(obs_batch, actions_batch)

        self.assertEqual(selected_obs.shape[0], 8)
        self.assertEqual(selected_actions.shape[0], 8)
        self.assertEqual(algo._constraint_sampling_mode(), "full_batch")

    def test_artifact_payload_uses_logprob_constraint_keys(self):
        algo = self.build_algo()
        algo.constraint_trace = [
            {
                "constraint_sample_count": 64,
                "constraint_violation_rate": 0.25,
                "lagrange_multiplier": 0.5,
                "logprob_gradient_cost_mean": 6.1,
                "logprob_gradient_cost_update": 6.1,
                "logprob_gradient_cost_max": 8.0,
                "logprob_gradient_cost_quantile": 7.2,
                "logprob_gradient_norm_mean": 2.3,
                "logprob_gradient_norm_max": 4.2,
            }
        ]
        algo.latest_stats = algo.constraint_trace[0].copy()

        payload = algo.get_artifact_payload()
        metrics = payload["constraint_metrics"]

        self.assertEqual(metrics["constraint_source"], "logprob_gradient_hard_constraint")
        self.assertEqual(metrics["constraint_threshold"], 6.0)
        self.assertEqual(metrics["logprob_gradient_cost_mean"], 6.1)
        self.assertEqual(metrics["logprob_gradient_norm_mean"], 2.3)
        self.assertEqual(payload["lagrange_multiplier_trace"], algo.constraint_trace)


if __name__ == "__main__":
    unittest.main()
