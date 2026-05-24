from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_GYM_ROOT = REPO_ROOT / ".external" / "humanoid-gym"


def load_action_scaling_module():
    root_str = str(HUMANOID_GYM_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("humanoid.algo.ppo.action_scaling_ppo")


class DummyActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        self._action_output_scale = 1.0

    def to(self, device):
        return self

    def set_action_output_scale(self, value):
        self._action_output_scale = float(value)

    def get_action_output_scale(self):
        return float(self._action_output_scale)

    def act_inference(self, observations):
        return self.linear(observations) * self._action_output_scale


class ActionScalingPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_action_scaling_module()

    def build_algo(self, **constraint_overrides):
        constraint = {
            "lambda_init": 0.5,
            "action_scale_mode": "inverse_lagrange",
            "action_scale_gain": 0.2,
            "action_scale_min": 0.25,
            "action_scale_max": 1.0,
        }
        constraint.update(constraint_overrides)
        return self.module.ActionScalingPPO(
            DummyActorCritic(),
            constraint=constraint,
            device="cpu",
        )

    def test_initial_action_scale_is_derived_from_lagrange_multiplier(self):
        algo = self.build_algo()

        expected = 1.0 / (1.0 + 0.2 * 0.5)
        self.assertAlmostEqual(algo.current_action_scale, expected)
        self.assertAlmostEqual(algo.actor_critic.get_action_output_scale(), expected)

    def test_action_scale_respects_minimum_bound(self):
        algo = self.build_algo()

        with torch.no_grad():
            algo.lagrange_multiplier.fill_(100.0)
        algo._apply_action_scale(algo._compute_action_scale())

        self.assertAlmostEqual(algo.current_action_scale, 0.25)
        self.assertAlmostEqual(algo.actor_critic.get_action_output_scale(), 0.25)

    def test_local_sensitivity_metrics_do_not_build_higher_order_graph(self):
        algo = self.build_algo()
        obs_batch = torch.randn(8, 4)

        metrics = algo._local_sensitivity_metrics(obs_batch)

        self.assertFalse(metrics["cost_mean"].requires_grad)
        self.assertFalse(metrics["cost_for_update"].requires_grad)


if __name__ == "__main__":
    unittest.main()
