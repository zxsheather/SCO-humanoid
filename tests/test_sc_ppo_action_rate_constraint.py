from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
PPO_DIR = REPO_ROOT / ".external" / "humanoid-gym" / "humanoid" / "algo" / "ppo"


def load_scppo_class():
    package_name = "test_scppo_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PPO_DIR)]
    sys.modules[package_name] = package

    for module_name in ("actor_critic", "rollout_storage", "ppo", "sc_ppo"):
        full_name = f"{package_name}.{module_name}"
        if full_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(full_name, PPO_DIR / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.sc_ppo"].SCPPO


class ConstantActionActor(nn.Module):
    def __init__(self, action_values: list[float]):
        super().__init__()
        self.action_bias = nn.Parameter(torch.tensor(action_values, dtype=torch.float32))

    def act_inference(self, obs):
        return self.action_bias.unsqueeze(0).expand(obs.shape[0], -1)


class SCPPOActionRateTests(unittest.TestCase):
    def test_action_rate_metrics_use_latest_frame_previous_actions(self) -> None:
        scppo_cls = load_scppo_class()
        actor = ConstantActionActor([1.0] * 12)
        alg = scppo_cls(
            actor,
            device="cpu",
            constraint={
                "enabled": True,
                "objective": "action_rate",
                "threshold": 1.0,
                "cost_aggregation": "mean",
                "cost_quantile": 0.5,
                "subsample_obs": 0,
                "action_rate_observation_frame_size": 47,
                "action_rate_observation_action_slice": [29, 41],
            },
        )

        obs = torch.zeros(2, 94)
        obs[0, 29:41] = 9.0
        obs[0, 76:88] = 0.0
        obs[1, 29:41] = 7.0
        obs[1, 76:88] = 1.0

        previous_actions = alg._action_rate_previous_actions(obs, action_dim=12)
        metrics = alg._action_rate_metrics(obs)

        expected_norm = float(torch.sqrt(torch.tensor(12.0)))
        expected_zero_norm = 0.001
        self.assertTrue(torch.allclose(previous_actions[0], torch.zeros(12)))
        self.assertTrue(torch.allclose(previous_actions[1], torch.ones(12)))
        self.assertAlmostEqual(float(metrics["cost_mean"]), (expected_norm + expected_zero_norm) / 2.0, places=6)
        self.assertAlmostEqual(
            float(metrics["cost_for_update"]),
            (expected_norm + expected_zero_norm) / 2.0,
            places=6,
        )
        self.assertAlmostEqual(float(metrics["cost_max"]), expected_norm, places=6)
        self.assertAlmostEqual(float(metrics["violation_rate"]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
