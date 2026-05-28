from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import run_omnisafe_adapter_smoke as smoke  # noqa: E402


class FakeHumanoidGymEnv:
    def __init__(self) -> None:
        self.num_envs = 3
        self.num_actions = 4
        self.reset_called = False
        self._obs = torch.ones(self.num_envs, 6)

    def reset(self) -> None:
        self.reset_called = True

    def get_observations(self) -> torch.Tensor:
        return self._obs.clone()

    def step(self, action: torch.Tensor):
        self.last_action = action
        obs = torch.full((self.num_envs, 6), 2.0)
        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.tensor([False, True, True])
        infos = {"time_outs": torch.tensor([False, False, True]), "episode": {"r": rewards}}
        return obs, None, rewards, dones, infos


class OmniSafeAdapterSmokeTests(unittest.TestCase):
    def test_reset_returns_observation_and_noncanonical_cost_metadata(self) -> None:
        fake_env = FakeHumanoidGymEnv()
        adapter = smoke.OmniSafeHumanoidGymAdapter(
            fake_env,
            cost_source="non_canonical_zero_smoke",
            cost_is_canonical=False,
        )

        obs, info = adapter.reset()

        self.assertTrue(fake_env.reset_called)
        self.assertEqual(list(obs.shape), [3, 6])
        self.assertEqual(info["cost_source"], "non_canonical_zero_smoke")
        self.assertFalse(info["cost_is_canonical"])

    def test_step_maps_humanoid_gym_outputs_to_omnisafe_tuple_contract(self) -> None:
        fake_env = FakeHumanoidGymEnv()
        adapter = smoke.OmniSafeHumanoidGymAdapter(
            fake_env,
            cost_source="non_canonical_zero_smoke",
            cost_is_canonical=False,
        )
        action = torch.zeros(fake_env.num_envs, fake_env.num_actions)

        obs, reward, cost, terminated, truncated, info = adapter.step(action)

        self.assertEqual(list(obs.shape), [3, 6])
        self.assertTrue(torch.equal(reward, torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(cost, torch.zeros(3)))
        self.assertTrue(torch.equal(terminated, torch.tensor([False, True, False])))
        self.assertTrue(torch.equal(truncated, torch.tensor([False, False, True])))
        self.assertEqual(info["cost_source"], "non_canonical_zero_smoke")
        self.assertFalse(info["cost_is_canonical"])
        self.assertEqual(info["humanoid_gym_info_keys"], ["episode", "time_outs"])

    def test_step_summary_records_omnisafe_contract_shapes(self) -> None:
        fake_env = FakeHumanoidGymEnv()
        adapter = smoke.OmniSafeHumanoidGymAdapter(
            fake_env,
            cost_source="non_canonical_zero_smoke",
            cost_is_canonical=False,
        )
        action = torch.zeros(fake_env.num_envs, fake_env.num_actions)
        result = adapter.step(action)

        summary = smoke.summarize_step(0, action, result)

        self.assertEqual(summary["action_shape"], [3, 4])
        self.assertEqual(summary["observation_shape"], [3, 6])
        self.assertEqual(summary["reward_shape"], [3])
        self.assertEqual(summary["cost_shape"], [3])
        self.assertEqual(summary["terminated_shape"], [3])
        self.assertEqual(summary["truncated_shape"], [3])
        self.assertEqual(summary["cost_source"], "non_canonical_zero_smoke")
        self.assertFalse(summary["cost_is_canonical"])


if __name__ == "__main__":
    unittest.main()
