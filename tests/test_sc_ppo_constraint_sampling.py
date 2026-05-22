from __future__ import annotations

import unittest

import torch.nn as nn
import torch

from tests._sc_ppo_loader import load_sc_ppo_module


class DummyActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def to(self, device):
        return self


class ConstraintSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc_ppo_module = load_sc_ppo_module()

    def build_algo(self, subsample_obs):
        return self.sc_ppo_module.SCPPO(
            DummyActorCritic(),
            constraint={"subsample_obs": subsample_obs},
            device="cpu",
        )

    def test_zero_subsample_means_full_batch(self):
        algo = self.build_algo(0)
        obs_batch = torch.randn(64, 8)
        selected = algo._constraint_obs_batch(obs_batch)
        self.assertEqual(selected.shape[0], 64)
        self.assertEqual(algo._constraint_sampling_mode(), "full_batch")

    def test_all_alias_means_full_batch(self):
        algo = self.build_algo("all")
        obs_batch = torch.randn(16, 8)
        selected = algo._constraint_obs_batch(obs_batch)
        self.assertEqual(selected.shape[0], 16)
        self.assertEqual(algo._constraint_sampling_mode(), "full_batch")

    def test_positive_subsample_caps_batch(self):
        algo = self.build_algo(8)
        obs_batch = torch.randn(64, 8)
        selected = algo._constraint_obs_batch(obs_batch)
        self.assertEqual(selected.shape[0], 8)
        self.assertEqual(algo._constraint_sampling_mode(), "random_subsample")


if __name__ == "__main__":
    unittest.main()
