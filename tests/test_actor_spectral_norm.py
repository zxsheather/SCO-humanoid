from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_GYM_ROOT = REPO_ROOT / ".external" / "humanoid-gym"


def load_actor_critic_module():
    root_str = str(HUMANOID_GYM_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("humanoid.algo.ppo.actor_critic")


class ActorSpectralNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor_critic_module = load_actor_critic_module()

    def test_actor_spectral_norm_wraps_actor_only(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
            actor_spectral_norm=True,
        )

        actor_linears = [module for module in model.actor if isinstance(module, nn.Linear)]
        critic_linears = [module for module in model.critic if isinstance(module, nn.Linear)]

        self.assertTrue(actor_linears)
        self.assertTrue(critic_linears)
        self.assertTrue(all(hasattr(layer, "weight_orig") for layer in actor_linears))
        self.assertTrue(all(not hasattr(layer, "weight_orig") for layer in critic_linears))
        self.assertTrue(model.actor_spectral_norm)

    def test_actor_spectral_norm_is_disabled_by_default(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
        )

        actor_linears = [module for module in model.actor if isinstance(module, nn.Linear)]

        self.assertTrue(actor_linears)
        self.assertTrue(all(not hasattr(layer, "weight_orig") for layer in actor_linears))
        self.assertFalse(model.actor_spectral_norm)


if __name__ == "__main__":
    unittest.main()
