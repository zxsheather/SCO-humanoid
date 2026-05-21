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

    def test_actor_spectral_norm_can_leave_output_layer_unwrapped(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
            actor_spectral_norm=True,
            actor_spectral_norm_output_layer=False,
        )

        actor_linears = [module for module in model.actor if isinstance(module, nn.Linear)]

        self.assertEqual(len(actor_linears), 3)
        self.assertTrue(all(hasattr(layer, "weight_orig") for layer in actor_linears[:-1]))
        self.assertFalse(hasattr(actor_linears[-1], "weight_orig"))
        self.assertTrue(model.actor_spectral_norm)
        self.assertFalse(model.actor_spectral_norm_output_layer)
        self.assertEqual(actor_linears[-1].output_scale, 1.0)

    def test_actor_spectral_norm_coeff_scales_wrapped_actor_layers(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
            actor_spectral_norm=True,
            actor_spectral_norm_coeff=2.0,
        )

        actor_linears = [module for module in model.actor if isinstance(module, nn.Linear)]

        self.assertTrue(all(hasattr(layer, "weight_orig") for layer in actor_linears))
        self.assertTrue(all(layer.output_scale == 2.0 for layer in actor_linears))

    def test_actor_spectral_norm_can_wrap_first_hidden_layer_only(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16, 16],
            critic_hidden_dims=[16, 16],
            actor_spectral_norm=True,
            actor_spectral_norm_layer_scope="first_hidden",
        )

        actor_linears = [module for module in model.actor if isinstance(module, nn.Linear)]

        self.assertEqual(len(actor_linears), 4)
        self.assertTrue(hasattr(actor_linears[0], "weight_orig"))
        self.assertTrue(all(not hasattr(layer, "weight_orig") for layer in actor_linears[1:]))
        self.assertEqual(model.actor_spectral_norm_layer_scope, "first_hidden")

    def test_actor_spectral_norm_rejects_unknown_layer_scope(self):
        with self.assertRaises(ValueError):
            self.actor_critic_module.ActorCritic(
                num_actor_obs=8,
                num_critic_obs=8,
                num_actions=4,
                actor_hidden_dims=[16, 16],
                critic_hidden_dims=[16, 16],
                actor_spectral_norm=True,
                actor_spectral_norm_layer_scope="middle",
            )

    def test_actor_spectral_norm_rejects_non_positive_coeff(self):
        with self.assertRaises(ValueError):
            self.actor_critic_module.ActorCritic(
                num_actor_obs=8,
                num_critic_obs=8,
                num_actions=4,
                actor_hidden_dims=[16, 16],
                critic_hidden_dims=[16, 16],
                actor_spectral_norm=True,
                actor_spectral_norm_coeff=0.0,
            )

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
