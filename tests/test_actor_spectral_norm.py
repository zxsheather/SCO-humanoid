from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import torch
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
        self.assertFalse(model.actor_orthogonal_parametrization)

    def test_actor_orthogonal_parametrization_wraps_actor_layers_and_applies_output_gain(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
            actor_orthogonal_parametrization=True,
            actor_orthogonal_layer_scope="all",
            actor_output_gain=0.5,
        )

        actor_linears = [module for module in model.actor if isinstance(module, nn.Linear)]

        self.assertTrue(actor_linears)
        self.assertTrue(all(hasattr(layer, "parametrizations") and "weight" in layer.parametrizations for layer in actor_linears))
        self.assertEqual(actor_linears[-1].output_scale, 0.5)
        self.assertTrue(all(layer.output_scale == 1.0 for layer in actor_linears[:-1]))
        self.assertTrue(model.actor_orthogonal_parametrization)
        self.assertAlmostEqual(model.actor_output_gain, 0.5)

    def test_actor_orthogonal_parametrization_and_spectral_norm_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
            self.actor_critic_module.ActorCritic(
                num_actor_obs=8,
                num_critic_obs=8,
                num_actions=4,
                actor_hidden_dims=[16, 16],
                critic_hidden_dims=[16, 16],
                actor_spectral_norm=True,
                actor_orthogonal_parametrization=True,
            )

    def test_actor_layer_norm_wraps_hidden_layers_only_by_default(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
            actor_layer_norm=True,
        )

        actor_layer_norms = [module for module in model.actor if isinstance(module, nn.LayerNorm)]

        self.assertEqual(len(actor_layer_norms), 2)
        self.assertEqual(actor_layer_norms[0].normalized_shape, (16,))
        self.assertEqual(actor_layer_norms[1].normalized_shape, (16,))
        self.assertTrue(model.actor_layer_norm)
        self.assertFalse(model.actor_layer_norm_output_layer)
        self.assertEqual(model.actor_layer_norm_layer_scope, "hidden")

    def test_actor_layer_norm_can_wrap_first_hidden_layer_only(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16, 16],
            critic_hidden_dims=[16, 16],
            actor_layer_norm=True,
            actor_layer_norm_layer_scope="first_hidden",
        )

        actor_layer_norms = [module for module in model.actor if isinstance(module, nn.LayerNorm)]

        self.assertEqual(len(actor_layer_norms), 1)
        self.assertEqual(actor_layer_norms[0].normalized_shape, (16,))
        self.assertEqual(model.actor_layer_norm_layer_scope, "first_hidden")

    def test_actor_layer_norm_can_wrap_output_layer_when_requested(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
            actor_layer_norm=True,
            actor_layer_norm_output_layer=True,
            actor_layer_norm_layer_scope="all",
        )

        actor_layer_norms = [module for module in model.actor if isinstance(module, nn.LayerNorm)]

        self.assertEqual(len(actor_layer_norms), 3)
        self.assertEqual(actor_layer_norms[-1].normalized_shape, (4,))
        self.assertTrue(model.actor_layer_norm_output_layer)

    def test_actor_layer_norm_rejects_unknown_layer_scope(self):
        with self.assertRaises(ValueError):
            self.actor_critic_module.ActorCritic(
                num_actor_obs=8,
                num_critic_obs=8,
                num_actions=4,
                actor_hidden_dims=[16, 16],
                critic_hidden_dims=[16, 16],
                actor_layer_norm=True,
                actor_layer_norm_layer_scope="middle",
            )

    def test_action_output_scale_affects_inference_mean(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
        )
        obs = torch.randn(3, 8)

        base_mean = model.act_inference(obs)
        model.set_action_output_scale(0.5)
        scaled_mean = model.act_inference(obs)

        self.assertAlmostEqual(model.get_action_output_scale(), 0.5)
        self.assertTrue(torch.allclose(scaled_mean, base_mean * 0.5, atol=1e-6, rtol=1e-5))

    def test_policy_output_scale_affects_distribution_mean_and_std(self):
        model = self.actor_critic_module.ActorCritic(
            num_actor_obs=8,
            num_critic_obs=8,
            num_actions=4,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
        )
        obs = torch.randn(3, 8)

        model.update_distribution(obs)
        base_mean = model.action_mean.clone()
        base_std = model.action_std.clone()

        model.set_policy_output_scale(0.5)
        model.update_distribution(obs)

        self.assertAlmostEqual(model.get_policy_output_scale(), 0.5)
        self.assertAlmostEqual(model.get_action_std_scale(), 0.5)
        self.assertTrue(torch.allclose(model.action_mean, base_mean * 0.5, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(model.action_std, base_std * 0.5, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
