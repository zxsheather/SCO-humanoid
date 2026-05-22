from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from tests._sc_ppo_loader import load_sc_ppo_module


class LinearActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [2.0, 0.0],
                        [0.0, 3.0],
                        [0.0, 4.0],
                    ]
                )
            )

    def to(self, device):
        super().to(device)
        return self

    def act_inference(self, obs):
        return self.linear(obs)


class AnisotropicConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc_ppo_module = load_sc_ppo_module()

    def build_algo(self, constraint):
        return self.sc_ppo_module.SCPPO(
            LinearActorCritic(),
            constraint=constraint,
            device="cpu",
        )

    def test_unit_weights_match_scalar_cost(self):
        obs_batch = torch.randn(6, 2)
        scalar_algo = self.build_algo({"enabled": True, "subsample_obs": 0})
        anisotropic_algo = self.build_algo(
            {
                "enabled": True,
                "subsample_obs": 0,
                "anisotropic_enabled": True,
                "anisotropic_group_slices": [[0, 2], [2, 4]],
                "anisotropic_group_weights": [1.0, 1.0],
            }
        )

        scalar_stats = scalar_algo._local_sensitivity_metrics(obs_batch)
        anisotropic_stats = anisotropic_algo._local_sensitivity_metrics(obs_batch)

        self.assertAlmostEqual(
            scalar_stats["cost_mean"].item(),
            anisotropic_stats["cost_mean"].item(),
            places=6,
        )
        self.assertAlmostEqual(
            scalar_stats["cost_quantile"].item(),
            anisotropic_stats["cost_quantile"].item(),
            places=6,
        )
        self.assertEqual(anisotropic_stats["effective_mode"], "anisotropic_group_weighted")

    def test_weighted_groups_shift_effective_cost_and_expose_group_stats(self):
        obs_batch = torch.randn(6, 2)
        algo = self.build_algo(
            {
                "enabled": True,
                "subsample_obs": 0,
                "cost_aggregation": "mean",
                "anisotropic_enabled": True,
                "anisotropic_group_slices": [[0, 2], [2, 4]],
                "anisotropic_group_weights": [4.0, 1.0],
                "anisotropic_group_labels": ["x_axis", "y_axis"],
            }
        )

        stats = algo._local_sensitivity_metrics(obs_batch)

        self.assertAlmostEqual(stats["legacy_cost_mean"].item(), torch.sqrt(torch.tensor(30.0)).item(), places=6)
        self.assertAlmostEqual(stats["cost_mean"].item(), torch.sqrt(torch.tensor(45.0)).item(), places=6)
        self.assertEqual(stats["group_labels"], ["x_axis", "y_axis"])
        self.assertEqual(stats["group_slices"], [[0, 2], [2, 4]])
        self.assertEqual(stats["group_weights"], [4.0, 1.0])
        self.assertEqual(len(stats["group_cost_mean"]), 2)
        self.assertAlmostEqual(stats["group_cost_mean"][0].item(), torch.sqrt(torch.tensor(5.0)).item(), places=6)
        self.assertAlmostEqual(stats["group_cost_mean"][1].item(), 5.0, places=6)

    def test_legacy_guard_prevents_masked_groups_from_under_reporting_update_cost(self):
        obs_batch = torch.randn(6, 2)
        base_constraint = {
            "enabled": True,
            "subsample_obs": 0,
            "cost_aggregation": "mean",
            "anisotropic_enabled": True,
            "anisotropic_group_slices": [[0, 2], [2, 4]],
            "anisotropic_group_weights": [1.0, 0.0],
        }
        unguarded_algo = self.build_algo(base_constraint)
        guarded_algo = self.build_algo(
            {
                **base_constraint,
                "legacy_guard_mode": "max_with_legacy",
            }
        )

        unguarded_stats = unguarded_algo._local_sensitivity_metrics(obs_batch)
        guarded_stats = guarded_algo._local_sensitivity_metrics(obs_batch)

        self.assertAlmostEqual(
            unguarded_stats["effective_cost_for_update"].item(),
            torch.sqrt(torch.tensor(5.0)).item(),
            places=6,
        )
        self.assertAlmostEqual(
            unguarded_stats["cost_for_update"].item(),
            unguarded_stats["effective_cost_for_update"].item(),
            places=6,
        )
        self.assertAlmostEqual(
            guarded_stats["legacy_cost_for_update"].item(),
            torch.sqrt(torch.tensor(30.0)).item(),
            places=6,
        )
        self.assertAlmostEqual(
            guarded_stats["effective_cost_for_update"].item(),
            torch.sqrt(torch.tensor(5.0)).item(),
            places=6,
        )
        self.assertAlmostEqual(
            guarded_stats["cost_for_update"].item(),
            guarded_stats["legacy_cost_for_update"].item(),
            places=6,
        )

    def test_group_partition_must_cover_action_dimension(self):
        obs_batch = torch.randn(4, 2)
        algo = self.build_algo(
            {
                "enabled": True,
                "subsample_obs": 0,
                "anisotropic_enabled": True,
                "anisotropic_group_slices": [[0, 2]],
                "anisotropic_group_weights": [1.0],
            }
        )

        with self.assertRaises(ValueError):
            algo._local_sensitivity_metrics(obs_batch)

    def test_positive_part_penalty_mode_clamps_negative_error(self):
        signed_algo = self.build_algo({"enabled": True, "subsample_obs": 0, "penalty_mode": "signed"})
        positive_part_algo = self.build_algo(
            {"enabled": True, "subsample_obs": 0, "penalty_mode": "positive_part"}
        )

        negative_error = torch.tensor(-0.5)
        positive_error = torch.tensor(0.25)

        self.assertAlmostEqual(signed_algo._constraint_penalty_error(negative_error).item(), -0.5, places=6)
        self.assertAlmostEqual(
            positive_part_algo._constraint_penalty_error(negative_error).item(),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            positive_part_algo._constraint_penalty_error(positive_error).item(),
            0.25,
            places=6,
        )

    def test_positive_part_update_error_mode_clamps_negative_error(self):
        signed_algo = self.build_algo({"enabled": True, "subsample_obs": 0, "update_error_mode": "signed"})
        positive_part_algo = self.build_algo(
            {"enabled": True, "subsample_obs": 0, "update_error_mode": "positive_part"}
        )

        negative_error = torch.tensor(-0.5)
        positive_error = torch.tensor(0.25)

        self.assertAlmostEqual(signed_algo._constraint_update_error(negative_error).item(), -0.5, places=6)
        self.assertAlmostEqual(
            positive_part_algo._constraint_update_error(negative_error).item(),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            positive_part_algo._constraint_update_error(positive_error).item(),
            0.25,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
