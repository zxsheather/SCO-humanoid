from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import _common as common  # noqa: E402
import run_sn_task_stabilized_diagnostic as task_stabilized  # noqa: E402


class SnTaskStabilizedRecipeConfigTests(unittest.TestCase):
    def test_config_preserves_scppo_and_enables_first_hidden_sn(self) -> None:
        config = common.load_config(task_stabilized.DEFAULT_CONFIG)
        train = config["overrides"]["train"]

        self.assertEqual(train["runner.algorithm_class_name"], "SCPPO")
        self.assertTrue(train["algorithm.constraint.enabled"])
        self.assertEqual(train["algorithm.constraint.threshold"], 3.8)
        self.assertTrue(train["policy.actor_spectral_norm"])
        self.assertFalse(train["policy.actor_spectral_norm_output_layer"])
        self.assertEqual(train["policy.actor_spectral_norm_layer_scope"], "first_hidden")
        self.assertEqual(train["policy.actor_spectral_norm_coeff"], 1.0)


class SnTaskStabilizedRunnerTests(unittest.TestCase):
    def build_args(self, **overrides):
        defaults = {
            "preset": "smoke",
            "run_name": None,
            "train_num_envs": None,
            "max_iterations": None,
            "eval_num_envs": None,
            "episodes": None,
            "seed": None,
            "humanoid_gym_root": None,
            "rl_device": None,
            "sim_device": None,
            "analysis_root": "artifacts/analysis/sn_task_stabilized_diagnostic",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_default_run_name_includes_preset_and_seed(self) -> None:
        config = common.load_config(task_stabilized.DEFAULT_CONFIG)
        args = self.build_args()

        self.assertEqual(
            task_stabilized.run_name_for(config, args),
            "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_sn_first_hidden_rough_terrain_smoke_seed123145",
        )

    def test_train_command_uses_hybrid_config(self) -> None:
        args = self.build_args(preset="short")
        command = task_stabilized.build_train_command(
            task_stabilized.DEFAULT_CONFIG,
            "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_sn_first_hidden_rough_terrain_short_seed123145",
            args,
        )

        self.assertIn("--num-envs=32", command)
        self.assertIn("--max-iterations=20", command)
        self.assertIn(
            "--config=configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_sn_first_hidden.json",
            command,
        )


if __name__ == "__main__":
    unittest.main()
