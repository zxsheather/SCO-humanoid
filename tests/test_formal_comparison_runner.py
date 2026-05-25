from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import run_formal_comparison as formal_comparison  # noqa: E402


class FormalComparisonRunnerTests(unittest.TestCase):
    def build_args(self, **overrides):
        defaults = {
            "humanoid_gym_root": None,
            "train_num_envs": None,
            "eval_num_envs": None,
            "max_iterations": None,
            "episodes": None,
            "rl_device": None,
            "sim_device": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_train_command_uses_sweep_defaults(self) -> None:
        args = self.build_args()

        command = formal_comparison.build_train_command(
            formal_comparison.resolve_config_path("configs/methods/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp.json"),
            "action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed11",
            11,
            args,
            train_num_envs=512,
            max_iterations=400,
        )

        self.assertIn("--num-envs=512", command)
        self.assertIn("--max-iterations=400", command)

    def test_cli_overrides_still_win_over_sweep_defaults(self) -> None:
        args = self.build_args(train_num_envs=64, max_iterations=20)

        command = formal_comparison.build_train_command(
            formal_comparison.resolve_config_path("configs/methods/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp.json"),
            "action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed11",
            11,
            args,
            train_num_envs=512,
            max_iterations=400,
        )

        self.assertIn("--num-envs=64", command)
        self.assertIn("--max-iterations=20", command)
        self.assertNotIn("--num-envs=512", command)
        self.assertNotIn("--max-iterations=400", command)

    def test_evaluate_command_uses_sweep_defaults(self) -> None:
        args = self.build_args()

        command = formal_comparison.build_evaluate_command(
            formal_comparison.resolve_config_path("configs/methods/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp.json"),
            "action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed11",
            "demo_load_run",
            11,
            args,
            eval_num_envs=32,
            episodes=20,
        )

        self.assertIn("--num-envs", command)
        self.assertIn("32", command)
        self.assertIn("--episodes", command)
        self.assertIn("20", command)

    def test_latest_metrics_row_picks_highest_checkpoint(self) -> None:
        rows = [
            {"checkpoint": 100, "fall_rate": 1.0},
            {"checkpoint": 400, "fall_rate": 0.7},
            {"checkpoint": 300, "fall_rate": 0.8},
        ]

        row = formal_comparison.latest_metrics_row(rows)

        self.assertIsNotNone(row)
        self.assertEqual(row["checkpoint"], 400)

    def test_metrics_snapshot_from_row_extracts_comparison_metrics(self) -> None:
        row = {
            "checkpoint": 400,
            "velocity_tracking_error_mean": 1.0,
            "joint_acceleration_l2_mean": 200.0,
            "action_jitter_l2_mean": 0.3,
            "episode_return_mean": 40.0,
            "fall_rate": 0.7,
            "unused_metric": 123.0,
        }

        snapshot = formal_comparison.metrics_snapshot_from_row(row)

        self.assertEqual(
            snapshot,
            {
                "velocity_tracking_error_mean": 1.0,
                "joint_acceleration_l2_mean": 200.0,
                "action_jitter_l2_mean": 0.3,
                "episode_return_mean": 40.0,
                "fall_rate": 0.7,
            },
        )


if __name__ == "__main__":
    unittest.main()
