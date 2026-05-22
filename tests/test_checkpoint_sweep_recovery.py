from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import evaluate_checkpoint_sweep as checkpoint_sweep  # noqa: E402


class CheckpointSweepRecoveryTests(unittest.TestCase):
    def test_recover_checkpoint_metrics_requires_matching_manifest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint_sweep.write_json(
                output_dir / "metrics.json",
                {
                    "velocity_tracking_error_mean": 0.5,
                    "joint_acceleration_l2_mean": 100.0,
                },
            )
            checkpoint_sweep.write_json(
                output_dir / "manifest.json",
                {
                    "checkpoint_path": ".external/humanoid-gym/logs/example/model_350.pt",
                },
            )

            recovered = checkpoint_sweep.recover_checkpoint_metrics(output_dir, 350)
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["velocity_tracking_error_mean"], 0.5)

            rejected = checkpoint_sweep.recover_checkpoint_metrics(output_dir, 300)
            self.assertIsNone(rejected)

    def test_checkpoint_train_constraint_metrics_reads_latest_stats_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model_25.pt"
            torch.save(
                {
                    "alg_extra_state_dict": {
                        "latest_stats": {
                            "constraint_objective": "action_rate",
                            "constraint_cost_mean": 0.18,
                            "constraint_cost_update": 0.22,
                            "lagrange_multiplier": 0.5,
                            "policy_local_sensitivity_cost_mean": 0.42,
                            "action_rate_cost_mean": 0.18,
                            "constraint_effective_mode": "anisotropic_group_weighted",
                            "constraint_legacy_guard_mode": "max_with_legacy",
                        },
                        "constraint_trace": [{}, {}, {}],
                    }
                },
                checkpoint_path,
            )

            metrics = checkpoint_sweep.checkpoint_train_constraint_metrics(checkpoint_path)

            self.assertEqual(metrics["train_constraint_objective"], "action_rate")
            self.assertEqual(metrics["train_constraint_cost_mean"], 0.18)
            self.assertEqual(metrics["train_constraint_cost_update"], 0.22)
            self.assertEqual(metrics["train_lagrange_multiplier"], 0.5)
            self.assertEqual(metrics["train_policy_local_sensitivity_cost_mean"], 0.42)
            self.assertEqual(metrics["train_action_rate_cost_mean"], 0.18)
            self.assertEqual(metrics["train_constraint_effective_mode"], "anisotropic_group_weighted")
            self.assertEqual(metrics["train_constraint_legacy_guard_mode"], "max_with_legacy")
            self.assertEqual(metrics["train_constraint_trace_length"], 3)


if __name__ == "__main__":
    unittest.main()
