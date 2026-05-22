from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
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
                            "lagrange_multiplier": 0.5,
                            "policy_local_sensitivity_cost_mean": 0.42,
                            "constraint_effective_mode": "anisotropic_group_weighted",
                            "constraint_legacy_guard_mode": "max_with_legacy",
                        },
                        "constraint_trace": [{}, {}, {}],
                    }
                },
                checkpoint_path,
            )

            metrics = checkpoint_sweep.checkpoint_train_constraint_metrics(checkpoint_path)

            self.assertEqual(metrics["train_lagrange_multiplier"], 0.5)
            self.assertEqual(metrics["train_policy_local_sensitivity_cost_mean"], 0.42)
            self.assertEqual(metrics["train_constraint_effective_mode"], "anisotropic_group_weighted")
            self.assertEqual(metrics["train_constraint_legacy_guard_mode"], "max_with_legacy")
            self.assertEqual(metrics["train_constraint_trace_length"], 3)

    def test_build_evaluate_policy_command_forwards_trace_capture_options(self) -> None:
        command = checkpoint_sweep.build_evaluate_policy_command(
            config_path="configs/methods/example.json",
            run_name="demo",
            load_run="May22_demo",
            checkpoint=300,
            humanoid_gym_root="/tmp/humanoid-gym",
            num_envs=16,
            episodes=2,
            rl_device="cuda:0",
            sim_device="cuda:0",
            seed=11,
            capture_traces=True,
            trace_max_episodes=1,
            trace_max_steps=256,
        )

        self.assertIn("--capture-traces", command)
        self.assertIn("--trace-max-episodes", command)
        self.assertEqual(command[command.index("--trace-max-episodes") + 1], "1")
        self.assertIn("--trace-max-steps", command)
        self.assertEqual(command[command.index("--trace-max-steps") + 1], "256")

    def test_persist_trace_artifacts_writes_checkpoint_specific_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            trace_path = output_dir / "episode_traces.json"
            t = np.linspace(0.0, 1.0, 64)
            pos = np.stack([np.sin(2.0 * math.pi * t), np.cos(2.0 * math.pi * t)], axis=1)
            vel = np.gradient(pos, t[1] - t[0], axis=0)
            payload = {
                "metric_schema_version": 2,
                "run_name": "demo",
                "checkpoint_path": "model_300.pt",
                "episodes": [
                    {
                        "episode_index": 0,
                        "env_id": 0,
                        "episode_length": 64,
                        "fell": False,
                        "dt": float(t[1] - t[0]),
                        "dof_pos": pos.tolist(),
                        "dof_vel": vel.tolist(),
                        "truncated": False,
                    }
                ],
            }
            trace_path.write_text(json.dumps(payload), encoding="utf-8")

            artifacts = checkpoint_sweep.persist_trace_artifacts(
                output_dir,
                300,
                {
                    "trace_capture": {
                        "enabled": True,
                        "trace_path": str(trace_path),
                    }
                },
            )

            self.assertIsNotNone(artifacts)
            self.assertTrue((output_dir / "episode_traces_checkpoint_300.json").exists())
            self.assertTrue((output_dir / "behavior_smoothness_metrics_checkpoint_300.json").exists())
            self.assertIsNotNone(artifacts["summary"]["joint_position_ldlj_mean"]["mean"])
            self.assertIsNotNone(artifacts["summary"]["joint_velocity_sparc_mean"]["mean"])


if __name__ == "__main__":
    unittest.main()
