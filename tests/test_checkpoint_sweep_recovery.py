from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
