from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from _behavior_trace_metrics import (  # noqa: E402
    compute_episode_smoothness_metrics,
    compute_log_dimensionless_jerk,
    compute_sparc,
    should_capture_traces,
    trace_capture_config,
)


class BehaviorTraceMetricTests(unittest.TestCase):
    def test_trace_capture_defaults_to_disabled(self) -> None:
        cfg = trace_capture_config({})

        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["max_episodes"], 0)
        self.assertFalse(should_capture_traces(cfg))

    def test_ldlj_prefers_smooth_position_trace(self) -> None:
        dt = 0.02
        t = np.linspace(0.0, 2.0 * math.pi, 256)
        smooth = np.sin(t)[:, None]
        rough = np.sign(np.sin(8.0 * t))[:, None]

        smooth_metric = compute_log_dimensionless_jerk(smooth, dt)
        rough_metric = compute_log_dimensionless_jerk(rough, dt)

        self.assertIsNotNone(smooth_metric)
        self.assertIsNotNone(rough_metric)
        self.assertGreater(smooth_metric, rough_metric)

    def test_sparc_prefers_smooth_velocity_trace(self) -> None:
        dt = 0.02
        t = np.linspace(0.0, 2.0 * math.pi, 256)
        smooth = np.sin(t)[:, None]
        rough = (np.sin(t) + 0.4 * np.sin(11.0 * t))[:, None]

        smooth_metric = compute_sparc(smooth, dt)
        rough_metric = compute_sparc(rough, dt)

        self.assertIsNotNone(smooth_metric)
        self.assertIsNotNone(rough_metric)
        self.assertGreater(smooth_metric, rough_metric)

    def test_episode_metric_summary_uses_position_and_velocity_traces(self) -> None:
        dt = 0.02
        t = np.linspace(0.0, 1.0, 128)
        position = np.stack([np.sin(2.0 * math.pi * t), np.cos(2.0 * math.pi * t)], axis=1)
        velocity = np.gradient(position, dt, axis=0)

        metrics = compute_episode_smoothness_metrics(
            {
                "dt": dt,
                "dof_pos": position.tolist(),
                "dof_vel": velocity.tolist(),
            }
        )

        self.assertIn("joint_position_ldlj_mean", metrics)
        self.assertIn("joint_velocity_sparc_mean", metrics)
        self.assertIsNotNone(metrics["joint_position_ldlj_mean"])
        self.assertIsNotNone(metrics["joint_velocity_sparc_mean"])


if __name__ == "__main__":
    unittest.main()
