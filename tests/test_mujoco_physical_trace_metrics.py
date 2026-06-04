from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from _mujoco_physical_trace_metrics import compute_episode_physical_metrics  # noqa: E402


class MujocoPhysicalTraceMetricTests(unittest.TestCase):
    def test_episode_physical_metrics_match_simple_hand_computation(self) -> None:
        episode = {
            "fell": False,
            "trace": [
                {
                    "applied_control": [1.0, -2.0],
                    "joint_velocity": [0.5, -0.25],
                    "base_linear_velocity": [0.4, 0.0, 0.0],
                },
                {
                    "applied_control": [3.0, 4.0],
                    "joint_velocity": [0.5, 0.5],
                    "base_linear_velocity": [0.6, 0.0, 0.0],
                },
            ],
        }

        metrics = compute_episode_physical_metrics(episode, control_dt=0.1)

        expected_rms = math.sqrt((1.0**2 + (-2.0) ** 2 + 3.0**2 + 4.0**2) / 4.0)
        expected_tau_l2 = (math.sqrt(5.0) + 5.0) / 2.0
        expected_abs_power_mean = ((abs(1.0 * 0.5) + abs(-2.0 * -0.25)) + (abs(3.0 * 0.5) + abs(4.0 * 0.5))) / 2.0
        expected_pos_power_mean = ((1.0 * 0.5 + (-2.0 * -0.25)) + (3.0 * 0.5 + 4.0 * 0.5)) / 2.0
        expected_energy_abs = (1.0 + 3.5) * 0.1
        expected_distance = (0.4 + 0.6) * 0.1

        self.assertAlmostEqual(metrics["control_torque_rms"], expected_rms)
        self.assertAlmostEqual(metrics["control_torque_l2_mean"], expected_tau_l2)
        self.assertAlmostEqual(metrics["mechanical_power_abs_mean"], expected_abs_power_mean)
        self.assertAlmostEqual(metrics["mechanical_power_positive_mean"], expected_pos_power_mean)
        self.assertAlmostEqual(metrics["mechanical_energy_abs"], expected_energy_abs)
        self.assertAlmostEqual(metrics["forward_distance_abs"], expected_distance)
        self.assertAlmostEqual(metrics["mechanical_cot_proxy"], expected_energy_abs / expected_distance)

    def test_smaller_controls_reduce_torque_and_power_metrics(self) -> None:
        smooth = {
            "trace": [
                {
                    "applied_control": [0.5, 0.5],
                    "joint_velocity": [0.2, 0.2],
                    "base_linear_velocity": [0.4, 0.0, 0.0],
                }
                for _ in range(4)
            ]
        }
        rough = {
            "trace": [
                {
                    "applied_control": [2.0, 2.0],
                    "joint_velocity": [0.2, 0.2],
                    "base_linear_velocity": [0.4, 0.0, 0.0],
                }
                for _ in range(4)
            ]
        }

        smooth_metrics = compute_episode_physical_metrics(smooth, control_dt=0.05)
        rough_metrics = compute_episode_physical_metrics(rough, control_dt=0.05)

        self.assertLess(smooth_metrics["control_torque_rms"], rough_metrics["control_torque_rms"])
        self.assertLess(smooth_metrics["mechanical_power_abs_mean"], rough_metrics["mechanical_power_abs_mean"])
        self.assertLess(smooth_metrics["mechanical_energy_abs"], rough_metrics["mechanical_energy_abs"])


if __name__ == "__main__":
    unittest.main()
