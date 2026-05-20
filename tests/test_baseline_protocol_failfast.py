from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import evaluate_checkpoint_sweep as checkpoint_sweep  # noqa: E402
import select_heuristic_sweep as heuristic_selection  # noqa: E402


class HeuristicSelectionCollapseTests(unittest.TestCase):
    def test_all_candidates_collapsed_when_every_fall_rate_is_one(self) -> None:
        records = [
            {"status": "complete", "fall_rate": 1.0},
            {"status": "complete", "fall_rate": 1.0},
            {"status": "complete", "fall_rate": 1.0},
        ]
        self.assertTrue(heuristic_selection.all_candidates_collapsed(records))

    def test_all_candidates_collapsed_is_false_when_any_candidate_survives(self) -> None:
        records = [
            {"status": "complete", "fall_rate": 1.0},
            {"status": "complete", "fall_rate": 0.95},
            {"status": "complete", "fall_rate": 1.0},
        ]
        self.assertFalse(heuristic_selection.all_candidates_collapsed(records))


class CheckpointSweepCollapseTests(unittest.TestCase):
    def test_selection_status_marks_all_collapsed_checkpoints(self) -> None:
        rows = [
            {"checkpoint": 0, "fall_rate": 1.0},
            {"checkpoint": 50, "fall_rate": 1.0},
            {"checkpoint": 100, "fall_rate": 1.0},
        ]
        self.assertEqual(checkpoint_sweep.selection_status(rows), "all_checkpoints_collapsed")

    def test_selection_status_marks_selected_when_any_checkpoint_survives(self) -> None:
        rows = [
            {"checkpoint": 0, "fall_rate": 1.0},
            {"checkpoint": 50, "fall_rate": 0.9},
            {"checkpoint": 100, "fall_rate": 1.0},
        ]
        self.assertEqual(checkpoint_sweep.selection_status(rows), "selected")

    def test_selected_row_prefers_task_valid_later_checkpoint(self) -> None:
        rows = [
            {
                "checkpoint": 0,
                "fall_rate": 1.0,
                "velocity_tracking_error_mean": 1.39910786151886,
                "joint_acceleration_l2_mean": 71.9741563796997,
                "action_jitter_l2_mean": 0.024284730665385723,
                "composite_score": 211.8849425315857,
            },
            {
                "checkpoint": 100,
                "fall_rate": 1.0,
                "velocity_tracking_error_mean": 1.3346443057060242,
                "joint_acceleration_l2_mean": 105.77372665405274,
                "action_jitter_l2_mean": 0.16251830533146858,
                "composite_score": 239.23815722465514,
            },
            {
                "checkpoint": 200,
                "fall_rate": 0.85,
                "velocity_tracking_error_mean": 1.0465339809656142,
                "joint_acceleration_l2_mean": 125.63904609680176,
                "action_jitter_l2_mean": 0.2138153374195099,
                "composite_score": 230.29244419336317,
            },
        ]
        best, task_floor = checkpoint_sweep.selected_row(rows)
        self.assertEqual(best["checkpoint"], 200)
        self.assertIsNotNone(task_floor)
        self.assertEqual(task_floor["reference_checkpoint"], 200)
        self.assertEqual(task_floor["eligible_checkpoints"], [200])

    def test_selected_row_preserves_current_scppo_choice(self) -> None:
        rows = [
            {
                "checkpoint": 0,
                "fall_rate": 1.0,
                "velocity_tracking_error_mean": 1.1851443946361542,
                "joint_acceleration_l2_mean": 102.20052585601806,
                "action_jitter_l2_mean": 0.018,
                "composite_score": 220.7149653196335,
            },
            {
                "checkpoint": 100,
                "fall_rate": 1.0,
                "velocity_tracking_error_mean": 1.0046475052833557,
                "joint_acceleration_l2_mean": 121.90033302307128,
                "action_jitter_l2_mean": 0.12,
                "composite_score": 222.36508355140686,
            },
            {
                "checkpoint": 200,
                "fall_rate": 0.8,
                "velocity_tracking_error_mean": 0.9117474764585495,
                "joint_acceleration_l2_mean": 155.14236068725586,
                "action_jitter_l2_mean": 0.19,
                "composite_score": 246.31710833311082,
            },
            {
                "checkpoint": 300,
                "fall_rate": 0.1,
                "velocity_tracking_error_mean": 0.563059039413929,
                "joint_acceleration_l2_mean": 106.09529037475586,
                "action_jitter_l2_mean": 0.23,
                "composite_score": 162.40119431614878,
            },
            {
                "checkpoint": 400,
                "fall_rate": 0.1,
                "velocity_tracking_error_mean": 0.5431461602449417,
                "joint_acceleration_l2_mean": 126.14202079772949,
                "action_jitter_l2_mean": 0.25,
                "composite_score": 180.45663682222365,
            },
        ]
        best, task_floor = checkpoint_sweep.selected_row(rows)
        self.assertEqual(best["checkpoint"], 300)
        self.assertIsNotNone(task_floor)
        self.assertEqual(task_floor["reference_checkpoint"], 400)
        self.assertEqual(task_floor["eligible_checkpoints"], [300, 400])


if __name__ == "__main__":
    unittest.main()
