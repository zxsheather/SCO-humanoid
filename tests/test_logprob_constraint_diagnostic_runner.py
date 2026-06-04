from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import run_logprob_constraint_diagnostic as diagnostic  # noqa: E402


class LogProbConstraintDiagnosticRunnerTests(unittest.TestCase):
    def build_args(self, **overrides):
        defaults = {
            "preset": "smoke",
            "run_name": None,
            "train_num_envs": None,
            "max_iterations": None,
            "save_interval": None,
            "eval_num_envs": None,
            "episodes": None,
            "checkpoints": None,
            "seed": None,
            "humanoid_gym_root": None,
            "rl_device": None,
            "sim_device": None,
            "analysis_root": "artifacts/analysis/logprob_constraint_diagnostic",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_default_run_name_includes_preset_and_seed(self) -> None:
        config = {"run_name": "logprob_constraint_tau_60_mean_pid_lower_bound_clamp_rough_terrain"}
        args = self.build_args()

        self.assertEqual(
            diagnostic.run_name_for(config, args),
            "logprob_constraint_tau_60_mean_pid_lower_bound_clamp_rough_terrain_smoke_seed23",
        )

    def test_train_command_uses_smoke_budget(self) -> None:
        args = self.build_args()
        command = diagnostic.build_train_command(
            diagnostic.DEFAULT_CONFIG,
            "logprob_constraint_tau_60_mean_pid_lower_bound_clamp_rough_terrain_smoke_seed23",
            args,
        )

        self.assertIn("--num-envs=1", command)
        self.assertIn("--max-iterations=1", command)
        self.assertIn("--seed=23", command)
        self.assertIn(
            "--config=configs/methods/logprob_constraint_tau_60_mean_pid_lower_bound_clamp.json",
            command,
        )

    def test_calibration_preset_requests_seed23_and_checkpoint50_save(self) -> None:
        args = self.build_args(preset="calibration")
        train_command = diagnostic.build_train_command(
            diagnostic.DEFAULT_CONFIG,
            "logprob_constraint_tau_60_mean_pid_lower_bound_clamp_rough_terrain_calibration_seed23",
            args,
        )
        config = diagnostic.load_config(diagnostic.DEFAULT_CONFIG)
        eval_command = diagnostic.build_evaluate_command(
            config,
            diagnostic.DEFAULT_CONFIG,
            "logprob_constraint_tau_60_mean_pid_lower_bound_clamp_rough_terrain_calibration_seed23",
            args,
        )

        self.assertIn("--num-envs=512", train_command)
        self.assertIn("--max-iterations=100", train_command)
        self.assertIn("--save-interval=50", train_command)
        self.assertIn("--seed=23", train_command)
        self.assertIn("--checkpoints", eval_command)
        self.assertEqual(eval_command[eval_command.index("--checkpoints") + 1], "0,50,100")

    def test_evaluate_command_uses_run_name_when_manifest_absent(self) -> None:
        args = self.build_args()
        config = diagnostic.load_config(diagnostic.DEFAULT_CONFIG)
        run_name = "logprob_constraint_tau_60_mean_pid_lower_bound_clamp_missing_manifest"
        command = diagnostic.build_evaluate_command(
            config,
            diagnostic.DEFAULT_CONFIG,
            run_name,
            args,
        )

        self.assertIn("--load-run", command)
        self.assertEqual(command[command.index("--load-run") + 1], run_name)
        self.assertNotIn("--checkpoints", command)
        self.assertIn("--episodes", command)
        self.assertEqual(command[command.index("--episodes") + 1], "1")

    def test_constraint_artifact_paths_follow_config_filenames(self) -> None:
        config = {
            "artifacts_root": "artifacts/methods/logprob_constraint_probe",
            "evaluation": {
                "constraint_logging": {
                    "sidecar_metrics_filename": "custom_constraint_metrics.json",
                    "multiplier_trace_filename": "custom_trace.json",
                }
            },
        }
        run_name = "demo"

        self.assertEqual(
            diagnostic.constraint_sidecar_path(config, run_name),
            REPO_ROOT / "artifacts" / "methods" / "logprob_constraint_probe" / run_name / "custom_constraint_metrics.json",
        )
        self.assertEqual(
            diagnostic.constraint_trace_path(config, run_name),
            REPO_ROOT / "artifacts" / "methods" / "logprob_constraint_probe" / run_name / "custom_trace.json",
        )

    def test_calibration_assessment_stops_for_effectively_inactive_constraint(self) -> None:
        assessment = diagnostic.calibration_recommendation(
            {
                "constraint_threshold": 6.0,
                "lagrange_multiplier_max": 5.0,
            },
            [
                {"lagrange_multiplier": 0.0, "logprob_gradient_cost_update": 0.2},
                {"lagrange_multiplier": 0.0, "logprob_gradient_cost_update": 0.3},
                {"lagrange_multiplier": 0.0, "logprob_gradient_cost_update": 0.4},
                {"lagrange_multiplier": 0.0, "logprob_gradient_cost_update": 0.5},
                {"lagrange_multiplier": 0.0, "logprob_gradient_cost_update": 0.6},
            ],
            {
                "selection_status": "selected",
                "rows": [
                    {"checkpoint": 0, "fall_rate": 0.5},
                    {"checkpoint": 50, "fall_rate": 0.4},
                    {"checkpoint": 100, "fall_rate": 0.3},
                ],
            },
        )

        self.assertEqual(assessment["cost_scale_reading"], "too_loose_below_tau")
        self.assertEqual(assessment["recommendation"], "stop_branch_here")

    def test_calibration_assessment_requests_three_seed_budget_when_on_scale(self) -> None:
        assessment = diagnostic.calibration_recommendation(
            {
                "constraint_threshold": 6.0,
                "lagrange_multiplier_max": 5.0,
            },
            [
                {"lagrange_multiplier": 0.0, "logprob_gradient_cost_update": 5.2},
                {"lagrange_multiplier": 0.1, "logprob_gradient_cost_update": 5.8},
                {"lagrange_multiplier": 0.2, "logprob_gradient_cost_update": 6.4},
                {"lagrange_multiplier": 0.2, "logprob_gradient_cost_update": 6.1},
                {"lagrange_multiplier": 0.3, "logprob_gradient_cost_update": 5.9},
            ],
            {
                "selection_status": "selected",
                "rows": [
                    {"checkpoint": 0, "fall_rate": 0.8},
                    {"checkpoint": 50, "fall_rate": 0.4},
                    {"checkpoint": 100, "fall_rate": 0.2},
                ],
            },
        )

        self.assertEqual(assessment["cost_scale_reading"], "on_scale")
        self.assertEqual(assessment["recommendation"], "request_three_seed_budget")


if __name__ == "__main__":
    unittest.main()
